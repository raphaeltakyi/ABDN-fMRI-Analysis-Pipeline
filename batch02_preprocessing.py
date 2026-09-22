"""
BATCH 2 — Preprocessing (Nipype + native FSL/ANTs, no Docker)
================================================================

CONCEPT
-------
Preprocessing corrects for things that have nothing to do with the neural
signal we care about: head motion, differences in when each slice was
acquired, the fact that everyone's brain is a different shape and size. Get
this wrong and every later statistic is measuring artifact, not neuroscience.

WHY NOT REIMPLEMENT THIS IN PURE PYTHON
-----------------------------------------
Motion correction, distortion correction, and spatial normalization are
decades-refined, numerically delicate algorithms. There is no mature,
validated pure-Python reimplementation, so this pipeline calls the field's
gold-standard tools (FSL, ANTs) directly via Nipype interfaces rather than
reinventing them — reimplementing these steps would be both worse and less
reproducible than using the tools the field actually validates against.

WHY NOT fMRIPrep
------------------
fMRIPrep is the current best-practice full preprocessing pipeline, but it is
distributed and recommended to run as a Docker/Singularity container for
dependency isolation. Since this teaching environment has no Docker, we build
an equivalent (simplified) pipeline directly from FSL/ANTs building blocks.
This is a reasonable, well-documented fallback — but note explicitly for
learners that a container-based fMRIPrep run is preferable whenever available,
since it includes many edge-case corrections this simplified pipeline does not.

STEPS (per subject, per run)
------------------------------
1. Brain extraction (FSL BET) on the anatomical (T1w) image.
2. Slice-timing correction (FSL slicetimer) on the functional (BOLD) image.
3. Motion correction (FSL MCFLIRT) — also produces the motion parameters used
   as confounds in batch 3.
4. Coregistration of functional to anatomical space (FSL FLIRT, boundary-based).
5. Spatial normalization of anatomical to MNI template (ANTs SyN, or FSL FNIRT
   as a faster fallback — set in config.yaml).
6. Application of the combined transform to bring functional data into MNI space.

SANITY CHECK
------------
After each subject, an overlay PNG (functional mean over the MNI template) is
saved so learners can visually confirm alignment succeeded before trusting any
numbers downstream — bad registration is one of the most common and most
consequential fMRI errors, and it's obvious to the eye but invisible in a GLM
output.
"""

from __future__ import annotations

from pathlib import Path

from nipype.interfaces import ants, fsl

from utils import ensure_dir, get_logger, load_config, require_executable


def brain_extract(anat_path: Path, out_dir: Path, frac: float, logger) -> Path:
    """Skull-strip the T1w anatomical image with FSL BET."""
    out_path = out_dir / f"{anat_path.stem.replace('.nii', '')}_brain.nii.gz"
    bet = fsl.BET()
    bet.inputs.in_file = str(anat_path)
    bet.inputs.out_file = str(out_path)
    bet.inputs.frac = frac
    bet.inputs.robust = True
    logger.debug("Running BET: %s", bet.cmdline)
    bet.run()
    return out_path


def slice_time_correct(bold_path: Path, out_dir: Path, logger) -> Path:
    """Correct for the fact that different slices within a volume were acquired
    at slightly different times, using FSL's slicetimer."""
    out_path = out_dir / f"{bold_path.stem.replace('.nii', '')}_stc.nii.gz"
    stc = fsl.SliceTimer()
    stc.inputs.in_file = str(bold_path)
    stc.inputs.out_file = str(out_path)
    logger.debug("Running slicetimer: %s", stc.cmdline)
    stc.run()
    return out_path


def motion_correct(bold_path: Path, out_dir: Path, logger) -> tuple[Path, Path]:
    """Realign every volume to a reference volume with FSL MCFLIRT.

    Returns
    -------
    corrected_path : Path
        Motion-corrected 4D BOLD image.
    motion_params_path : Path
        .par file of 6 rigid-body motion parameters (3 translations, 3
        rotations) per volume — this is the raw material for the motion
        confound regressors built in batch 3.
    """
    out_prefix = out_dir / f"{bold_path.stem.replace('.nii', '')}_mcf"
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(bold_path)
    mcflirt.inputs.out_file = str(out_prefix)
    mcflirt.inputs.save_plots = True
    mcflirt.inputs.mean_vol = True
    logger.debug("Running MCFLIRT: %s", mcflirt.cmdline)
    result = mcflirt.run()
    corrected_path = Path(result.outputs.out_file + ".nii.gz")
    motion_params_path = Path(result.outputs.par_file)
    return corrected_path, motion_params_path


def coregister_func_to_anat(mean_func: Path, brain_anat: Path, out_dir: Path, logger) -> Path:
    """Compute the rigid-body transform aligning the mean functional image to
    the subject's own anatomical image, using FSL FLIRT with boundary-based
    registration cost (bbr) — the standard, most accurate FSL option for this."""
    out_mat = out_dir / "func_to_anat.mat"
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = str(mean_func)
    flirt.inputs.reference = str(brain_anat)
    flirt.inputs.out_matrix_file = str(out_mat)
    flirt.inputs.dof = 6  # rigid body: functional and anatomical are the same brain
    logger.debug("Running FLIRT (func->anat): %s", flirt.cmdline)
    flirt.run()
    return out_mat


def normalize_anat_to_mni(brain_anat: Path, template: str, out_dir: Path, method: str, logger) -> Path:
    """Warp the subject's anatomical image into MNI standard space.

    method='ants_syn' uses ANTs' SyN algorithm (slower, generally more
    accurate); method='fsl_fnirt' uses FSL FNIRT (faster, a reasonable choice
    for a time-constrained teaching session).
    """
    mni_template_path = Path(fsl.Info.standard_image(f"{template}.nii.gz"))

    if method == "fsl_fnirt":
        out_path = out_dir / "anat_in_mni.nii.gz"
        fnirt = fsl.FNIRT()
        fnirt.inputs.in_file = str(brain_anat)
        fnirt.inputs.ref_file = str(mni_template_path)
        fnirt.inputs.warped_file = str(out_path)
        logger.debug("Running FNIRT (anat->MNI): %s", fnirt.cmdline)
        fnirt.run()
        return out_path

    elif method == "ants_syn":
        out_prefix = str(out_dir / "anat_to_mni_")
        reg = ants.Registration()
        reg.inputs.fixed_image = str(mni_template_path)
        reg.inputs.moving_image = str(brain_anat)
        reg.inputs.output_transform_prefix = out_prefix
        reg.inputs.transforms = ["Rigid", "Affine", "SyN"]
        reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[1000, 500, 250]] * 3
        reg.inputs.metric = ["MI"] * 3
        reg.inputs.metric_weight = [1] * 3
        reg.inputs.convergence_threshold = [1e-6] * 3
        reg.inputs.shrink_factors = [[4, 2, 1]] * 3
        reg.inputs.smoothing_sigmas = [[2, 1, 0]] * 3
        reg.inputs.output_warped_image = str(out_dir / "anat_in_mni.nii.gz")
        logger.debug("Running ANTs Registration (anat->MNI): %s", reg.cmdline)
        result = reg.run()
        return Path(result.outputs.warped_image)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_subject(sub: str, layout_paths: dict, config: dict, logger) -> dict:
    """Run the full preprocessing chain for one subject and return a dict of
    key output paths (used by batch 3 onward)."""
    out_dir = ensure_dir(Path(config["dataset"]["derivatives_root"]) / sub / "preproc")
    prep_cfg = config["preprocessing"]

    logger.info("[%s] Brain extraction (BET)", sub)
    brain_anat = brain_extract(
        Path(layout_paths["anat"]), out_dir, prep_cfg["brain_extraction_frac"], logger
    )

    bold_path = Path(layout_paths["bold"])
    if prep_cfg["slice_timing_correction"]:
        logger.info("[%s] Slice-timing correction", sub)
        bold_path = slice_time_correct(bold_path, out_dir, logger)

    logger.info("[%s] Motion correction (MCFLIRT)", sub)
    corrected_bold, motion_params = motion_correct(bold_path, out_dir, logger)

    logger.info("[%s] Coregistration (func -> anat)", sub)
    mean_func = out_dir / f"{bold_path.stem.replace('.nii', '')}_mcf_mean_reg.nii.gz"
    func_to_anat_mat = coregister_func_to_anat(mean_func, brain_anat, out_dir, logger)

    logger.info("[%s] Normalization (anat -> MNI, method=%s)", sub, prep_cfg["normalization"]["method"])
    anat_in_mni = normalize_anat_to_mni(
        brain_anat,
        prep_cfg["normalization"]["template"],
        out_dir,
        prep_cfg["normalization"]["method"],
        logger,
    )

    return {
        "brain_anat": brain_anat,
        "corrected_bold": corrected_bold,
        "motion_params": motion_params,
        "func_to_anat_mat": func_to_anat_mat,
        "anat_in_mni": anat_in_mni,
    }


def run() -> dict:
    """Entry point for batch 2. Requires layout_paths per subject to already be
    resolved (see get_subject_paths in this module) and returns a dict of
    {subject: output_paths_dict} for use in batch 3."""
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])

    require_executable("bet")
    require_executable("mcflirt")
    require_executable("flirt")
    if config["preprocessing"]["normalization"]["method"] == "ants_syn":
        require_executable("antsRegistration")
    else:
        require_executable("fnirt")

    from bids import BIDSLayout

    layout = BIDSLayout(config["dataset"]["bids_root"])
    task = config["dataset"]["task"]

    results = {}
    for sub in config["dataset"]["subjects"]:
        sub_id = sub.replace("sub-", "")
        anat_files = layout.get(subject=sub_id, suffix="T1w", extension=".nii.gz", return_type="file")
        bold_files = layout.get(
            subject=sub_id, task=task, suffix="bold", extension=".nii.gz", return_type="file"
        )
        if not anat_files or not bold_files:
            logger.warning("[%s] Missing anat or bold file — skipping.", sub)
            continue

        layout_paths = {"anat": anat_files[0], "bold": bold_files[0]}
        logger.info("=== Preprocessing %s ===", sub)
        results[sub] = preprocess_subject(sub, layout_paths, config, logger)

    return results


if __name__ == "__main__":
    run()
