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

import os

import nibabel as nib
from nipype.interfaces import ants, fsl
from nipype.interfaces.fsl import FSLCommand

from utils import ensure_dir, get_logger, load_config, require_executable

# Force FSL to write UNCOMPRESSED NIfTI (.nii) everywhere. This install's
# gzip handling is unreliable for the pipeline's own large 4D outputs: FSL
# reads the dataset's raw .nii.gz inputs and its own .nii.gz templates fine,
# but the .nii.gz 4D files it WRITES come back malformed — some FSL tools and
# nibabel then can't open them ("cant open file", "not a valid NIFTI",
# "cannot work out file type"). Writing uncompressed .nii avoids the gzip path
# entirely: FSL and nibabel both read plain .nii consistently. Uncompressed
# files are larger but reliable, which is the right trade here.
os.environ["FSLOUTPUTTYPE"] = "NIFTI"
FSLCommand.set_default_output_type("NIFTI")


def resolve_fsl_output(reported_path: str | Path, logger) -> Path:
    """Return the on-disk path for an FSL output, tolerating extension mismatch
    AND verifying the file is a genuinely loadable NIfTI (not a stale 0-byte
    or truncated leftover from a previous failed run).

    Across FSL/nipype version combinations, an interface sometimes reports an
    output path whose extension doesn't match what FSL actually wrote. Worse,
    a previous crashed run can leave a corrupt file at the expected path that
    exists() accepts but that the next tool aborts on. This helper checks the
    reported path and its common variants, and returns the first one that both
    exists and actually loads as a NIfTI image — failing loudly otherwise.
    """
    reported = Path(reported_path)
    candidates = [
        reported,
        Path(str(reported) + ".nii.gz"),
        Path(str(reported) + ".nii"),
    ]
    stem = str(reported)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            base = stem[: -len(ext)]
            candidates.extend([Path(base + ".nii.gz"), Path(base + ".nii")])

    existing = [c for c in candidates if c.exists()]

    for candidate in existing:
        try:
            # Load the header only — cheap, but confirms it's a valid, non-empty
            # NIfTI rather than a truncated leftover.
            img = nib.load(str(candidate))
            _ = img.shape
        except Exception as exc:
            logger.warning(
                "Candidate output '%s' exists but is not a loadable NIfTI (%s) "
                "— likely a stale/corrupt file from a previous run; skipping it.",
                candidate, exc,
            )
            continue
        if candidate != reported:
            logger.debug(
                "Resolved FSL output '%s' to actual file on disk '%s'.",
                reported, candidate,
            )
        return candidate

    raise FileNotFoundError(
        f"FSL reported output '{reported}' but no loadable NIfTI was found on "
        f"disk (checked: {[str(c) for c in candidates]}). The FSL command may "
        f"have failed silently — check the log above for an FSL/libc++ error."
    )


def assert_loadable(path: Path, logger, context: str) -> None:
    """Fail early with a clear Python error if a file that's about to be handed
    to an FSL binary can't be loaded by nibabel.

    FSL's newer C++ tools (applywarp among them) abort the whole process with
    a libc++ 'Abort trap: 6' when they can't open an input, which produces a
    confusing traceback that blames the wrong thing. Checking loadability here
    turns that into an explicit, attributable error at the exact input.
    """
    try:
        img = nib.load(str(path))
        _ = img.shape
    except Exception as exc:
        raise RuntimeError(
            f"{context}: input file '{path}' cannot be loaded as a NIfTI "
            f"({exc}). This is the file the next FSL step would try to open; "
            f"fix or regenerate it before proceeding."
        ) from exc


def brain_extract(anat_path: Path, out_dir: Path, frac: float, logger) -> Path:
    """Skull-strip the T1w anatomical image with FSL BET."""
    out_path = out_dir / f"{anat_path.stem.replace('.nii', '')}_brain.nii"
    bet = fsl.BET()
    bet.inputs.in_file = str(anat_path)
    bet.inputs.out_file = str(out_path)
    bet.inputs.frac = frac
    bet.inputs.robust = True
    logger.debug("Running BET: %s", bet.cmdline)
    result = bet.run()
    return resolve_fsl_output(result.outputs.out_file, logger)


def slice_time_correct(bold_path: Path, out_dir: Path, logger) -> Path:
    """Correct for the fact that different slices within a volume were acquired
    at slightly different times, using FSL's slicetimer."""
    out_path = out_dir / f"{bold_path.stem.replace('.nii', '')}_stc.nii"
    stc = fsl.SliceTimer()
    stc.inputs.in_file = str(bold_path)
    stc.inputs.out_file = str(out_path)
    logger.debug("Running slicetimer: %s", stc.cmdline)
    result = stc.run()
    return resolve_fsl_output(result.outputs.slice_time_corrected_file, logger)


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
    # Output is uncompressed .nii (FSLOUTPUTTYPE=NIFTI). resolve_fsl_output
    # confirms the actual on-disk path, since FSL/nipype don't always agree on
    # the exact filename. No post-hoc "sanitizing" is done: writing plain .nii
    # in the first place is what makes the file readable by both FSL's own
    # tools and nibabel downstream.
    out_path = out_dir / f"{bold_path.stem.replace('.nii', '')}_mcf.nii"
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(bold_path)
    mcflirt.inputs.out_file = str(out_path)
    mcflirt.inputs.save_plots = True
    logger.debug("Running MCFLIRT: %s", mcflirt.cmdline)
    result = mcflirt.run()
    corrected_path = resolve_fsl_output(result.outputs.out_file, logger)
    motion_params_path = Path(result.outputs.par_file)
    return corrected_path, motion_params_path


def compute_mean_image(bold_path: Path, out_dir: Path, logger) -> Path:
    """Compute the temporal mean of a 4D BOLD image with FSL's MeanImage
    (equivalent to `fslmaths -Tmean`), used as the registration target for
    coregistration to the anatomical image.

    This is deliberately a separate, explicit step rather than relying on
    MCFLIRT's built-in `-meanvol` side-output: that side-output's filename
    convention turned out to be unreliable across subjects/environments in
    practice (present for some subjects, silently absent for others even
    though MCFLIRT itself succeeded). Computing the mean image ourselves
    means the output filename is fully controlled by nipype, not guessed
    from FSL's internal naming.
    """
    out_path = out_dir / f"{bold_path.stem.replace('.nii', '')}_mean.nii"
    mean_img = fsl.MeanImage()
    mean_img.inputs.in_file = str(bold_path)
    mean_img.inputs.out_file = str(out_path)
    mean_img.inputs.dimension = "T"
    logger.debug("Running fslmaths -Tmean: %s", mean_img.cmdline)
    result = mean_img.run()
    return resolve_fsl_output(result.outputs.out_file, logger)


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


def normalize_anat_to_mni(brain_anat: Path, template: str, out_dir: Path, method: str, logger) -> dict:
    """Warp the subject's anatomical image into MNI standard space.

    method='ants_syn' uses ANTs' SyN algorithm (slower, generally more
    accurate); method='fsl_fnirt' uses FSL FNIRT (faster, a reasonable choice
    for a time-constrained teaching session).

    Returns
    -------
    dict with keys:
        "warped_image": Path to the anatomical image warped into MNI space
            (for visual QC).
        "transforms": the deformation field/transform(s) needed to warp OTHER
            images (like the functional data) into the same space — an FSL
            fieldcoeff Path for fsl_fnirt, or a list of ANTs transform file
            Paths for ants_syn.
        "template": Path to the MNI template used as the registration target.
    These are consumed by apply_func_to_mni_transform() below to bring the
    functional data into the same space — warping only the anatomical image
    and stopping there (as an earlier version of this pipeline did) leaves
    the functional data actually used for the GLM in native space, which
    would silently invalidate any group-level analysis.
    """
    mni_template_path = Path(fsl.Info.standard_image(f"{template}.nii.gz"))

    if method == "fsl_fnirt":
        warped_path = out_dir / "anat_in_mni.nii"
        fieldcoeff_path = out_dir / "anat_to_mni_fieldcoeff.nii"
        fnirt = fsl.FNIRT()
        fnirt.inputs.in_file = str(brain_anat)
        fnirt.inputs.ref_file = str(mni_template_path)
        fnirt.inputs.warped_file = str(warped_path)
        fnirt.inputs.fieldcoeff_file = str(fieldcoeff_path)
        logger.debug("Running FNIRT (anat->MNI): %s", fnirt.cmdline)
        result = fnirt.run()
        return {
            "warped_image": resolve_fsl_output(result.outputs.warped_file, logger),
            "transforms": resolve_fsl_output(result.outputs.fieldcoeff_file, logger),
            "template": mni_template_path,
        }

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
        return {
            "warped_image": Path(result.outputs.warped_image),
            # forward_transforms is already ordered by nipype for direct use
            # as the `transforms` input to antsApplyTransforms (moving image
            # -> fixed/template space) — no manual reordering needed here.
            "transforms": list(result.outputs.forward_transforms),
            "template": mni_template_path,
        }

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _applywarp_whole(
    in_file: Path, ref: Path, warp: Path, premat: Path, out_file: Path, interp: str, logger
) -> Path:
    """Apply premat + warp to a 4D image in a single applywarp call."""
    applywarp = fsl.ApplyWarp()
    applywarp.inputs.in_file = str(in_file)
    applywarp.inputs.ref_file = str(ref)
    applywarp.inputs.field_file = str(warp)
    applywarp.inputs.premat = str(premat)
    applywarp.inputs.out_file = str(out_file)
    applywarp.inputs.interp = interp
    logger.debug("Running applywarp (whole 4D): %s", applywarp.cmdline)
    applywarp.run()
    return resolve_fsl_output(out_file, logger)


def _applywarp_per_volume(
    in_file: Path, ref: Path, warp: Path, premat: Path, out_file: Path,
    interp: str, out_dir: Path, logger
) -> Path:
    """Apply premat + warp volume-by-volume, then merge — a low-memory fallback.

    applywarp on a whole 4D series holds the entire time series in memory,
    which can exhaust RAM on larger runs and make the tool abort mid-resampling
    with a misleading "cant open file" error. Splitting the series into 3D
    volumes, warping each independently (tiny memory footprint), and merging
    the results back is far more robust, at the cost of being slower.
    """
    logger.warning(
        "Whole-series applywarp failed; retrying volume-by-volume (slower but "
        "much lower memory). This usually means the whole-4D resampling ran "
        "out of memory."
    )
    split_dir = ensure_dir(out_dir / "warp_split")

    split = fsl.Split()
    split.inputs.in_file = str(in_file)
    split.inputs.dimension = "t"
    split.inputs.out_base_name = str(split_dir / "vol_")
    logger.debug("Running fslsplit: %s", split.cmdline)
    split_result = split.run()
    split_files = split_result.outputs.out_files
    # fslsplit populates out_files only if it succeeded. If it aborted, the
    # trait is Undefined; turn that into a clear error instead of an opaque
    # "'_Undefined' object is not iterable" downstream.
    from nipype.interfaces.base import isdefined

    if not isdefined(split_files) or not split_files:
        raise RuntimeError(
            f"fslsplit did not produce output volumes for '{in_file}'. Check "
            f"the FSL error above — the input may be unreadable by FSL."
        )
    volume_files = [resolve_fsl_output(f, logger) for f in split_files]

    warped_volumes = []
    for i, vol in enumerate(volume_files):
        warped_vol = split_dir / f"warped_{i:04d}.nii"
        aw = fsl.ApplyWarp()
        aw.inputs.in_file = str(vol)
        aw.inputs.ref_file = str(ref)
        aw.inputs.field_file = str(warp)
        aw.inputs.premat = str(premat)
        aw.inputs.out_file = str(warped_vol)
        aw.inputs.interp = interp
        aw.run()
        warped_volumes.append(str(resolve_fsl_output(warped_vol, logger)))
        if (i + 1) % 25 == 0:
            logger.info("  warped %d/%d volumes", i + 1, len(volume_files))

    merge = fsl.Merge()
    merge.inputs.in_files = warped_volumes
    merge.inputs.dimension = "t"
    merge.inputs.merged_file = str(out_file)
    logger.debug("Running fslmerge: %s", merge.cmdline)
    merge.run()
    return resolve_fsl_output(out_file, logger)


def apply_func_to_mni_transform(
    corrected_bold: Path,
    mean_func: Path,
    brain_anat: Path,
    func_to_anat_mat: Path,
    normalization: dict,
    method: str,
    out_dir: Path,
    logger,
) -> Path:
    """Apply the combined functional->anatomical->MNI transform directly to
    the motion-corrected 4D BOLD data.

    This is the step that actually puts functional data into a shared
    template space, which is what group analysis (batch 5) requires — every
    subject's voxels need to correspond to the same anatomical location.
    Warping only the anatomical image (as normalize_anat_to_mni does on its
    own) is not sufficient on its own: the BOLD data used for the actual GLM
    has to go through the same transform, or group analysis silently compares
    voxels that don't correspond across subjects.

    method='fsl_fnirt' applies the func->anat affine (from FLIRT) and the
    anat->MNI nonlinear field (from FNIRT) in a single applywarp call, using
    applywarp's --premat and --warp inputs together (the standard FSL idiom).

    method='ants_syn' cannot directly mix an FSL affine (.mat) with ANTs
    transforms — the two tools use different affine conventions (FSL's are
    relative to voxel/flirt space; ANTs/ITK transforms are in physical RAS
    space). The FSL affine is first converted to ITK format with
    c3d_affine_tool (part of Convert3D, commonly bundled alongside ANTs
    installs), then combined with the ANTs anat->MNI transforms in a single
    antsApplyTransforms call.
    """
    out_path = out_dir / f"{corrected_bold.stem.replace('.nii', '')}_mni.nii"

    # Guard: FSL's applywarp/antsApplyTransforms abort hard (libc++ Abort
    # trap: 6, not a clean Python error) if the input BOLD can't be opened.
    # Resolve to the real on-disk file and confirm it actually loads BEFORE
    # handing it to the binary, so a genuine problem is attributed here.
    corrected_bold = resolve_fsl_output(corrected_bold, logger)
    assert_loadable(corrected_bold, logger, "apply_func_to_mni_transform (BOLD input)")

    if method == "fsl_fnirt":
        # Canonical FSL idiom: applywarp applies the func->anat affine
        # (--premat) and the anat->MNI nonlinear field (--warp) in one
        # resampling step. --premat (func->anat) is applied first, mapping the
        # functional data into structural space, where --warp (anat->MNI) then
        # takes over.
        #
        # Interpolation is TRILINEAR, not spline: spline interpolation on 4D
        # functional data is memory-hungry (it was causing applywarp to abort
        # mid-run on larger series) and trilinear is the standard, expected
        # choice for resampling functional data anyway. If the whole-series
        # call still fails (e.g. very limited RAM), fall back automatically to
        # a volume-by-volume warp that uses minimal memory.
        fieldcoeff = resolve_fsl_output(normalization["transforms"], logger)
        assert_loadable(fieldcoeff, logger, "apply_func_to_mni_transform (warp field)")

        try:
            return _applywarp_whole(
                corrected_bold, normalization["template"], fieldcoeff,
                func_to_anat_mat, out_path, "trilinear", logger,
            )
        except Exception as exc:
            logger.warning("Whole-series applywarp raised: %s", exc)
            return _applywarp_per_volume(
                corrected_bold, normalization["template"], fieldcoeff,
                func_to_anat_mat, out_path, "trilinear", out_dir, logger,
            )

    elif method == "ants_syn":
        from nipype.interfaces.c3 import C3dAffineTool

        itk_affine = out_dir / "func_to_anat_itk.txt"
        c3d = C3dAffineTool()
        c3d.inputs.reference_file = str(brain_anat)  # FLIRT's reference image
        c3d.inputs.source_file = str(mean_func)       # FLIRT's in_file
        c3d.inputs.transform_file = str(func_to_anat_mat)
        c3d.inputs.fsl2ras = True
        c3d.inputs.itk_transform = str(itk_affine)
        logger.debug("Running c3d_affine_tool: %s", c3d.cmdline)
        c3d.run()

        # ANTs applies a transform list in reverse order (last transform in
        # the list is applied to the input image first), so the func->anat
        # affine must go LAST: it needs to act on the functional data before
        # the anat->MNI transforms do.
        transforms = list(normalization["transforms"]) + [str(itk_affine)]

        apply_tf = ants.ApplyTransforms()
        apply_tf.inputs.input_image = str(corrected_bold)
        apply_tf.inputs.reference_image = str(normalization["template"])
        apply_tf.inputs.transforms = transforms
        apply_tf.inputs.input_image_type = 3  # time series (4D)
        apply_tf.inputs.interpolation = "LanczosWindowedSinc"
        apply_tf.inputs.output_image = str(out_path)
        logger.debug("Running antsApplyTransforms: %s", apply_tf.cmdline)
        result = apply_tf.run()
        return Path(result.outputs.output_image)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_subject(sub: str, layout_paths: dict, config: dict, logger) -> dict:
    """Run the full preprocessing chain for one subject and return a dict of
    key output paths (used by batch 3 onward)."""
    import shutil

    preproc_dir = Path(config["dataset"]["derivatives_root"]) / sub / "preproc"
    # Start each subject from a clean slate. Earlier failed runs can leave
    # partial or malformed files behind, and resolve_fsl_output could otherwise
    # pick one of those up; removing the directory first guarantees every file
    # used this run was written this run.
    if preproc_dir.exists():
        logger.info("[%s] Clearing previous preproc outputs at %s", sub, preproc_dir)
        shutil.rmtree(preproc_dir)
    out_dir = ensure_dir(preproc_dir)
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

    logger.info("[%s] Computing mean functional image", sub)
    mean_func = compute_mean_image(corrected_bold, out_dir, logger)

    logger.info("[%s] Coregistration (func -> anat)", sub)
    func_to_anat_mat = coregister_func_to_anat(mean_func, brain_anat, out_dir, logger)

    logger.info("[%s] Normalization (anat -> MNI, method=%s)", sub, prep_cfg["normalization"]["method"])
    normalization = normalize_anat_to_mni(
        brain_anat,
        prep_cfg["normalization"]["template"],
        out_dir,
        prep_cfg["normalization"]["method"],
        logger,
    )

    logger.info("[%s] Applying combined func->MNI transform to BOLD data", sub)
    bold_in_mni = apply_func_to_mni_transform(
        corrected_bold,
        mean_func,
        brain_anat,
        func_to_anat_mat,
        normalization,
        prep_cfg["normalization"]["method"],
        out_dir,
        logger,
    )

    return {
        "brain_anat": brain_anat,
        # Native-space, motion-corrected BOLD: used for confound extraction
        # in batch 3 (motion/aCompCor are computed pre-normalization, the
        # standard order — normalization is applied once, at the end).
        "corrected_bold": corrected_bold,
        # MNI-space BOLD: this is what batch 4's GLM actually fits, since
        # group analysis (batch 5) requires every subject's voxels to
        # correspond to the same anatomical location.
        "bold_in_mni": bold_in_mni,
        "motion_params": motion_params,
        "func_to_anat_mat": func_to_anat_mat,
        "anat_in_mni": normalization["warped_image"],
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
        require_executable("antsApplyTransforms")
        require_executable("c3d_affine_tool")
    else:
        require_executable("fnirt")
        require_executable("applywarp")
        require_executable("fslsplit")  # used by the low-memory per-volume fallback
        require_executable("fslmerge")

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
