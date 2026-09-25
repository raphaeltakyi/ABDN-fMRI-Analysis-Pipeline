"""
BATCH 3 — Confound Extraction & Denoising
============================================

CONCEPT
-------
Even after motion correction, residual motion-related signal remains in the
data, along with physiological noise (heartbeat, breathing) and scanner drift.
If we don't model these out, the GLM in batch 4 will attribute some of this
noise to our task, inflating false positives. This batch builds an explicit,
documented confound matrix rather than silently trusting default denoising —
what goes into that matrix is a real analytic decision, not a technicality.

Since we're not using fMRIPrep (which ships a rich confounds.tsv), this batch
computes the equivalent confounds directly:
  - 24-parameter motion model (6 realignment params + their derivatives +
    squares + squared derivatives) — captures both motion and spin-history
    effects.
  - Framewise displacement (FD) and DVARS, used to flag (not silently delete)
    high-motion volumes.
  - aCompCor: principal components of signal from white matter / CSF, which
    contains physiological/scanner noise but should contain no task signal.

WHAT THIS BATCH DOES
---------------------
1. Loads MCFLIRT motion parameters from batch 2.
2. Expands them to the 24-parameter model.
3. Computes FD and DVARS, flags scrubbing candidates per config thresholds.
4. Computes aCompCor components from WM/CSF masks.
5. Assembles and saves one confounds.tsv per subject — the same format
   convention fMRIPrep uses, so this is a drop-in replacement if you later
   switch to fMRIPrep.

SANITY CHECK
------------
A per-subject plot of FD over time is saved so learners can see at a glance
whether a subject has isolated motion spikes (fine, scrub a few volumes) or
pervasive motion (a candidate for exclusion) — a judgment call the pipeline
surfaces rather than making silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import ensure_dir, get_logger, load_config


def load_motion_params(par_file: Path) -> pd.DataFrame:
    """Load MCFLIRT's .par file: 6 columns (3 rotations in radians, then 3
    translations in mm), one row per volume."""
    cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    df = pd.read_csv(par_file, sep=r"\s+", header=None, names=cols)
    return df


def expand_to_24_param_model(motion_df: pd.DataFrame) -> pd.DataFrame:
    """Expand the raw 6 motion parameters to the standard 24-parameter model:
    params, their temporal derivatives, squares, and squared derivatives.

    This is a widely used, well-validated confound expansion (Friston et al.,
    1996) that captures motion effects beyond simple linear realignment.
    """
    df = motion_df.copy()
    derivatives = df.diff().fillna(0)
    derivatives.columns = [f"{c}_derivative1" for c in df.columns]

    squares = df.pow(2)
    squares.columns = [f"{c}_power2" for c in df.columns]

    squared_derivatives = derivatives.pow(2)
    squared_derivatives.columns = [f"{c}_power2" for c in derivatives.columns]

    return pd.concat([df, derivatives, squares, squared_derivatives], axis=1)


def compute_fd(motion_df: pd.DataFrame, head_radius_mm: float = 50.0) -> np.ndarray:
    """Compute framewise displacement (Power et al., 2012 formulation): the sum
    of absolute derivatives of the 6 motion parameters, with rotations
    converted to displacement on a sphere of the given head radius."""
    rot_rad = motion_df[["rot_x", "rot_y", "rot_z"]].values
    trans_mm = motion_df[["trans_x", "trans_y", "trans_z"]].values

    rot_mm = rot_rad * head_radius_mm  # arc length approximation
    combined = np.concatenate([trans_mm, rot_mm], axis=1)

    diffs = np.diff(combined, axis=0, prepend=combined[[0]])
    fd = np.sum(np.abs(diffs), axis=1)
    return fd


def compute_dvars(bold_data: np.ndarray) -> np.ndarray:
    """Compute standardized DVARS: the root-mean-square of the temporal
    derivative of the BOLD signal across brain voxels, then z-standardized so a
    unitless threshold (e.g. 1.5) is meaningful.

    Raw DVARS scales with the data's intensity units (raw BOLD can be in the
    hundreds or thousands), so comparing it to a fixed small threshold would
    flag almost every volume. Restricting to in-brain voxels and standardizing
    to (value - median) / SD gives the conventional unitless DVARS that a
    threshold around 1.5 is calibrated for.
    """
    # In-brain voxels only: background/air voxels are pure noise and would
    # dominate a whole-volume RMS.
    mean_img = bold_data.mean(axis=-1)
    brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)

    brain_ts = bold_data[brain_mask]  # (n_brain_voxels, t)
    diff = np.diff(brain_ts, axis=-1)  # (n_brain_voxels, t-1)
    dvars = np.sqrt(np.mean(diff**2, axis=0))  # (t-1,)

    # Standardize to unitless values comparable to a fixed threshold.
    median = np.median(dvars)
    sd = np.std(dvars)
    dvars_std = (dvars - median) / sd if sd > 0 else np.zeros_like(dvars)

    dvars_std = np.insert(dvars_std, 0, 0)  # align length with n_volumes
    return dvars_std


def compute_acompcor(bold_data: np.ndarray, wm_csf_mask: np.ndarray, n_components: int) -> pd.DataFrame:
    """Compute anatomical CompCor components: principal components of the BOLD
    timeseries within white matter + CSF voxels, which should contain
    physiological/scanner noise but no task-related grey-matter signal."""
    n_t = bold_data.shape[-1]
    voxel_ts = bold_data[wm_csf_mask].reshape(-1, n_t).T  # (t, n_voxels)
    voxel_ts = (voxel_ts - voxel_ts.mean(axis=0)) / (voxel_ts.std(axis=0) + 1e-8)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(voxel_ts)
    col_names = [f"acompcor_{i:02d}" for i in range(n_components)]
    return pd.DataFrame(components, columns=col_names)


def build_confounds_table(
    motion_params_path: Path,
    bold_data: np.ndarray,
    wm_csf_mask: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Assemble the full confounds table for one subject/run."""
    denoise_cfg = config["denoising"]["confound_strategy"]

    motion_df = load_motion_params(motion_params_path)
    motion_expanded = expand_to_24_param_model(motion_df)

    fd = compute_fd(motion_df)
    dvars = compute_dvars(bold_data)

    confounds = motion_expanded.copy()
    confounds["framewise_displacement"] = fd
    confounds["dvars"] = dvars

    if denoise_cfg["use_acompcor"]:
        acompcor_df = compute_acompcor(
            bold_data, wm_csf_mask, denoise_cfg["acompcor_n_components"]
        )
        confounds = pd.concat([confounds.reset_index(drop=True), acompcor_df], axis=1)

    confounds["motion_outlier"] = (
        fd > denoise_cfg["fd_scrubbing_threshold"]
    ) | (dvars > denoise_cfg["dvars_scrubbing_threshold"])

    return confounds


def run(preprocessing_outputs: dict) -> dict:
    """Entry point for batch 3.

    Parameters
    ----------
    preprocessing_outputs : dict
        The dict returned by batch02_preprocessing.run(), i.e.
        {subject: {"corrected_bold": Path, "motion_params": Path, ...}}.

    Returns
    -------
    dict
        {subject: Path to confounds.tsv}
    """
    import nibabel as nib

    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    denoise_cfg = config["denoising"]["confound_strategy"]

    confound_paths = {}
    for sub, paths in preprocessing_outputs.items():
        logger.info("[%s] Building confound matrix", sub)
        out_dir = ensure_dir(Path(config["dataset"]["derivatives_root"]) / sub / "confounds")

        bold_img = nib.load(paths["corrected_bold"])
        bold_data = bold_img.get_fdata()

        # NOTE: a real WM/CSF mask should come from segmenting the anatomical
        # image (e.g. FSL FAST) and resampling into functional space. This is
        # a placeholder threshold-based mask for teaching purposes — flagged
        # explicitly so learners know to replace it with a proper segmentation
        # in a real analysis.
        mean_img = bold_data.mean(axis=-1)
        wm_csf_mask = mean_img < np.percentile(mean_img[mean_img > 0], 30)
        logger.warning(
            "[%s] Using a placeholder intensity-threshold WM/CSF mask for "
            "aCompCor. Replace with an FSL FAST segmentation for real analyses.",
            sub,
        )

        confounds_df = build_confounds_table(
            paths["motion_params"], bold_data, wm_csf_mask, config
        )

        n_flagged = int(confounds_df["motion_outlier"].sum())
        logger.info(
            "[%s] %d / %d volumes flagged as high-motion (FD > %.2fmm or DVARS > %.2f)",
            sub,
            n_flagged,
            len(confounds_df),
            denoise_cfg["fd_scrubbing_threshold"],
            denoise_cfg["dvars_scrubbing_threshold"],
        )
        if n_flagged / len(confounds_df) > 0.25:
            logger.warning(
                "[%s] Over 25%% of volumes flagged for motion. Consider "
                "excluding this subject rather than scrubbing.",
                sub,
            )

        out_path = out_dir / "confounds.tsv"
        confounds_df.to_csv(out_path, sep="\t", index=False)
        confound_paths[sub] = out_path
        logger.info("[%s] Confounds saved to %s", sub, out_path)

    return confound_paths


if __name__ == "__main__":
    print("Batch 3 expects preprocessing outputs from batch 2 — run via run_pipeline.py")
