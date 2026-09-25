"""
BATCH 1 — Data Organization & Quality Control
===============================================

CONCEPT
-------
Every downstream step assumes the data is (a) organized in BIDS format, so
tools can find files by convention rather than hardcoded paths, and (b) free
of gross artifacts (excessive motion, scanner dropout) that would silently
corrupt statistics later. Checking this *first* saves hours of debugging a
GLM that's actually failing because of bad input data.

WHY NOT MRIQC HERE
-------------------
MRIQC is the field-standard QC tool, but it's typically run as a container
(Docker/Singularity) for dependency isolation. Since this pipeline runs
natively, we use a lightweight custom QC step instead: mean/std images,
temporal SNR, and per-volume framewise displacement computed directly from
the motion parameters produced in batch 2. If your teaching environment DOES
have MRIQC installed natively, you can swap this step out — the interface
(a per-subject QC report) stays the same.

WHAT THIS BATCH DOES
---------------------
1. Validates the dataset is BIDS-compliant using pybids.
2. Confirms the requested subjects/task actually exist in the dataset.
3. Computes a quick temporal SNR (tSNR) map per subject as a first-pass
   data-quality check (low tSNR = noisy data).

SANITY CHECK
------------
A tSNR summary printed per subject, plus a saved tSNR map image learners can
eyeball for obvious problems (e.g. signal dropout in orbitofrontal cortex).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from bids import BIDSLayout
from nilearn.masking import compute_epi_mask

from utils import ensure_dir, get_logger, load_config


def validate_bids(bids_root: str) -> BIDSLayout:
    """Load and validate the dataset with pybids.

    Raises
    ------
    RuntimeError
        If the dataset fails BIDS validation — deliberately fails loudly here
        rather than letting a malformed dataset propagate into preprocessing.
    """
    layout = BIDSLayout(bids_root, validate=True)
    return layout


def check_requested_subjects_exist(layout: BIDSLayout, subjects: list[str], task: str) -> None:
    """Confirm every subject/task combination requested in config.yaml actually
    exists in the dataset, and fail with a clear message if not."""
    available_subjects = {f"sub-{s}" for s in layout.get_subjects()}
    available_tasks = set(layout.get_tasks())

    for sub in subjects:
        if sub not in available_subjects:
            raise ValueError(
                f"Subject '{sub}' requested in config.yaml was not found in the "
                f"dataset. Available subjects: {sorted(available_subjects)}"
            )
    if task not in available_tasks:
        raise ValueError(
            f"Task '{task}' requested in config.yaml was not found. "
            f"Available tasks: {sorted(available_tasks)}"
        )


def compute_tsnr(bold_path: Path) -> tuple[np.ndarray, float]:
    """Compute a temporal signal-to-noise ratio (tSNR) map for one BOLD run.

    tSNR is simply mean(signal) / std(signal) at each voxel over time — a
    standard, fast first-pass indicator of data quality that doesn't require
    any preprocessing to compute.

    Because this runs on RAW, non-skull-stripped data (brain extraction
    doesn't happen until batch 2), the median must be restricted to an actual
    brain mask — not just "any voxel with positive mean" — or background/
    scalp voxels dominated by thermal noise (where mean ~ std, so tSNR
    collapses toward 1-3) will swamp the median and make even good data look
    unusable. We use nilearn's compute_epi_mask, which separates brain from
    background using a validated intensity-histogram heuristic rather than an
    arbitrary percentile cutoff.

    Returns
    -------
    tsnr_map : np.ndarray
        3D tSNR map (full field of view, not masked — useful for visual QC).
    median_tsnr : float
        Median tSNR within the estimated brain mask only.
    """
    img = nib.load(bold_path)
    data = img.get_fdata()  # shape: (x, y, z, t)

    mean_img = data.mean(axis=-1)
    std_img = data.std(axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_map = np.where(std_img > 0, mean_img / std_img, 0)

    brain_mask_img = compute_epi_mask(img)
    brain_mask = brain_mask_img.get_fdata().astype(bool)

    median_tsnr = float(np.median(tsnr_map[brain_mask]))

    return tsnr_map, median_tsnr


def run() -> dict:
    """Entry point for batch 1. Returns a dict of {subject: median_tsnr} for
    logging/reporting in later batches."""
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    bids_root = config["dataset"]["bids_root"]
    subjects = config["dataset"]["subjects"]
    task = config["dataset"]["task"]

    logger.info("Batch 1: validating BIDS dataset at %s", bids_root)
    layout = validate_bids(bids_root)
    logger.info("BIDS validation passed.")

    check_requested_subjects_exist(layout, subjects, task)
    logger.info("Confirmed subjects %s and task '%s' exist in dataset.", subjects, task)

    report_dir = ensure_dir(Path(config["output"]["report_dir"]) / "qc")
    tsnr_summary = {}

    for sub in subjects:
        sub_id = sub.replace("sub-", "")
        bold_files = layout.get(
            subject=sub_id, task=task, suffix="bold", extension=".nii.gz", return_type="file"
        )
        if not bold_files:
            logger.warning("No BOLD file found for %s / task-%s — skipping QC.", sub, task)
            continue

        bold_path = Path(bold_files[0])
        logger.info("Computing tSNR for %s: %s", sub, bold_path.name)
        tsnr_map, median_tsnr = compute_tsnr(bold_path)
        tsnr_summary[sub] = median_tsnr

        # Save the tSNR map as a NIfTI so learners can view it in fsleyes/nilearn
        ref_img = nib.load(bold_path)
        tsnr_img = nib.Nifti1Image(tsnr_map, affine=ref_img.affine)
        out_path = report_dir / f"{sub}_task-{task}_tsnr.nii.gz"
        nib.save(tsnr_img, out_path)

        logger.info("%s median tSNR = %.2f (map saved to %s)", sub, median_tsnr, out_path)
        if median_tsnr < 40:
            logger.warning(
                "%s has a low median tSNR (%.2f). Typical usable fMRI data is "
                "often tSNR > 50-60; inspect this subject's data before proceeding.",
                sub,
                median_tsnr,
            )

    return tsnr_summary


if __name__ == "__main__":
    run()
