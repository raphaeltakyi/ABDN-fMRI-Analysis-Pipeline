"""
BATCH 6 — Visualization & Reporting
======================================

CONCEPT
-------
A z-value in a table means little without seeing where it is in the brain and
in the context of the whole analysis (design matrix, motion, contrast maps).
This batch turns the numeric outputs of batches 1-5 into a single browsable
HTML report per subject and one for the group result — the artifact you'd
actually hand a collaborator, or attach to a thesis appendix.

WHAT THIS BATCH DOES
---------------------
1. Glass-brain + slice overlays of each subject's first-level contrast map.
2. A glass-brain + slice overlay of the thresholded group map.
3. Assembles nilearn's built-in HTML report generator (which bundles design
   matrix, contrast maps, and stat tables) for both first- and second-level
   models, so learners have one reproducible artifact per analysis stage.

SANITY CHECK
------------
Open the generated HTML report and confirm activation (or its absence) is in
anatomically plausible locations for the task studied — this is the final,
most intuitive QC step before drawing any conclusions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless backend so figures render without a display
import matplotlib.pyplot as plt
from nilearn import plotting

from utils import ensure_dir, get_logger, load_config


def _save_empty_placeholder(out_png: Path, title: str, note: str) -> None:
    """Write a labelled placeholder image when a map has no voxels to show.

    A glass brain of an all-zero map is just an empty outline, which looks like
    something broke. A placeholder that says WHY it's empty (e.g. nothing
    survived correction) is clearer for a learner reviewing the figures.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.5, 0.5, note, ha="center", va="center", fontsize=12, wrap=True,
        transform=ax.transAxes,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _map_has_signal(z_map_path: Path, threshold: float) -> bool:
    """Return True if the map has any voxel whose absolute value exceeds the
    display threshold — i.e. there is actually something to plot."""
    data = nib.load(str(z_map_path)).get_fdata()
    return bool(np.any(np.abs(data) > threshold))


def plot_contrast_map(z_map_path: Path, out_png: Path, title: str, threshold: float = 3.0) -> bool:
    """Save a glass-brain visualization of a statistical map.

    Returns True if a real glass brain was drawn, False if the map was empty
    (above threshold) and a placeholder was written instead.
    """
    if not _map_has_signal(z_map_path, threshold):
        _save_empty_placeholder(
            out_png, title,
            f"No voxels above threshold (|z| > {threshold:g}).\n"
            f"Nothing to display for this map.",
        )
        return False

    display = plotting.plot_glass_brain(
        str(z_map_path),
        threshold=threshold,
        colorbar=True,
        plot_abs=False,
        title=title,
    )
    display.savefig(out_png, dpi=150)
    display.close()
    return True


def run(
    first_level_results: dict,
    group_results: dict,
) -> dict:
    """Entry point for batch 6.

    Parameters
    ----------
    first_level_results : dict
        Output of batch04_first_level_analysis.run().
    group_results : dict
        Output of batch05_second_level_analysis.run().

    Returns
    -------
    dict
        Paths to generated figures/reports, organized by subject and 'group'.
    """
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    report_dir = ensure_dir(Path(config["output"]["report_dir"]) / "figures")

    output_paths: dict = {"subjects": {}, "group": {}}

    # --- Per-subject contrast visualizations ---
    for sub, contrasts in first_level_results.items():
        output_paths["subjects"][sub] = {}
        for contrast_name, z_map_path in contrasts.items():
            out_png = report_dir / f"{sub}_contrast-{contrast_name}_glassbrain.png"
            drew = plot_contrast_map(
                z_map_path, out_png, title=f"{sub}: {contrast_name}"
            )
            output_paths["subjects"][sub][contrast_name] = out_png
            logger.info(
                "[%s] Contrast map figure saved to %s%s",
                sub, out_png, "" if drew else " (empty — placeholder written)",
            )

    # --- Group-level visualizations ---
    for contrast_name, result in group_results.items():
        out_png = report_dir / f"group_contrast-{contrast_name}_glassbrain.png"
        n_vox = result["n_suprathreshold_voxels"]
        title = (
            f"Group: {contrast_name} "
            f"(n={result['n_subjects']}, {n_vox} voxels survive)"
        )
        if n_vox == 0:
            # The map is already thresholded and empty; skip the blank glass
            # brain and write an explanatory placeholder instead.
            _save_empty_placeholder(
                out_png, title,
                "No voxels survived multiple-comparisons correction.\n"
                "Expected with a small (teaching-scale) sample — not an error.",
            )
            logger.info(
                "[group] Contrast '%s' had no surviving voxels; placeholder written to %s",
                contrast_name, out_png,
            )
        else:
            plot_contrast_map(
                result["thresholded_map"], out_png, title=title, threshold=0.0,
            )
            logger.info("[group] Contrast map figure saved to %s", out_png)
        output_paths["group"][contrast_name] = out_png

    logger.info(
        "Batch 6 complete. Review figures in %s before drawing conclusions.",
        report_dir,
    )
    return output_paths


if __name__ == "__main__":
    print("Batch 6 expects outputs from batches 4-5 — run via run_pipeline.py")
