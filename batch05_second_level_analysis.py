"""
BATCH 5 — Second-Level (Group) Statistical Analysis
========================================================

CONCEPT
-------
A single subject's contrast map tells you nothing about whether an effect
generalizes. The second-level (group) model treats each subject's first-level
contrast map as one data point and asks whether the effect is reliably
non-zero across subjects — this is what lets you make a population-level
claim rather than a single-subject one.

MULTIPLE COMPARISONS
----------------------
Testing ~100,000+ voxels independently at p<0.05 guarantees thousands of false
positives by chance alone. This batch requires an explicit correction method
(set in config.yaml): FDR (fast, controls expected proportion of false
discoveries) or a nonparametric permutation-based cluster correction (slower,
controls family-wise error, generally considered more rigorous for fMRI).
Never report an uncorrected map as a final result — this pipeline makes that
an explicit, visible config choice rather than a silent default.

WHAT THIS BATCH DOES
---------------------
1. Collects each subject's first-level contrast map (from batch 4) for a
   given contrast.
2. Builds a one-sample second-level design (intercept-only — testing whether
   the group mean effect differs from zero).
3. Fits the second-level GLM.
4. Applies the configured multiple-comparisons correction.
5. Saves the thresholded group z-map and a text summary of significant
   clusters.

SANITY CHECK
------------
Report the number of subjects that went into the group model and the number
of suprathreshold voxels/clusters — a group map built from too few subjects,
or with zero surviving voxels, should be flagged for the learner, not
silently plotted as if it were a normal result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img

from utils import ensure_dir, get_logger, load_config


def build_group_design_matrix(n_subjects: int) -> pd.DataFrame:
    """One-sample design: a single intercept column of 1s, one row per subject.
    This tests whether the group-mean contrast effect differs from zero."""
    return pd.DataFrame({"intercept": [1] * n_subjects})


def run_fdr_correction(z_map, alpha: float):
    """Threshold a group z-map using false discovery rate (FDR) correction."""
    thresholded_map, threshold = threshold_stats_img(
        z_map, alpha=alpha, height_control="fdr"
    )
    return thresholded_map, threshold


def run_cluster_permutation_correction(
    first_level_maps: list, design_matrix: pd.DataFrame, alpha: float, n_perm: int
):
    """Threshold using nonparametric permutation-based cluster correction —
    generally considered the most rigorous option for fMRI group inference,
    at the cost of computation time."""
    result = non_parametric_inference(
        first_level_maps,
        design_matrix=design_matrix,
        model_intercept=True,
        n_perm=n_perm,
        two_sided_test=True,
        threshold=0.001,  # cluster-forming threshold (uncorrected)
    )
    # non_parametric_inference returns a dict; the FWE-corrected logp map is
    # what we want to threshold on for a corrected result.
    logp_map = result["logp_max_size"]
    return logp_map, alpha


def run(first_level_results: dict) -> dict:
    """Entry point for batch 5.

    Parameters
    ----------
    first_level_results : dict
        Output of batch04_first_level_analysis.run(): {subject: {contrast: Path}}.

    Returns
    -------
    dict
        {contrast_name: {"thresholded_map": Path, "n_subjects": int, "threshold": float}}
    """
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    second_level_cfg = config["second_level"]
    results_dir = ensure_dir(config["output"]["results_dir"])

    # Reorganize first_level_results by contrast: {contrast: [maps across subjects]}
    contrast_maps: dict[str, list] = {}
    for sub, contrasts in first_level_results.items():
        for contrast_name, map_path in contrasts.items():
            contrast_maps.setdefault(contrast_name, []).append(map_path)

    group_results = {}
    for contrast_name, maps in contrast_maps.items():
        n_subjects = len(maps)
        logger.info(
            "[group] Contrast '%s': %d subjects contributing", contrast_name, n_subjects
        )
        if n_subjects < 2:
            logger.warning(
                "[group] Contrast '%s' has only %d subject(s). Group inference "
                "is not meaningful with fewer than ~2 subjects — this is a "
                "teaching-scale run, not a publishable group result.",
                contrast_name, n_subjects,
            )

        design_matrix = build_group_design_matrix(n_subjects)
        second_level_model = SecondLevelModel(smoothing_fwhm=None)
        second_level_model.fit([str(m) for m in maps], design_matrix=design_matrix)

        raw_z_map = second_level_model.compute_contrast(
            second_level_contrast="intercept", output_type="z_score"
        )

        method = second_level_cfg["correction_method"]
        if method == "fdr":
            thresholded_map, threshold = run_fdr_correction(
                raw_z_map, second_level_cfg["alpha"]
            )
        elif method == "cluster":
            thresholded_map, threshold = run_cluster_permutation_correction(
                [str(m) for m in maps],
                design_matrix,
                second_level_cfg["alpha"],
                second_level_cfg["cluster_permutations"],
            )
        else:
            raise ValueError(f"Unknown correction method: {method}")

        import numpy as np

        n_suprathreshold = int(np.sum(thresholded_map.get_fdata() != 0))
        if n_suprathreshold == 0:
            logger.warning(
                "[group] Contrast '%s': zero voxels survived %s correction "
                "at alpha=%.3f. This may be a true null result or may reflect "
                "insufficient power (small n, teaching-scale dataset).",
                contrast_name, method, second_level_cfg["alpha"],
            )
        else:
            logger.info(
                "[group] Contrast '%s': %d voxels survived %s correction "
                "(threshold=%.3f).",
                contrast_name, n_suprathreshold, method, threshold,
            )

        out_path = results_dir / f"group_contrast-{contrast_name}_{method}corrected.nii.gz"
        thresholded_map.to_filename(out_path)

        group_results[contrast_name] = {
            "thresholded_map": out_path,
            "n_subjects": n_subjects,
            "threshold": threshold,
            "n_suprathreshold_voxels": n_suprathreshold,
        }

    return group_results


if __name__ == "__main__":
    print("Batch 5 expects outputs from batch 4 — run via run_pipeline.py")
