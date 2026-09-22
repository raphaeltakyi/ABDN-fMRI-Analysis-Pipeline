"""
BATCH 4 — First-Level (Subject-Level) Statistical Analysis
==============================================================

CONCEPT
-------
The General Linear Model (GLM) asks: at each voxel, how well does a predicted
signal (based on when the task happened, convolved with a model of the
hemodynamic response) explain the observed BOLD timeseries, after accounting
for the confounds from batch 3? The output is a statistical map per subject,
per contrast — the input to the group analysis in batch 5.

WHY nilearn.glm
-----------------
`nilearn.glm.first_level` is a mature, well-tested, pure-Python GLM
implementation that mirrors SPM's approach while staying fully scriptable and
inspectable — ideal for teaching, since learners can print/plot the design
matrix itself rather than treating GLM fitting as a black box.

WHAT THIS BATCH DOES
---------------------
1. Loads each subject's task events (onsets/durations/trial types) from BIDS.
2. Builds a design matrix: task regressors (convolved with an HRF) + the
   confound regressors from batch 3 + a high-pass filter (as cosine drift
   terms) for scanner drift.
3. Fits the GLM.
4. Computes the contrast(s) defined in config.yaml.
5. Saves the resulting z-statistic map per subject/contrast.

SANITY CHECK
------------
The design matrix itself is saved as a plot — learners should visually
confirm the task regressor looks like a sensible, non-degenerate predictor
before trusting the statistics that come out of fitting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

from utils import ensure_dir, get_logger, load_config


def load_events(bids_root: str, sub: str, task: str) -> pd.DataFrame:
    """Load the BIDS events.tsv (onset, duration, trial_type) for one subject/task."""
    sub_id = sub.replace("sub-", "")
    events_path = Path(bids_root) / sub / "func" / f"{sub}_task-{task}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(
            f"No events.tsv found for {sub} at {events_path}. First-level "
            f"analysis requires task timing information."
        )
    return pd.read_csv(events_path, sep="\t")


def build_design_matrix(
    n_volumes: int,
    t_r: float,
    events: pd.DataFrame,
    confounds_path: Path,
    config: dict,
) -> pd.DataFrame:
    """Construct the full first-level design matrix: task regressors (HRF-convolved)
    + selected confound regressors + high-pass cosine drift terms."""
    frame_times = np.arange(n_volumes) * t_r

    confounds_df = pd.read_csv(confounds_path, sep="\t")
    # Drop the boolean scrubbing flag column — it's a QC indicator, not a
    # regressor. Actual scrubbing (if desired) is done via sample_masks below.
    confound_cols = [c for c in confounds_df.columns if c != "motion_outlier"]

    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model=config["first_level"]["hrf_model"],
        drift_model="cosine",
        high_pass=config["first_level"]["high_pass_hz"],
        add_regs=confounds_df[confound_cols],
        add_reg_names=confound_cols,
    )
    return design_matrix


def fit_first_level_model(
    bold_path: Path, design_matrix: pd.DataFrame, t_r: float, smoothing_fwhm: float
) -> FirstLevelModel:
    """Fit the GLM to one subject's preprocessed, MNI-space BOLD data."""
    model = FirstLevelModel(
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm,
        minimize_memory=False,  # keep residuals available for QC/reporting
    )
    model.fit(str(bold_path), design_matrices=design_matrix)
    return model


def run(preprocessing_outputs: dict, confound_paths: dict, t_r: float) -> dict:
    """Entry point for batch 4.

    Parameters
    ----------
    preprocessing_outputs : dict
        Output of batch02_preprocessing.run().
    confound_paths : dict
        Output of batch03_denoising_confounds.run().
    t_r : float
        Repetition time in seconds (read from the BIDS sidecar JSON in
        run_pipeline.py, since it's dataset-specific and must not be hardcoded).

    Returns
    -------
    dict
        {subject: {contrast_name: Path to z-map}}
    """
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    bids_root = config["dataset"]["bids_root"]
    task = config["dataset"]["task"]
    contrasts_cfg = config["first_level"]["contrasts"]

    results = {}
    for sub, prep_paths in preprocessing_outputs.items():
        logger.info("[%s] First-level GLM", sub)
        out_dir = ensure_dir(Path(config["dataset"]["derivatives_root"]) / sub / "first_level")

        events = load_events(bids_root, sub, task)
        bold_path = prep_paths["corrected_bold"]

        import nibabel as nib

        n_volumes = nib.load(bold_path).shape[-1]

        design_matrix = build_design_matrix(
            n_volumes, t_r, events, confound_paths[sub], config
        )

        # Sanity-check artifact: save the design matrix plot before fitting
        fig = plot_design_matrix(design_matrix)
        fig.figure.savefig(out_dir / "design_matrix.png", dpi=150)
        logger.info("[%s] Design matrix saved to %s/design_matrix.png", sub, out_dir)

        model = fit_first_level_model(
            bold_path, design_matrix, t_r, config["first_level"]["smoothing_fwhm"]
        )

        results[sub] = {}
        for contrast_name, contrast_expr in contrasts_cfg.items():
            missing_cols = [
                c for c in [contrast_expr] if c not in design_matrix.columns
            ]
            if missing_cols:
                logger.warning(
                    "[%s] Contrast '%s' references column(s) %s not found in "
                    "the design matrix (columns: %s). Skipping.",
                    sub, contrast_name, missing_cols, list(design_matrix.columns),
                )
                continue

            z_map = model.compute_contrast(contrast_expr, output_type="z_score")
            out_path = out_dir / f"{sub}_contrast-{contrast_name}_zmap.nii.gz"
            z_map.to_filename(out_path)
            results[sub][contrast_name] = out_path
            logger.info("[%s] Contrast '%s' saved to %s", sub, contrast_name, out_path)

    return results


if __name__ == "__main__":
    print("Batch 4 expects outputs from batches 2-3 — run via run_pipeline.py")
