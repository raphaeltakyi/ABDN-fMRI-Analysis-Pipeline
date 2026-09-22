"""
run_pipeline.py

Orchestrator for the full preprocessing-to-statistics pipeline. This is the
"holistic" entry point that chains every batch together for a live end-to-end
demo — but each batch script (batch00_*.py ... batch07_*.py) is also fully
runnable and importable on its own, which is the intended way to *teach* this:
run one batch, inspect its output and sanity-check plot, then move to the
next.

USAGE
-----
    python run_pipeline.py                # run every batch, all configured subjects
    python run_pipeline.py --until 3       # run batches 0 through 3 only
    python run_pipeline.py --from 4        # resume from batch 4 (needs prior outputs)

Batches communicate by passing Python objects (dicts of paths) directly in
this orchestrator, rather than round-tripping through disk between every
step — but every batch ALSO writes its outputs to disk (see each batch's
docstring), so you can inspect intermediate results at any point, or resume
a session from a saved state without rerunning earlier batches.
"""

from __future__ import annotations

import argparse
import json

import nibabel as nib
from bids import BIDSLayout

import batch00_setup_and_reproducibility as batch00
import batch01_data_organization_qc as batch01
import batch02_preprocessing as batch02
import batch03_denoising_confounds as batch03
import batch04_first_level_analysis as batch04
import batch05_second_level_analysis as batch05
import batch06_visualization_reporting as batch06
import batch07_reproducibility_artifact as batch07
from utils import get_logger, load_config


def get_repetition_time(config: dict) -> float:
    """Read TR from the BIDS sidecar JSON rather than hardcoding it — TR is a
    dataset property, not a pipeline constant, and getting it wrong silently
    corrupts every downstream GLM timing."""
    layout = BIDSLayout(config["dataset"]["bids_root"])
    task = config["dataset"]["task"]
    metadata = layout.get(task=task, suffix="bold", extension=".nii.gz")[0].get_metadata()
    if "RepetitionTime" not in metadata:
        raise KeyError(
            f"No RepetitionTime found in the BIDS sidecar for task-{task}. "
            f"Check the dataset's .json sidecar files."
        )
    return float(metadata["RepetitionTime"])


def main(run_from: int = 0, run_until: int = 7) -> None:
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])
    logger.info(
        "Starting pipeline run: batches %d through %d, subjects=%s",
        run_from, run_until, config["dataset"]["subjects"],
    )

    state = {}

    if run_from <= 0 <= run_until:
        state["environment_manifest"] = batch00.run()

    if run_from <= 1 <= run_until:
        state["tsnr_summary"] = batch01.run()

    if run_from <= 2 <= run_until:
        state["preprocessing_outputs"] = batch02.run()

    if run_from <= 3 <= run_until:
        state["confound_paths"] = batch03.run(state["preprocessing_outputs"])

    if run_from <= 4 <= run_until:
        t_r = get_repetition_time(config)
        state["first_level_results"] = batch04.run(
            state["preprocessing_outputs"], state["confound_paths"], t_r
        )

    if run_from <= 5 <= run_until:
        state["group_results"] = batch05.run(state["first_level_results"])

    if run_from <= 6 <= run_until:
        state["figure_paths"] = batch06.run(
            state["first_level_results"], state["group_results"]
        )

    if run_from <= 7 <= run_until:
        state["bundle_dir"] = batch07.run(
            state["environment_manifest"],
            state["first_level_results"],
            state["group_results"],
        )

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fMRI teaching pipeline.")
    parser.add_argument(
        "--from", dest="run_from", type=int, default=0,
        help="First batch to run (0-7). Requires prior batch outputs to already exist "
             "if > 0 and you're not running interactively with `state` in scope.",
    )
    parser.add_argument(
        "--until", dest="run_until", type=int, default=7,
        help="Last batch to run (0-7).",
    )
    args = parser.parse_args()
    main(run_from=args.run_from, run_until=args.run_until)
