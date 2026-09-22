"""
BATCH 7 — Reproducibility Artifact
=====================================

CONCEPT
-------
The pipeline is only reproducible if the *record* of how it was run survives
past the session. This batch closes the loop opened in batch 0: it bundles
the environment manifest, the exact config used, and a summary of what ran
into a single dated folder, plus a Dockerfile stub so a learner (or you,
later) can eventually containerize this exact environment even though today's
session ran natively.

WHAT THIS BATCH DOES
---------------------
1. Copies the environment manifest (batch 0) and config.yaml into a
   timestamped reproducibility folder.
2. Writes a run summary: which subjects/contrasts were processed, and the
   headline group-level result counts.
3. Writes a Dockerfile stub capturing the native tool versions detected in
   batch 0, so this environment CAN be containerized later even though this
   teaching session ran without Docker.

SANITY CHECK
------------
Read run_summary.md at the end — it should be enough, on its own, for someone
who was not in the room to understand what was run and reproduce it.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from utils import ensure_dir, get_logger, load_config


DOCKERFILE_TEMPLATE = """\
# Dockerfile stub generated from a native (non-Docker) teaching run.
# Fill in exact FSL/ANTs versions from environment_manifest.json before
# building, so a future container run matches what was actually used here.
FROM continuumio/miniconda3

# --- FSL/ANTs installation goes here ---
# This teaching run used FSL: {fsl_version}
# This teaching run used ANTs: {ants_version}
# See https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation and
# https://github.com/ANTsX/ANTs for install instructions matching these versions.

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /pipeline
COPY . /pipeline
"""


def run(
    environment_manifest: dict,
    first_level_results: dict,
    group_results: dict,
) -> Path:
    """Entry point for batch 7. Returns the path to the reproducibility bundle."""
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = ensure_dir(Path(config["output"]["report_dir"]) / f"repro_bundle_{timestamp}")

    # 1. Copy environment manifest and config
    with open(bundle_dir / "environment_manifest.json", "w") as f:
        json.dump(environment_manifest, f, indent=2)
    shutil.copy(Path(__file__).parent / "config.yaml", bundle_dir / "config.yaml")

    # 2. Write run summary
    n_subjects_run = len(first_level_results)
    contrast_names = sorted(
        {c for contrasts in first_level_results.values() for c in contrasts}
    )

    summary_lines = [
        f"# Pipeline Run Summary — {timestamp}",
        "",
        f"- Subjects processed: {list(first_level_results.keys())} (n={n_subjects_run})",
        f"- Contrasts computed: {contrast_names}",
        "",
        "## Group-level results",
    ]
    for contrast_name, result in group_results.items():
        summary_lines.append(
            f"- **{contrast_name}**: n={result['n_subjects']}, "
            f"{result['n_suprathreshold_voxels']} voxels survived correction "
            f"(threshold={result['threshold']:.3f})"
        )

    summary_lines += [
        "",
        "## Environment",
        f"- Python: {environment_manifest['python_version'].splitlines()[0]}",
        f"- FSL: {environment_manifest['external_tools']['fsl']}",
        f"- ANTs: {environment_manifest['external_tools']['ants']}",
        f"- Random seed: {environment_manifest['random_seed']}",
        "",
        "See `environment_manifest.json` and `config.yaml` in this folder for full detail.",
    ]

    summary_path = bundle_dir / "run_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    # 3. Write Dockerfile stub for future containerization
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        fsl_version=environment_manifest["external_tools"]["fsl"],
        ants_version=environment_manifest["external_tools"]["ants"],
    )
    with open(bundle_dir / "Dockerfile.stub", "w") as f:
        f.write(dockerfile_content)

    logger.info("Reproducibility bundle written to %s", bundle_dir)
    logger.info("Run summary: %s", summary_path)

    return bundle_dir


if __name__ == "__main__":
    print("Batch 7 expects outputs from batches 0, 4, and 5 — run via run_pipeline.py")
