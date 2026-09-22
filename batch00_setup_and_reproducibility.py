"""
BATCH 0 — Environment & Reproducibility Setup
==============================================

CONCEPT
-------
"Reproducible" doesn't just mean "the code runs again" — it means someone else,
on a different machine, a year from now, can get the *same* result. That
requires knowing exactly which package versions, which external tools
(FSL/ANTs), and which dataset version were used. This batch captures all of
that automatically, before any analysis happens, so every later batch inherits
a documented, reproducible environment.

WHAT THIS BATCH DOES
---------------------
1. Sets a global random seed.
2. Records exact Python package versions (from the active environment).
3. Records external tool versions (FSL, ANTs) since they aren't pip packages.
4. Records basic OS/hardware info.
5. Writes everything to reports/environment_manifest.json so it can be cited
   or attached to a paper/thesis appendix.

SANITY CHECK
------------
Print the manifest at the end and confirm FSL/ANTs were actually found — if
they weren't, later batches will fail, and it's better to catch that now.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import numpy as np

from utils import ensure_dir, get_logger, load_config

CORE_PACKAGES = ["nilearn", "nipype", "nibabel", "pybids", "numpy", "pandas", "scipy"]


def set_seeds(seed: int) -> None:
    """Seed every RNG this pipeline touches so stochastic steps (e.g. permutation
    testing in batch 5) are reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Return installed version strings for the given package names."""
    installed = {d.metadata["Name"].lower(): d.version for d in distributions()}
    versions = {}
    for pkg in packages:
        versions[pkg] = installed.get(pkg.lower(), "NOT INSTALLED")
    return versions


def get_external_tool_version(command: list[str]) -> str:
    """Run a version-check command for an external tool and capture its output.

    Returns 'NOT FOUND' rather than raising, because this function's job is to
    *report* environment state, not to enforce it (that's utils.require_executable's
    job, used later in batch 2 right before the tool is actually needed).
    """
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        output = (result.stdout or result.stderr).strip()
        return output.splitlines()[0] if output else "UNKNOWN VERSION"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "NOT FOUND"


def build_manifest(config: dict) -> dict:
    """Assemble the full reproducibility manifest as a plain dict (JSON-serializable)."""
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": config["environment"]["random_seed"],
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_packages": get_package_versions(CORE_PACKAGES),
        "external_tools": {
            "fsl": get_external_tool_version(["flirt", "-version"]),
            "ants": get_external_tool_version(["antsRegistration", "--version"]),
        },
        "config_used": config,
    }
    return manifest


def run() -> dict:
    """Entry point for batch 0. Returns the manifest dict for use in later batches
    (e.g. batch 7 re-uses it in the final reproducibility artifact)."""
    config = load_config()
    logger = get_logger(__name__, config["output"]["log_dir"])

    logger.info("Batch 0: setting seed and building environment manifest.")
    set_seeds(config["environment"]["random_seed"])

    manifest = build_manifest(config)

    report_dir = ensure_dir(config["output"]["report_dir"])
    manifest_path = Path(report_dir) / "environment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Environment manifest written to %s", manifest_path)

    # Sanity check learners should look at before proceeding
    for tool, version in manifest["external_tools"].items():
        if version == "NOT FOUND":
            logger.warning(
                "%s was NOT FOUND on $PATH. Batch 2 (preprocessing) will fail "
                "until it is installed and sourced.",
                tool.upper(),
            )
        else:
            logger.info("%s detected: %s", tool.upper(), version)

    for pkg, version in manifest["python_packages"].items():
        if version == "NOT INSTALLED":
            logger.warning("Python package '%s' is not installed.", pkg)

    return manifest


if __name__ == "__main__":
    run()
