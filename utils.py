"""
utils.py

Shared helpers used by every batch script: config loading, logging setup, and
small path utilities. Keeping these in one place means each batch script stays
focused on its own analysis logic rather than repeating boilerplate.

Teaching note: every batch script below follows the same pattern —
    1. load_config()
    2. get_logger(__name__)
    3. do the actual neuroimaging work
    4. log + save outputs
This module is what makes that pattern possible without copy-pasting code.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Force FSL to write proper gzip-compressed NIfTI everywhere, in every batch
# that imports utils. If $FSLOUTPUTTYPE is left to the shell it can differ
# between machines, and a mismatch (FSL writing uncompressed data under a
# .nii.gz name) produces files that FSL's own lenient tools read but that
# strict readers — nibabel and FSL's newer applywarp — cannot open. Pinning it
# here makes the whole pipeline's file format deterministic.
os.environ["FSLOUTPUTTYPE"] = "NIFTI"

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the pipeline configuration from YAML.

    Parameters
    ----------
    config_path : Path or str
        Path to config.yaml. Defaults to the config.yaml shipped alongside
        this module.

    Returns
    -------
    dict
        Parsed configuration. Environment-variable placeholders like
        ``${FSLDIR}`` are expanded automatically.
    """
    with open(config_path, "r") as f:
        raw_text = f.read()
    expanded_text = os.path.expandvars(raw_text)
    config = yaml.safe_load(expanded_text)
    return config


def get_logger(name: str, log_dir: Path | str = "./logs") -> logging.Logger:
    """Create a logger that writes to both console and a per-run log file.

    Using ``logging`` instead of ``print`` means learners can see exactly
    which step produced which message, with timestamps, and can inspect a
    persistent log file after the session ends.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module, so log lines are
        attributable to a specific batch script.
    log_dir : Path or str
        Directory to write the log file into. Created if it doesn't exist.

    Returns
    -------
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers if get_logger() is called more than once
        # for the same module (e.g. when re-running a batch interactively).
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it doesn't exist, and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_executable(name: str) -> None:
    """Raise a clear, actionable error if a required external tool is missing.

    Since this pipeline calls FSL/ANTs natively (no Docker), silent failures
    when a tool isn't on $PATH are a common and confusing failure mode for
    learners. Fail loudly instead.
    """
    from shutil import which

    if which(name) is None:
        raise EnvironmentError(
            f"Required external tool '{name}' was not found on $PATH. "
            f"This pipeline runs FSL/ANTs natively (no Docker/Singularity), "
            f"so they must be installed and sourced before running this script. "
            f"Check with `which {name}` in your terminal."
        )
