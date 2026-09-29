"""
view_results.py

Standalone results browser. Scans the pipeline's output folders and builds a
single self-contained HTML page (reports/results_summary.html) that shows,
in one place:
  - the run summary and environment manifest
  - per-subject QC (tSNR) and design matrices
  - per-subject first-level contrast glass brains
  - group-level contrast maps
  - a table of how many volumes were flagged for motion per subject

This does NOT re-run any analysis — it only collects what the batches already
wrote to disk, so it's safe to run any time after a pipeline run (even a
partial one; it simply shows whatever exists). Images are embedded directly in
the HTML as base64, so the single file is portable — you can email it or open
it on another machine with no dependencies.

Usage
-----
    python view_results.py            # writes reports/results_summary.html
    python view_results.py --open     # also open it in the default browser

Run it from the pipeline directory, inside the fmri-teaching-pipeline
environment (same as run_pipeline.py).
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from utils import load_config


def embed_image(path: Path) -> str:
    """Return an <img> tag with the PNG at `path` embedded as base64, or an
    empty string if the file is missing."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}">'


def read_text_if_exists(path: Path) -> str:
    """Return the text content of a file, or a placeholder if it's missing."""
    if path.exists():
        return path.read_text()
    return f"(not found: {path})"


def collect_confound_summary(derivatives_root: Path, subjects: list[str]) -> list[dict]:
    """Read each subject's confounds.tsv and count flagged (high-motion)
    volumes, for a compact QC table. Uses only the standard library so this
    script has no heavy dependencies."""
    import csv

    rows = []
    for sub in subjects:
        conf_path = derivatives_root / sub / "confounds" / "confounds.tsv"
        if not conf_path.exists():
            rows.append({"subject": sub, "n_volumes": "-", "n_flagged": "-"})
            continue
        with open(conf_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            n_vol = 0
            n_flagged = 0
            for line in reader:
                n_vol += 1
                val = line.get("motion_outlier", "").strip().lower()
                if val in ("true", "1", "1.0"):
                    n_flagged += 1
        rows.append({"subject": sub, "n_volumes": n_vol, "n_flagged": n_flagged})
    return rows


def build_html(config: dict) -> str:
    """Assemble the full HTML page from whatever outputs exist on disk."""
    report_dir = Path(config["output"]["report_dir"])
    results_dir = Path(config["output"]["results_dir"])
    derivatives_root = Path(config["dataset"]["derivatives_root"])
    figures_dir = report_dir / "figures"
    qc_dir = report_dir / "qc"
    subjects = config["dataset"]["subjects"]
    contrasts = list(config["first_level"]["contrasts"].keys())

    parts: list[str] = []
    parts.append(
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>fMRI Pipeline Results</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; margin: 2rem auto; max-width: 1100px;
         padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
  h1 { border-bottom: 3px solid #333; padding-bottom: .3rem; }
  h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
  h3 { margin-top: 1.5rem; color: #333; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: .4rem 0; }
  table { border-collapse: collapse; margin: 1rem 0; }
  th, td { border: 1px solid #ccc; padding: .4rem .8rem; text-align: left; }
  th { background: #f0f0f0; }
  pre { background: #f6f6f6; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: .85rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
  .note { background: #fff8e1; border-left: 4px solid #ffb300; padding: .6rem 1rem; margin: 1rem 0; }
  .muted { color: #777; font-size: .9rem; }
</style></head><body>"""
    )

    parts.append(f"<h1>fMRI Pipeline Results</h1>")
    parts.append(
        f'<p class="muted">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} '
        f'· dataset: {config["dataset"]["bids_root"]} · task: {config["dataset"]["task"]} '
        f'· subjects: {", ".join(subjects)}</p>'
    )

    # --- Run summary (latest reproducibility bundle) ---
    bundles = sorted(report_dir.glob("repro_bundle_*"), reverse=True)
    if bundles:
        summary = read_text_if_exists(bundles[0] / "run_summary.md")
        parts.append("<h2>Run summary</h2>")
        parts.append(f"<pre>{summary}</pre>")

    # --- Motion QC table ---
    parts.append("<h2>Motion QC (volumes flagged)</h2>")
    parts.append(
        '<p class="muted">Volumes exceeding the FD/DVARS thresholds in config.yaml. '
        "A handful is normal; a large fraction suggests excluding that subject.</p>"
    )
    conf_rows = collect_confound_summary(derivatives_root, subjects)
    parts.append("<table><tr><th>Subject</th><th>Volumes</th><th>Flagged (high-motion)</th></tr>")
    for r in conf_rows:
        parts.append(
            f'<tr><td>{r["subject"]}</td><td>{r["n_volumes"]}</td><td>{r["n_flagged"]}</td></tr>'
        )
    parts.append("</table>")

    # --- tSNR QC maps (note: these are NIfTI, not PNG, so we just list them) ---
    tsnr_files = sorted(qc_dir.glob("*_tsnr.nii.gz")) if qc_dir.exists() else []
    if tsnr_files:
        parts.append("<h2>tSNR QC maps</h2>")
        parts.append(
            '<p class="muted">Temporal SNR maps were written as NIfTI files (open in '
            "fsleyes to inspect). Files:</p><ul>"
        )
        for f in tsnr_files:
            parts.append(f"<li>{f}</li>")
        parts.append("</ul>")

    # --- Per-subject section: design matrix + first-level contrasts ---
    parts.append("<h2>Per-subject results</h2>")
    for sub in subjects:
        parts.append(f"<h3>{sub}</h3>")
        dmat = derivatives_root / sub / "first_level" / "design_matrix.png"
        dmat_tag = embed_image(dmat)
        if dmat_tag:
            parts.append("<p><b>Design matrix</b></p>")
            parts.append(dmat_tag)
        parts.append('<p><b>First-level contrast maps</b></p><div class="grid">')
        for contrast in contrasts:
            png = figures_dir / f"{sub}_contrast-{contrast}_glassbrain.png"
            tag = embed_image(png)
            if tag:
                parts.append(f"<div><p class='muted'>{contrast}</p>{tag}</div>")
        parts.append("</div>")

    # --- Group-level section ---
    parts.append("<h2>Group-level results</h2>")
    if len(subjects) < 2:
        parts.append(
            '<p class="note">Only one subject was run, so no group analysis was '
            "performed. Add subjects in config.yaml for group results.</p>"
        )
    else:
        parts.append(
            '<p class="note">With a small (teaching-scale) sample, it is normal for '
            "few or no voxels to survive multiple-comparisons correction. Empty maps "
            "here are an expected statistical-power result, not an error.</p>"
        )
        parts.append('<div class="grid">')
        for contrast in contrasts:
            png = figures_dir / f"group_contrast-{contrast}_glassbrain.png"
            tag = embed_image(png)
            if tag:
                parts.append(f"<div><p class='muted'>{contrast}</p>{tag}</div>")
        parts.append("</div>")

    # --- Environment manifest ---
    manifest_path = report_dir / "environment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        env = {
            "timestamp_utc": manifest.get("timestamp_utc"),
            "python": manifest.get("python_version", "").splitlines()[0]
            if manifest.get("python_version") else "",
            "external_tools": manifest.get("external_tools"),
            "python_packages": manifest.get("python_packages"),
        }
        parts.append("<h2>Environment</h2>")
        parts.append(f"<pre>{json.dumps(env, indent=2)}</pre>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main(open_browser: bool = False) -> Path:
    config = load_config()
    html = build_html(config)
    out_path = Path(config["output"]["report_dir"]) / "results_summary.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Results summary written to: {out_path.resolve()}")

    if open_browser:
        import webbrowser

        webbrowser.open(out_path.resolve().as_uri())
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an HTML summary of pipeline results.")
    parser.add_argument("--open", action="store_true", help="Open the page in a browser when done.")
    args = parser.parse_args()
    main(open_browser=args.open)
