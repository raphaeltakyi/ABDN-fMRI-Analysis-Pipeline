# fMRI Teaching Pipeline: Preprocessing → Statistics

A modular, batch-by-batch Python pipeline for teaching open-source fMRI
analysis, from raw BIDS data to group-level statistics. Built for a
**no-Docker** environment: preprocessing calls FSL and ANTs natively via
Nipype rather than through fMRIPrep's container.

## Structure

Each stage is a standalone, importable, individually runnable script — run
one at a time in a teaching session, or chain them all with `run_pipeline.py`.

| File | Batch | What it does |
|---|---|---|
| `batch00_setup_and_reproducibility.py` | 0 | Seeds RNGs, logs package/tool versions |
| `batch01_data_organization_qc.py` | 1 | BIDS validation, tSNR-based QC |
| `batch02_preprocessing.py` | 2 | BET, slice-timing, MCFLIRT, FLIRT, normalization (FSL/ANTs) |
| `batch03_denoising_confounds.py` | 3 | 24-param motion model, FD/DVARS, aCompCor, confounds.tsv |
| `batch04_first_level_analysis.py` | 4 | Design matrix + subject-level GLM (nilearn) |
| `batch05_second_level_analysis.py` | 5 | Group GLM + FDR/cluster-permutation correction |
| `batch06_visualization_reporting.py` | 6 | Glass-brain figures per subject and group |
| `batch07_reproducibility_artifact.py` | 7 | Bundles manifest + config + run summary + Dockerfile stub |

`config.yaml` is the single place to point the pipeline at your data —
no paths are hardcoded in any batch script. `utils.py` holds shared logging/
config helpers used by every batch.

## Prerequisites

1. **Python 3.11**, installed via the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate fmri-teaching-pipeline
   ```
   or via `pip install -r requirements.txt` in your own virtualenv.

2. **FSL** and **ANTs**, installed natively (not via Docker) and sourced on
   `$PATH`. Verify before your teaching session:
   ```bash
   which bet mcflirt flirt fnirt antsRegistration
   ```
   Batch 0 will also check for these and warn you if any are missing, and
   batch 2 will refuse to run rather than fail silently.

3. **A BIDS dataset.** For teaching, a small, fast public dataset is
   recommended, e.g. a 2-3 subject subset of
   [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114). Fetch with:
   ```bash
   pip install openneuro-py
   openneuro-py download --dataset ds000114 --target ./data/bids_dataset \
       --include sub-01 --include sub-02
   ```
   Point `config.yaml`'s `dataset.bids_root` at the download location.

## Running

```bash
# Full pipeline, all batches
python run_pipeline.py

# Just preprocessing (batches 0-2), for a first teaching session
python run_pipeline.py --until 2

# Resume from confound extraction onward (requires batches 0-2 already run
# in the same interactive session or re-run programmatically — see the
# docstring in run_pipeline.py for the state-passing convention)
python run_pipeline.py --from 3
```

For live teaching, it's often more effective to import and call each
batch's `run()` function directly in a Jupyter/IPython session, so learners
can inspect intermediate variables (design matrices, confound tables, etc.)
between batches rather than only seeing saved files.

## Expected runtime (2 subjects, 1 run each, teaching-scale dataset)

| Batch | Approx. time |
|---|---|
| 0 — Setup | <10 sec |
| 1 — QC | ~30 sec |
| 2 — Preprocessing | 5-15 min (ANTs SyN is the slow step; switch `preprocessing.normalization.method` to `fsl_fnirt` in config.yaml for a faster, less accurate alternative in time-constrained sessions) |
| 3 — Denoising | <1 min |
| 4 — First-level | 1-2 min per subject |
| 5 — Second-level | <1 min (FDR) or several minutes (cluster permutation) |
| 6 — Visualization | <1 min |
| 7 — Reproducibility bundle | <5 sec |

## Known simplifications (flagged explicitly for learners)

- **No fMRIPrep**: this pipeline reimplements a simplified equivalent from
  FSL/ANTs building blocks because fMRIPrep expects Docker/Singularity.
  fMRIPrep remains the better choice whenever containerization is available.
- **Placeholder WM/CSF mask** in batch 3 (intensity-threshold based, not a
  real FSL FAST segmentation) — replace before using aCompCor in a real
  analysis; the code says exactly where.
- **No distortion correction** (no fieldmaps assumed) — add FSL `topup` as a
  step in batch 2 if your dataset includes fieldmaps.

## Outputs

- `./logs/pipeline.log` — full run log across all batches.
- `./reports/` — QC tSNR maps, design matrix plots, glass-brain figures,
  environment manifest, and the final reproducibility bundle.
- `./results/` — thresholded group-level statistical maps.
- `./data/derivatives/` — per-subject preprocessed data, confounds, and
  first-level contrast maps.
