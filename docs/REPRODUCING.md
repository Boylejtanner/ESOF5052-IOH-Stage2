# Reproduction Guide

## 1. Environment

The reference platform is Windows with Python 3.12. Scientific processing and synthetic model tests also use ordinary Python/PyTorch APIs, but the supplied Spark launch scripts and some environment checks target Windows.

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment\requirements-training.txt
$env:PYTHONPATH = "$PWD\notebooks"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
```

Dependency groups:
- `environment/requirements.txt`: scientific processing, EDA, acquisition libraries, and Jupyter.
- `environment/requirements-training.txt`: the base environment plus PyTorch for CPU tests and training APIs.
- `environment/requirements-gpu.txt`: the CUDA 12.6 package versions used for the recorded GPU experiments.
- `environment/requirements-spark.txt`: optional Spark 4.0.1; Java 17 is also required.

These manifests reflect the measured environment rather than an unrelated machine-wide package dump. Install the CPU/default training manifest for grading checks. For CUDA training, install the GPU manifest instead and verify the driver with `environment/verify_torch_cuda.py`.

## 2. Data-free review

```powershell
.\.venv\Scripts\python.exe scripts\review_results.py
.\.venv\Scripts\python.exe scripts\run_submission_checks.py
```

The result viewer reads the committed aggregate tables and identifies which scores belong to feature models and which belong to neural input trials. The check runner does not acquire clinical data, launch Spark, or start a training job on the cohort.

For the entire unit-test collection:

```powershell
$env:PYTHONPATH = "$PWD\notebooks"
.\.venv\Scripts\python.exe -m unittest discover -s notebooks\hemodynamics -p "test_*.py"
```

The full collection includes process, timing, filesystem, and optional cluster tests. It takes longer than the focused grading checks and may require optional dependencies. The focused verification result for this submission is recorded in `docs/VERIFICATION.md`.

## 3. Acquire the authorized datasets

Follow [DATA_ACCESS.md](DATA_ACCESS.md) before running acquisition notebooks. The repository deliberately contains no patient manifests, restricted crosswalks, waveforms, beat stores, or model checkpoints. Aggregate results can be inspected without those files, but reproducing the cohort and training requires them.

The safety flags `allow_full_archive_scan`, `allow_full_process`, and `allow_full_vitaldb_download` are disabled in the submission configuration. Enable only the required operation after reviewing local paths, disk reserve, and pilot results. Numerical feature/label definitions and the archived independence selection are retained. The machine-specific Java path is unset; use `BSHI_JAVA_HOME` for your own installation.

## 4. Notebook execution order

Open Jupyter from the root, select the project `.venv` kernel, and execute setup cells before processing cells.

| Order | Notebook | Main output or purpose |
|---|---|---|
| 1 | `00_environment_check.ipynb` | Inspect the selected Python environment. |
| 2 | `01-vitaldbaccess.ipynb` | VitalDB track discovery and candidate/usable cohort manifests. |
| 3a | `vitaldbwaveformdownload.ipynb` | Verified harmonized VitalDB cases. |
| 3b | `mover_cohort_builder.ipynb` | Verified MOVER patient linkage, eligible cohort, and harmonized cases. |
| 4 | `labels_and_baseline.ipynb` | Dataset snapshot, labels, and current-MAP reference. |
| 5 | `beats_extraction.ipynb` | Beat detections, morphology, quality, and stored waveforms. |
| 6 | `bshi_build.ipynb` | BSHI window index, tensor checks, and development subset. |
| 7 | `eda_report.ipynb` | Cohort, baseline, lead-time, quality, and shift analyses. |
| 8 | `independence_study.ipynb` | Native single-signal extraction and registered feature screening. |
| 9 | `train_bshi.ipynb` | Implemented BSHI recipe and optimizer experiments. |
| 10 | `spark_cluster.ipynb` | Optional Spark aggregation and streaming experiments. |

The training notebook requires CUDA for its recorded GPU workflow. Its existing recipes reproduce the implemented 25-value context, not the newly selected 21-value design. For a single workstation set `MODE = 'single_gpu'`; independent input-recipe trials can use the sequential fallback when no cluster is available.

Notebook outputs and execution counters are cleared in Git. Running acquisition, extraction, or training produces local files in ignored directories. Do not use **Run All** on a data-acquisition or evaluation notebook merely to inspect the submission.

## 5. Multiscale-context EDA

After the independent-parameter store and training feasibility screen exist:

```powershell
$env:PYTHONPATH = "$PWD\notebooks"
.\.venv\Scripts\python.exe -m hemodynamics.bshi_input_eda --root "$PWD" --bootstrap 200
```

The analysis creates `results/bshi_input_eda` with `exist_ok=False` so it cannot silently overwrite evidence. The submission already includes this aggregate evidence. To recompute, work in a separate experiment copy, move that copy's saved result directory aside, and provide the corresponding local derived data. Do not delete the committed evidence or evaluate the old test split again for tuning.

The EDA uses every third window within each case for fitting, an inner patient holdout for comparison, and all eligible validation windows for the shortlist. It reads no test outcomes. The original registered single-signal comparison and its plan remain separate evidence.

## 6. Spark setup

Install `environment/requirements-spark.txt` and Java 17. Process-local configuration is provided by `scripts/spark/env.ps1`. Override `BSHI_JAVA_HOME`, `BSHI_PYTHON`, `BSHI_BIND_HOST`, or `BSHI_MASTER_HOST` as required for the local machine. Review the launch script parameters before starting workers.

A second worker is optional. Copy the same code/dependencies and only data you are authorized to transfer. Do not expose data-serving endpoints or Spark control ports to the public internet. Machine probes, addresses, credentials, and worker logs remain outside version control.

## 7. Report

`report/main.tex` is the authoritative Stage 2 report. Upload it to Overleaf as `main.tex` and recompile from scratch. It contains all chart values, vector heatmap cells, and bibliography entries. No external figure or `.bib` files are required. The report was optimized after a compilation timeout by replacing per-cell PGFPlots calculations with embedded vector drawing operations.

The report selects eight physiological additions and a training methodology. The implementation status in the README and `configs/selected_features.json` makes explicit that integration and training of that context are the next project stage.
