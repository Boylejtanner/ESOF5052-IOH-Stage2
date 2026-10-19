# Reproducing the Work

## Setup

Use Python 3.12 and run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment\requirements-training.txt
$env:PYTHONPATH = "$PWD\notebooks"
$env:OMP_NUM_THREADS = "4"
```

Use `requirements-gpu.txt` instead for the recorded CUDA 12.6 setup. Spark is optional and needs `requirements-spark.txt` plus Java 17. Keep machine addresses and credentials local.

## Without clinical data

```powershell
.\.venv\Scripts\python.exe scripts\review_results.py
.\.venv\Scripts\python.exe scripts\run_submission_checks.py
```

These show saved aggregate results and run synthetic checks. They do not download data or train on the clinical cohort.

## With authorized local data

Follow [DATA_ACCESS.md](DATA_ACCESS.md), start Jupyter, and select the project `.venv` kernel. Run setup cells before processing cells.

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `00_environment_check.ipynb` | Check the Python environment. |
| 2 | `01-vitaldbaccess.ipynb` | Select the VitalDB cohort. |
| 3 | `vitaldbwaveformdownload.ipynb`, `mover_cohort_builder.ipynb` | Retrieve or decode the two datasets. |
| 4 | `labels_and_baseline.ipynb` | Create forecasting labels and the MAP baseline. |
| 5 | `beats_extraction.ipynb` | Extract beats, morphology, and quality. |
| 6 | `bshi_build.ipynb` | Build the image inputs and check their values. |
| 7 | `eda_report.ipynb`, `independence_study.ipynb` | Reproduce cohort and single-signal analyses. |
| 8 | `train_bshi.ipynb` | Run the implemented BSHI input recipes. |
| 9 | `spark_cluster.ipynb` | Run optional processing/replay benchmarks. |

Bulk download/processing flags are disabled by default in `configs/hemodynamics.json`. Review disk space and pilot results before enabling them. For one GPU, set the training notebook's `MODE` to `single_gpu`. Its existing recipes do not yet use the new eight-feature context.

The multiscale feature comparison can be run after the native feature store exists:

```powershell
.\.venv\Scripts\python.exe -m hemodynamics.bshi_input_eda --root "$PWD" --bootstrap 200
```

Run this in a separate experiment copy: the command refuses to overwrite the existing `results/bshi_input_eda` directory. Keep the committed results and do not reuse historical test outcomes for tuning.

For all unit tests, use `python -m unittest discover -s notebooks\hemodynamics -p "test_*.py"` with the project interpreter and `PYTHONPATH` above. Some checks require optional data or Spark.

The [report source](../report/main.tex) is self-contained and can be uploaded to Overleaf as `main.tex`.
