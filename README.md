# Quality-Aware Intraoperative Hypotension Forecasting

**Tanner Boyle | ESOF5052 | Lakehead University**

This project predicts low blood pressure during surgery using ECG, arterial pressure (ART), and optional photoplethysmography (PPG). It uses the VitalDB and MOVER datasets to study whether a model learns useful physiological changes rather than differences between hospitals.

## How it works

1. Select eligible surgeries and keep each patient's recordings in the same data split.
2. Resample waveforms to 100 Hz, preserve missing intervals, and create future-MAP labels.
3. Detect beats, measure waveform shape and signal quality, and build beat-synchronous images.
4. Compare feature sets and train models to predict hypotension risk, pressure drop, and burden.
5. Compare pooled learning with training on VitalDB and evaluating on MOVER.

## Completed work

Cohort selection, signal processing, labels, BSHI construction, EDA, feature selection, optimizer comparisons, and short neural input-recipe experiments are implemented. Spark processing and replay utilities are also included.

The recorded neural experiments use the existing seven-plane image and 25-value context. **The newly selected eight-parameter context has not yet been integrated into or trained with BSHI.** Its next-stage specification is in [selected_features.json](configs/selected_features.json).

The selected additions describe ECG beat-interval variability and ART pressure slope, pulse decay, short/long variability, pulse-pressure variation, and recent pressure drift. The planned input contains five MAP values, eight selected measurements, and eight availability flags. AdamW with learning rate 0.0011, beta2 0.95, batch size 128, and cosine decay is the selected optimizer configuration.

## Files

| Location | Contents |
|---|---|
| [notebooks/](notebooks/) | Cohort, processing, EDA, and training workflows |
| [notebooks/hemodynamics/](notebooks/hemodynamics/) | Reusable Python modules and tests |
| [configs/](configs/) | Processing settings and selected features |
| [results/](results/) | Saved aggregate results |
| [report/main.tex](report/main.tex) | Full methodology, results, references, and embedded figures |
| [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) | Short guide to the code |

## Run the project

Use Python 3.12. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment\requirements-training.txt
.\.venv\Scripts\python.exe scripts\review_results.py
```

The result viewer needs no clinical data. To run the synthetic checks:

```powershell
.\.venv\Scripts\python.exe scripts\run_submission_checks.py
```

For the data-processing notebooks, obtain the datasets under their access terms and follow [REPRODUCING.md](docs/REPRODUCING.md). Raw data, patient identifiers, weights, logs, and notebook outputs are excluded from Git. Read [DATA_ACCESS.md](docs/DATA_ACCESS.md) for the expected local files.

Upload [report/main.tex](report/main.tex) as `main.tex` to compile the report in Overleaf. All figures and bibliography entries are embedded; no external images are needed. This is retrospective research software, not a clinical decision-support device.
