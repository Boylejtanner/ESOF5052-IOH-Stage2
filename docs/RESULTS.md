# Stage 2 Results and Evidence

## Cohort and signal processing

The selected cohorts contain 3,185 VitalDB and 420 MOVER surgeries before beat-level restrictions. The common ECG+ART core contains 904,741 windows, of which one is excluded from feature-model analysis because the MAP baseline is missing. Requiring PPG reduces the available cohort disproportionately in MOVER, removing approximately 49% of its positive windows.

The `results/eda` tables summarize cohort selection, split counts, quality, labels, baseline discrimination, calibration, lead time, and domain shift. Counts of positive windows are not counts of independent hypotension episodes.

## Single-signal screening

The parameter catalogue contains 147 forecasting candidates. Coverage, partial correlation, incremental value beyond MAP, transfer direction, domain identifiability, and redundancy were considered. The strict screen retained six feasible candidates and forward selection identified a three-feature ECG+ART addition: ECG SDNN over 300 s, ART mean-pressure slope over 60 s, and ART minimum post-peak dP/dt over 60 s.

The registered protocol is preserved in [independence_study_plan.md](independence_study_plan.md). The corresponding tables remain under `results/independence`. Documentation cleanup does not rerun or retrospectively revise the frozen scientific selection. Archived hashes refer to the source/configuration used for those experiments, not to the documentation-only submission copy.

## Multiscale-context comparison

Patient-held-out inner training comparisons evaluated 15 feature sets and four learning strategies. A fixed shortlist was then evaluated on validation patients, with paired patient-cluster bootstrap intervals. The selected compact bundle adds five ART dynamics parameters to the original three.

| Feature-model inputs | Pooled validation AUROC | Pooled AUPRC | VitalDB to MOVER validation AUROC | Transfer AUPRC |
|---|---:|---:|---:|---:|
| Five MAP-baseline values | 0.8594 | 0.2280 | 0.7972 | 0.2241 |
| MAP plus the original three | 0.8684 | 0.2416 | 0.7857 | 0.1755 |
| MAP plus the eight selected additions | 0.8817 | 0.2707 | 0.8058 | 0.2180 |
| Large feature set, smoother trees | 0.8864 | 0.2780 | 0.8031 | 0.2064 |

The compact bundle's domain AUROC is 0.761, versus 0.991 for the large feature set. Its transfer improvement over the three-feature model is encouraging, while its advantage over MAP alone has a confidence interval spanning zero. Missing-history performance remains an important ablation.

Evidence: `results/bshi_input_eda/validation_comparisons.csv`, `paired_validation_uncertainty.csv`, `domain_identifiability.csv`, and `shortlist_before_validation.json`. These are gradient-boosting feature comparisons, not results from the selected BSHI network.

## Optimizer selection

The learning-rate range test established a moderate rate bracket. Nine optimizer/rate trials compared AdamW at beta2 0.999 and 0.95 with SGD-Nesterov. Four local configuration checks and a second seed for the leading three configurations produced 16 short runs.

The chosen setting is AdamW, beta2 0.95, peak rate 0.0011, weight decay 0.01, batch size 128, one warm-up epoch, and cosine decay. Batch 128 achieved mean validation AUPRC 0.2540 and AUROC approximately 0.8805 across two seeds. The leading AdamW configurations are close; the evidence supports a practical selection rather than a unique optimum.

Evidence: `results/train/sweep_trials.csv` and `results/train/lr_range.json`.

## Recorded BSHI input trials

Seven six-epoch trials used seed 5052, the selected AdamW configuration, and the implemented seven-plane/25-context architecture.

| Input recipe | Validation AUROC | Validation AUPRC |
|---|---:|---:|
| MAP context | 0.8693 | 0.2249 |
| ART-only recipe | 0.8833 | 0.2620 |
| Waveform planes | 0.8725 | 0.2288 |
| Waveforms plus MAP plane | 0.8789 | 0.2548 |
| Timing and morphology | 0.8827 | 0.2577 |
| ECG, ART, PPG, and SQI | 0.8802 | 0.2530 |
| ECG, ART, and SQI | 0.8824 | 0.2588 |

All recipes retain MAP-baseline context, including the recipe named `wave`. These experiments test the implemented feature recipes. They do not include the selected eight-feature addition. Evidence: `results/train/trials_20261009_111433.csv`.

## Next integration stage

The selected design has five MAP values, eight engineered values, and eight availability flags. The next implementation stage replaces the existing context, adds training-only imputation/scaling, updates masks and checkpoint metadata, and compares context ablations under matched protocols. The selected model has not yet been trained.

Source-only fitting on VitalDB and evaluation on MOVER is distinct from pooled learning on both hospitals. Feature design was informed by development data from both datasets. Already examined test sets must not be presented as a newly untouched confirmation cohort.
