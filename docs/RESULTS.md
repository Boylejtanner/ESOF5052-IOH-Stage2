# Results Summary

## Feature selection

The selected context adds eight ECG/ART measurements to five MAP-baseline values. The parameter definitions are in [selected_features.json](../configs/selected_features.json).

| Gradient-boosting inputs | Pooled validation AUROC | VitalDB to MOVER validation AUROC |
|---|---:|---:|
| Five MAP values | 0.8594 | 0.7972 |
| MAP plus three additions | 0.8684 | 0.7857 |
| MAP plus eight selected additions | 0.8817 | 0.8058 |
| Large set with smoother trees | 0.8864 | 0.8031 |

The compact set has hospital-identification AUROC 0.761, versus 0.991 for the large set. This favored the compact selection. Transfer superiority over MAP alone is not established by its confidence interval.

Evidence: `results/bshi_input_eda/` and the original single-signal screen in `results/independence/`.

## Optimizer and neural experiments

AdamW was selected with beta2 0.95, learning rate 0.0011, weight decay 0.01, batch size 128, and cosine decay. Its two-seed mean validation AUPRC was 0.2540. Evidence: `results/train/sweep_trials.csv` and `lr_range.json`.

| Recorded six-epoch neural recipe | Validation AUROC | AUPRC |
|---|---:|---:|
| MAP context | 0.8693 | 0.2249 |
| ART-only | 0.8833 | 0.2620 |
| Waveform planes | 0.8725 | 0.2288 |
| Waveforms plus MAP | 0.8789 | 0.2548 |
| Timing and morphology | 0.8827 | 0.2577 |
| ECG, ART, PPG, and SQI | 0.8802 | 0.2530 |
| ECG, ART, and SQI | 0.8824 | 0.2588 |

These neural runs retain MAP context and use the implemented 25-value context. **They do not use the newly selected eight additions.** Integrating and training the selected 21-value context is the next stage.

See [report/main.tex](../report/main.tex) for cohort counts, methods, uncertainty intervals, figures, and interpretation.
