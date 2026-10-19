# Code Guide

The notebooks run the workflow; reusable processing and modeling code lives in `notebooks/hemodynamics`. Comments explain each main calculation, decision, and data-handling step.

| Modules | What they do |
|---|---|
| `mover_signals`, `mover_pipeline`, `mover_fast` | Decode MOVER recordings, check patient linkage, select eligible cases, and support restart. |
| `vitaldb_pipeline` | Retrieve selected VitalDB tracks, resample waveforms, and verify case outputs. |
| `canonical`, `snapshot`, `validate_data` | Store cases, record checksums, and check completeness. |
| `windows`, `unified`, `labels` | Build shared window rules, patient partitions, and MAP-derived outcomes. |
| `beats`, `beats_pipeline`, `beats_qc`, `case_quality` | Detect beats, describe pulse shape, measure signal quality, and check detections. |
| `independent_features`, `independence_study` | Extract each signal's own parameters and screen their value beyond current MAP. |
| `eda_models`, `eda_metrics`, `bshi_input_eda`, `figstyle` | Compare features, summarize the datasets, and produce analysis figures. |
| `bshi`, `bshi_checks`, `bshi_audit`, `bshi_bench` | Build and validate beat images, masks, context, and loading performance. |
| `bshi_store`, `bshi_torch`, `bshi_data` | Store compact inputs and prepare normalized model batches. |
| `bshi_model`, `bshi_train`, `bshi_run` | Define the network, calculate masked losses, train, checkpoint, and evaluate. |
| `bshi_measure`, `bshi_sweep`, `bshi_dryrun` | Measure hardware use and compare optimizer/input configurations. |
| `bshi_aug`, `bshi_raw` | Support corruption tests and raw/spectral input comparisons. |
| `bshi_notebook`, `bshi_progress` | Manage experiment setup and display saved training progress. |
| `cluster`, `spark_jobs`, `bshi_ddp`, `cluster_probe`, `scaling_estimate` | Run optional Spark/distributed experiments and measure their throughput. |
| `test_*.py` | Check processing and model behavior with synthetic inputs and local fixtures. |

Important conventions:
- A patient stays in one partition even if they have several surgeries or overlapping windows.
- Future MAP creates labels but never prediction inputs.
- Signal gaps stay missing and are represented by validity masks.
- Stored files are verified before being reused after a restart.
- Neural trials use the implemented 25-value context; the selected 21-value replacement is the next integration step.

See the [report](../report/main.tex) for algorithm details and the [execution guide](REPRODUCING.md) for notebook order.
