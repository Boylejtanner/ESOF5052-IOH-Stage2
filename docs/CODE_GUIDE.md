# Code Guide

## Design and execution flow

The notebooks are orchestration and inspection entry points. Reusable logic lives in `notebooks/hemodynamics`; tests are colocated with those modules. Keeping this layout preserves the import and root-discovery conventions used by the measured experiments.

The processing flow is:

1. Establish dataset-specific patient identity and candidate surgery eligibility.
2. Decode or retrieve the required signals, harmonize waveform sampling, and commit verified case files.
3. Generate ready-window indices and MAP-derived forecasting labels.
4. Detect beats and extract morphology, timing, and quality measurements.
5. Construct BSHI inputs or independently detected single-signal parameters.
6. Evaluate features, train comparison models, and save patient-aware metrics.

## Cohort construction and storage

| Module | Responsibility and important behavior |
|---|---|
| `mover_signals.py` | Decodes one MOVER XML member. Resolves waveform groups, calibration, timestamps, and numeric MAP without loading a complete archive. |
| `mover_fast.py` | Streams each archive once, bounds the payload queue, parallelizes decoding, merges case fragments, and records progress. Capture gaps remain missing. |
| `mover_pipeline.py` | Audits patient linkage, inventories archives, performs pilot validation, selects eligible patients, and verifies processing completeness. Also retains the sequential reference implementation. |
| `vitaldb_pipeline.py` | Retrieves selected tracks from VitalDB with bounded requests, retries, restart checkpoints, native-to-100-Hz resampling, and case-level verification. |
| `canonical.py` | Defines the case-storage interface, atomic metadata writes, checksum verification, disk-space protection, and deterministic patient splits. |
| `windows.py` | Implements the common ready-window validity, numeric-MAP, and arterial-plausibility rules and the eligibility ladder. |
| `unified.py` | Combines the two sources into one case/window index while preserving dataset identity and patient partitions. |
| `snapshot.py` | Records the dataset definition and checksummed processing configuration for reproducibility. |
| `validate_data.py` | Checks persisted selected cases against expected schemas, manifests, and completeness requirements. |

A verified case has a 100-Hz waveform table and a separate native-time numeric-MAP table. Numeric MAP is not inflated to 100 Hz by long-gap forward filling. A checkpoint records completed cases so an interrupted run can resume without treating partial files as complete.

## Labels and signal features

| Module | Responsibility and important behavior |
|---|---|
| `labels.py` | Creates sustained-hypotension onset, pressure-drop, and AUT65 targets from numeric arterial MAP; implements the MAP baseline and patient-cluster metrics. |
| `beats.py` | Detects ECG R peaks and arterial/PPG pulse landmarks; computes beat morphology, cross-signal comparison features, and SQIs. |
| `beats_pipeline.py` | Applies beat extraction case by case, writes compressed feature/waveform tables, and invalidates outputs when detector settings change. |
| `beats_qc.py` | Selects representative cases, checks extraction agreement, and plots detection overlays for quality review. |
| `case_quality.py` | Identifies unusable signal channels, delayed PPG, and nonfunctional ECG before beat-image eligibility is decided. |
| `independent_features.py` | Detects each signal independently and calculates causal 60-second/300-second features. It does not use ECG to segment native ART or PPG parameters. |

`art_window`, `ecg_window`, and `ppg_window` summarize only completed beats available at the prediction time. Missing or insufficient histories produce missing features rather than fabricated measurements. The test suite includes causality checks that alter future samples and verify earlier outputs remain unchanged.

## BSHI representation and training

| Module | Responsibility and important behavior |
|---|---|
| `bshi.py` | Builds the window index; assembles seven 60-by-200 planes; handles waveform/context masks, case caching, and compact encoding. |
| `bshi_checks.py` | Verifies finite values, masking, and expected tensor structure, and displays actual example images. |
| `bshi_bench.py` | Measures loader throughput and constructs a bounded development subset. |
| `bshi_audit.py` | Audits label coverage, context ranges, recipe masks, patient balance, and training readiness. |
| `bshi_store.py` | Writes memory-mapped shards so shuffled training does not reopen a case file for each window. |
| `bshi_torch.py` | Provides importable PyTorch dataset classes compatible with Windows worker spawning. |
| `bshi_data.py` | Applies training-derived context/target statistics, constructs loaders, and attaches observation masks. |
| `bshi_aug.py` | Implements controlled signal corruption and missingness transformations for robustness experiments. |
| `bshi_raw.py` | Supplies raw-waveform and spectral representations for matched baseline experiments. |
| `bshi_model.py` | Defines the GroupNorm residual CNN, context MLP, SQI gate, fusion layer, and prediction heads. |
| `bshi_train.py` | Defines optimizer groups, warm-up/schedules, masked multi-task loss, early stopping, and exact-resume checkpoints. |
| `bshi_run.py` | Connects data, model configuration, training, evaluation, and experiment output directories. |
| `bshi_measure.py` | Measures GPU throughput/memory and learning-rate range behavior. |
| `bshi_sweep.py` | Executes short optimizer, learning-rate, batch-size, and schedule trials with recorded validation metrics. |
| `bshi_dryrun.py` | Runs a bounded training experiment to verify the full data path and resource use. It is an engineering experiment, not the final model evaluation. |
| `bshi_notebook.py` | Supplies experiment identifiers, notebook summaries, and optional distributed launch commands. |
| `bshi_progress.py` | Records and presents training status, resource use, metrics, and learning curves. |
| `bshi_ddp.py` | Supports optional multi-process gradient synchronization using Gloo. Single-GPU training does not require it. |

The implemented training recipes have seven image planes and 25 context values. The image branch uses GroupNorm rather than batch-dependent normalization. The SQI gate uses quality bands and beat masks to modulate waveform branches before fusion.

The loss combines event binary cross-entropy, Huber loss for pressure drop, and a two-part AUT65 objective. Observation masks prevent missing future targets from contributing to regression loss. Checkpoints include model, optimizer, scheduler/stopping state, and random-generator states.

The selected 21-value context is a separate Stage 2 design specification in `configs/selected_features.json`. Loading that JSON alone does not modify the implemented network. Integration requires coordinated changes to context construction, training-only imputation/scaling, masks, projection size, and checkpoint metadata.

## EDA and feature selection

| Module | Responsibility and important behavior |
|---|---|
| `eda_models.py` | Builds the fixed MAP-to-full-feature baseline ladder and measures discrimination, calibration, lead-time behavior, and domain shift. |
| `eda_metrics.py` | Produces cohort/label/quality summaries, model comparison tables, figures, and readiness evidence. |
| `independence_study.py` | Runs correlation analysis, feasibility gates, grouped selection, validation confirmation, and the registered fixed test comparison. |
| `bshi_input_eda.py` | Tests physiological bundles on an inner patient holdout, freezes a shortlist, and compares it on validation patients without evaluating test outcomes. |
| `figstyle.py` | Enforces consistent labels, units, and readable scientific plot annotations. |

In the multiscale-context EDA, missing features are retained and handled by histogram boosting. Model comparisons use identical evaluation windows. Domain classifiers predict hospital identity to detect combinations that encode acquisition practice. The reported confidence intervals resample patients rather than treating repeated windows as independent observations.

## Spark and systems experiments

| Module or script | Responsibility |
|---|---|
| `cluster.py` | Creates Spark sessions, inspects workers, and dispatches independent trials. |
| `spark_jobs.py` | Implements aggregate checks, feature-extraction workloads, and streaming replay. |
| `cluster_probe.py` | Measures the local machine and optional inter-worker link. Generated host-specific output is not included in Git. |
| `scaling_estimate.py` | Uses machine/link measurements to compare synchronous multi-node training with independent trials. |
| `scripts/spark/` | Configures process-local Java/Python settings and starts or stops the owned Spark processes. |

## Documentation and tests

Source comments explain invariants, units, masks, and non-obvious algorithm choices. Decorative separator banners and task-queue references have been removed from comments/docstrings in the submission copy. Public API docstrings were added to the principal storage, extraction, model, and feature-selection interfaces. Executable Python syntax trees were compared with the research source after removing documentation nodes to verify that comment cleanup did not change algorithms.

The tests use synthetic waveforms, mocked acquisition, and isolated fixture directories. `scripts/run_submission_checks.py` runs a focused cross-pipeline subset; the complete `test_*.py` collection remains available for deeper inspection. No test or training result is replaced with a claimed result from the newly selected context.
