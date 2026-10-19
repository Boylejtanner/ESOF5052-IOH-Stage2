# Quality-Aware Intraoperative Hypotension Forecasting

**ESOF5052: Big Data Machine Learning | Stage 2 | Tanner Boyle | Lakehead University**

This repository contains the Stage 2 implementation and evidence for forecasting intraoperative hypotension from VitalDB and MOVER. The project combines patient-level cohort construction, causal physiological feature extraction, exploratory analysis, and a Beat-Synchronous Hemodynamic Image (BSHI) training framework.

## Start here

| Grading item | Location |
|---|---|
| IEEE report with embedded LaTeX figures and bibliography | [report/main.tex](report/main.tex) |
| Implementation overview and module responsibilities | [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) |
| Installation and reproducible execution order | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| Dataset access, cohort rules, and excluded files | [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) |
| Measured EDA, optimizer, and neural trial results | [docs/RESULTS.md](docs/RESULTS.md) |
| GitHub publication and grader access | [docs/SUBMISSION.md](docs/SUBMISSION.md) |

## Stage 2 scope

Implemented and evaluated:
- MOVER archive discovery, patient linkage, window-level eligibility, and resumable ingestion.
- VitalDB modality selection, waveform retrieval, resampling, and quality checks.
- Patient-disjoint splits, numeric-MAP forecasting labels, beat extraction, morphology, and signal-quality indices.
- BSHI construction, input masking, model-side preprocessing, multi-task training, checkpointing, and evaluation utilities.
- Cohort EDA, baseline comparisons, single-signal feature screening, and the multiscale-context comparison.
- Optimizer selection and six-epoch BSHI input-recipe experiments using the implemented seven-plane image and 25-value context.
- Spark aggregation, feature-extraction, streaming-replay, and throughput utilities.

**The eight newly selected parameters have not yet been integrated into or trained with the BSHI network.** Their selection and the planned 21-value context are recorded in [configs/selected_features.json](configs/selected_features.json). The existing neural trial results must not be interpreted as results from that new input configuration. The report's training methodology specifies the next model integration.

## Selected design

The primary waveform inputs are ECG and ART. PPG is a matched ablation rather than a requirement for cohort inclusion. Within-signal morphology and per-signal quality are retained; cross-signal delays are excluded from the selected primary design.

The eight engineered parameters are:
1. ECG SDNN over 300 seconds.
2. Native ART mean-pressure slope over 60 seconds.
3. Median minimum post-peak ART pressure derivative over 60 seconds.
4. Native ART mean-pressure standard deviation over 300 seconds.
5. Native ART mean-pressure standard deviation over 60 seconds.
6. Pulse-pressure coefficient of variation over 300 seconds.
7. Recent minus five-minute pulse-mean-pressure median.
8. Native ART mean-pressure slope over 300 seconds.

Five MAP-baseline values plus these eight values and eight availability indicators form the planned 21-value context. The selected optimizer is AdamW with beta2 0.95, peak learning rate 0.0011, weight decay 0.01, batch size 128, and cosine decay with one warm-up epoch.

## Repository structure

```text
configs/                  Processing settings and selected feature specification
notebooks/                Ordered cohort, EDA, extraction, and training workflows
  hemodynamics/           Reusable processing/model modules and unit tests
environment/              Pinned analysis, training, CUDA, and Spark dependencies
scripts/                  Data-free result review, checks, and Spark launch scripts
results/                  Selected aggregate tables supporting the report
report/main.tex           Self-contained IEEE source with 14 vector figures
docs/                     Code guide, reproduction, results, and submission guidance
```

Raw archives, clinical reference files, patient linkage, waveform/beat stores, patient-level predictions, trained weights, logs, machine settings, and development backups are deliberately excluded. The same dataset-relative directory layout is used when authorized data are provisioned locally.

## Report compilation

Upload [report/main.tex](report/main.tex) to Overleaf as `main.tex`, choose it as the main document, and use **Recompile from scratch**. All figures are embedded vector drawing instructions; no PNG, PDF figure, or external bibliography file is required. The source has been optimized to avoid expensive heatmap calculations during compilation. The submitted PDF is compiled separately by the author.

## Scientific and access boundaries

Patients, not overlapping windows, are the split and uncertainty unit. Parameter selection is based on development data; historical test evaluations are identified separately. MOVER access requires the repository provider's authorization. Neither a private GitHub repository nor a course submission permits redistribution of restricted data. The implementation is research software, not a clinical decision-support device.
