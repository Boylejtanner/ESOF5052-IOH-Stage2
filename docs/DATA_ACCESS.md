# Dataset Access and Cohort Construction

## Data access

Obtain VitalDB and MOVER from their official providers under their current terms. MOVER requires authorization and a data-use agreement. A GitHub course repository is not a place to redistribute the source archives, clinical tables, patient crosswalks, identifiers, waveform stores, or patient-level model predictions.

- VitalDB: https://vitaldb.net/ and https://physionet.org/content/vitaldb/
- MOVER publication and access information: https://doi.org/10.1093/jamiaopen/ooad084

The report contains aggregate statistics and deidentified illustrative signal representations. The repository contains the code that produces the research artifacts, not the clinical dataset.

## Local directory contract

The pipeline creates and reads these ignored locations relative to the project root:

```text
data/
  reference/vitaldb/    Public clinical/track dictionaries obtained from VitalDB
  reference/mover/     Authorized MOVER waveform archives, EMR archive, and linkage tables
  manifests/           Local cohort, eligibility, patient split, and integrity records
  checkpoints/         SQLite progress and restart state
  parquet/             Verified canonical cases by dataset and case_id
  derived/             Ready windows, labels, beats, BSHI stores, and native parameters
```

Place the MOVER archives in `data/reference/mover/`. The linkage audit also requires the authorized `EPIC_EMR.tar.gz` archive there. The three configured MOVER waveform archive names are `epic_wave_1_v2.tar.gz`, `epic_wave_2_v2.tar.gz`, and `epic_wave_3_v2.tar.gz`. The linkage audit reads `data/reference/mover/EPIC_MRN_PAT_ID.csv`. VitalDB processing uses the local `usable_cohort` and `stage_d_full_cohort` manifests produced by its selection notebook, plus `clinical_data.csv`, `clinical_parameters.csv`, and `track_names.csv` under `data/reference/vitaldb`.

## VitalDB selection

The discovery step identifies surgeries with ART, ECG lead II, PPG, and an arterial-derived numeric MAP track. Quality checks require at least 30 minutes of recording, limited waveform missingness, and adequate numeric-MAP density. The measured selected cohort contains 3,185 surgeries from 3,085 patients. Waveforms are resampled to 100 Hz with anti-aliasing; numeric MAP retains its own timestamps.

Patient linkage uses the dataset's true patient identifiers, not surgery IDs. The download pipeline records source provenance, sampling rates, output hashes, verification results, and resumable progress for each selected case.

## MOVER selection

The linkage audit establishes which surgeries belong to the same patient. Archive discovery identifies the required waveform groups and arterial-line evidence. The decoder fails closed for unsupported monitor layouts rather than silently choosing an unverified MAP channel.

MOVER's intermittent capture makes whole-case completeness unsuitable. Selection instead uses valid prediction windows. The chosen rule requires at least 30 minutes of analyzable windows with at least 75% valid samples. The measured cohort contains 420 surgeries from 405 patients; later beat-level restrictions determine the ECG+ART and PPG-available analysis sets.

The one-pass implementation reads each compressed archive sequentially, parallelizes decoding, commits verified fragments, and finalizes cases without interpolating through long capture gaps. Processing completeness is verified separately from archive-read progress.

## Shared safeguards

- Split patients before model fitting; keep all surgeries and overlapping windows for one patient together.
- Preserve waveform gaps and explicit quality masks.
- Use future MAP only to construct outcomes, never as a prediction input.
- Commit case files atomically and verify hashes and schemas before reuse.
- Keep raw sources while any required output is incomplete.
- Treat full-download and bulk-processing switches as explicit user decisions.

## What is excluded from Git

Excluded: all of `data/`, `models/`, logs, caches, local environments, machine probes, checkpoints, archives, patient/case-level result tables, individual predictions, generated image folders, worker kits, development backups, and planning queues. Notebook output cells are cleared so they cannot carry identifiers or local paths into the submission.

Included result CSVs summarize model metrics, parameter definitions, cohort counts, distributions, and selection decisions. No row-level clinical record is needed for the grading quick start.
