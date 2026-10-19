# Data Access

Obtain both datasets from their official providers and follow their terms:
- [VitalDB](https://vitaldb.net/) and its [PhysioNet entry](https://physionet.org/content/vitaldb/).
- [MOVER access information](https://doi.org/10.1093/jamiaopen/ooad084). MOVER requires authorization and a data-use agreement.

Clinical data and patient-level artifacts are not included in this repository.

## Expected local files

| Location | Contents |
|---|---|
| `data/reference/vitaldb/` | `clinical_data.csv`, `clinical_parameters.csv`, and `track_names.csv` from VitalDB. |
| `data/reference/mover/` | Authorized `EPIC_EMR.tar.gz`, patient crosswalk `EPIC_MRN_PAT_ID.csv`, and the three `epic_wave_*_v2.tar.gz` waveform archives. |
| `data/manifests/` | Cohort, eligibility, linkage, and patient-split records generated locally. |
| `data/checkpoints/` | Processing progress for restart. |
| `data/parquet/` | Verified cases with separate waveform and numeric-MAP tables. |
| `data/derived/` | Windows, labels, beats, images, and independent parameters. |

Run the selection notebooks before ingestion. VitalDB ingestion uses the resulting `usable_cohort` and `stage_d_full_cohort` manifests. MOVER first audits linkage and its monitor-channel layout.

## Cohort rules

VitalDB selection requires ECG, invasive ART, PPG, numeric MAP, sufficient recording duration, limited missingness, and adequate MAP density. MOVER uses window-level coverage because capture is intermittent. The selected cohorts contain 3,185 VitalDB surgeries and 420 MOVER surgeries before beat-quality restrictions.

Both pipelines preserve missing intervals and keep all surgeries from one patient in one split. They verify written outputs and retain restart checkpoints. Full processing requires explicit permission flags in the configuration.

Do not commit archives, crosswalks, identifiers, waveform stores, patient-level predictions, model weights, or credentials. Only the selected aggregate tables belong in `results/`. More detail is provided in the [report](../report/main.tex).
