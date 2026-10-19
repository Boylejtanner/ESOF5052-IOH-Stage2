# Independence study: pre-registered plan (QUEUE v5, Q1)

Status: written BEFORE any outcome (`y_event`, `delta_map`) of the new parameters was examined. The sha256 of this file is stored in
`results/independence/summary.json`. Later changes go only into "Deviations" at the end, with the reason. Seeds 5052 everywhere.

## 1. Question
Do parameters built from ONE independently measured signal (ECG, arterial pressure, PPG) predict intraoperative hypotension (IOH) beyond the current
mean arterial pressure (G0), transfer between hospitals (VitalDB <-> MOVER), and compare well with the old cross-signal parameters? Should PPG stay?

## 2. Definitions
- Signals: **E** = ECG, **A** = invasive arterial pressure (ART), **P** = PPG.
- **Independent (single-signal) parameter**: every number in it (beat detection, segmentation, polarity, normalization) comes from ONE signal. Normalization uses only
  the same signal's own past in the same case (causal).
- **Within-signal coupling**: two quantities of the same signal (for example ART pulse interval vs SBP); allowed, reported as its own sub-family.
- **Temporal correlation**: correlation of a signal's beat series with its own past (lag-1 autocorrelation, variance, their Kendall trends, DFA alpha1, sample entropy,
  LF/HF spectral ratio) = critical-slowing-down indicators (Scheffer et al. 2009).
- **Cross-signal (old)**: anything mixing two signals (PTT, `art_ppg_corr`, `sqi_cross`, HR-to-MAP coupling, ART segmented on ECG beats, PPG polarity chosen with ECG).
  Computed ONLY as the comparison arm (old G1-G4 columns of `window_features_v1.parquet` plus `x_hr_map_corr_l60`).
- **G0** = `map_current, map_mean_60, map_slope_60, map_min_60, map_std_60`. **IND-A, IND-E, IND-P, IND-EA, IND-EAP** = compact sets frozen in Q6.

## 3. Data
Windows of the `ea_core` protocol (904,741; 28,278 events): E/A parameters on core windows; anything with P on the `in_full` subset (795,759; 24,573 events).
Splits `split_pooled` (patient level). Training = screening and selection; validation = confirmation once (Q6); test = ONCE in Q7 (Q8 test models are part of
that single pass). MOVER test has 64 patients / 419 events, so its confidence intervals are wide.

## 4. Parameter catalogue (`results/independence/table_param_catalogue.csv`)
Native beat series per signal, each from its own signal only (`hemodynamics/independent_features.py`, version `if1`):
ECG R peaks (`beats.detect_r_peaks`, polarity from a running ECG-only vote); ART pulses (`beats.detect_pulses(kind="art")`); PPG pulses
(`detect_pulses(kind="ppg")`, polarity from a running PPG-only steeper-upstroke vote). Strict causality: chunks end at each window end, detection sees only samples
<= that end, a beat is used once its cycle is complete plus 1 s. Lookbacks L60 (>= 50% covered, >= 20 beats) and L300 (>= 70%, >= 100 beats), otherwise NaN (never 0).
`_sn` = ratio to (or difference from) the case's expanding median of earlier windows (current included, >= 11 windows ~ 5 min).
147 candidates: **A 88** (level/shape 13, level/shape `_sn` 11, variability incl. PPV/SPV 14, trend 16, temporal 32, within-signal coupling 2);
**E 32** (HR 1, `_sn` 5, HRV/ectopy/RSA/EDR 14, trend 3, temporal 9); **P 27** (level 3, `_sn` 5, variability incl. PVI 8, trend 4, temporal 6, coupling 1).
Meta (not candidates): own-quality and coverage columns (`a_q_l60`, `e_q_l60`, `p_q_l60`, `*_cover_l60`), `a_map_l60`, absolute PPG amplitude and AC/DC.

## 5. Hypotheses (operational)
- **H1** single-signal ART parameters predict IOH beyond G0 in BOTH datasets. Supported if >= 1 A parameter is feasible (F1-F6) AND G0 + IND-A beats G0 on validation
  (pooled paired Delta AUROC 95% CI > 0) with VitalDB-train -> MOVER-validation Delta AUROC >= 0.
- **H2** ECG-only HRV / temporal parameters (L300) add information beyond G0 and are less hospital-specific when self-normalized. Supported if (a) >= 1 E parameter
  passes F2, F3 and F4 AND (b) the median domain AUROC of the five `e_*_sn` parameters is >= 0.05 lower than that of their raw versions.
- **H3** critical-slowing-down indicators of the beat-to-beat ART series rise before IOH. Primary indicators `a_map_ac1_l300` and `a_map_rsd_l300` (secondary: SBP, PP,
  pulse interval, ECG RR). Training windows, per dataset: event windows binned by `time_to_event_sec` (0-60, ..., 240-300 s) minus the mean of non-event windows of
  the same dataset and current-MAP decile. Supported if, for at least one primary indicator, in BOTH datasets the 0-60 s difference has a patient-bootstrap 95% CI
  above 0 AND exceeds the 240-300 s difference. Partially supported if this holds in VitalDB only.
- **H4** PPG-only parameters add nothing beyond E+A and stay hospital-specific when self-normalized. Supported if the PPG rule (section 8) says drop AND the median
  domain AUROC of P candidates is > 0.70 for raw AND for `_sn` parameters.
- **H5** G0 + IND-EA beats G0, is non-inferior to old G4 and transfers VitalDB -> MOVER at least as well as G0. Supported if S1 and S2 both hold.

## 6. Feasibility gates (training split; 5-fold `GroupKFold` by patient, out-of-fold; every 3rd training window)
- **F1 coverage**: non-missing in >= 90% of its window set (core for E/A, full for P) and >= 85% of MOVER windows.
- **F2 partial correlation**: Spearman with `delta_map` after rank-residualizing parameter and outcome on G0; 200 patient-bootstrap; VitalDB 95% CI excludes 0 and the
  MOVER point estimate has the same sign.
- **F3 incremental value**: `_model("lr")` (median imputation) G0 + parameter vs G0, pooled out-of-fold; paired 200 patient-bootstrap; pass if Delta AUROC has CI lower
  bound > 0 and BH-adjusted p < 0.05 (normal approximation from the bootstrap SE; BH over all 147 candidates), OR the same for Delta AUPRC (separate BH family).
- **F4 transfer**: lr fit on VitalDB training, evaluated on MOVER TRAINING windows (MOVER validation stays clean): Delta AUROC >= -0.002.
- **F5 hospital identifiability**: orientation-free univariate domain AUROC max(AUC, 1 - AUC) (VitalDB vs MOVER, training) <= 0.70. When a raw parameter fails and its `_sn`
  version passes, only the `_sn` version can be used (both are catalogued candidates, so this follows automatically).
- **F6 redundancy**: |Spearman| with `map_current` < 0.5 (pooled training) AND the parameter is the representative of its redundancy cluster (average linkage on
  1 - |rho| over all 147 candidates, cut at |rho| >= 0.8): among members passing F1-F5, the one with the largest F3 Delta AUPRC.
- **F7 robustness (warning only)**: the pooled partial-correlation sign is unchanged in the top and bottom tertile of the parameter's own signal quality.
- **Feasible** = F1-F6 all pass. Every metric is recorded even when a gate fails. Boosting check (`_model("hgb")`, 3 folds, `subsample=3`) for each cluster
  representative passing F1-F5: descriptive, it orders the selection pool. Incremental value in the gray zone and in the 180-300 s lead-time bins: descriptive only.

## 7. Compact sets, confirmation and success criteria
Pools: IND-A = feasible A, IND-E = feasible E, IND-P = feasible P, IND-EA = A + E, IND-EAP = A + E + P (full windows). Greedy forward selection on pooled training
out-of-fold hgb predictions (3-fold, every 3rd window), starting from G0, adding the parameter with the largest Delta AUPRC; stop when the best step has Delta AUROC < 0.001
AND Delta AUPRC < 0.002, or at 12 parameters; pool capped at the top 20 by the boosting check. Cross-check: L1-logistic path, overlap reported. An empty pool gives an
empty set (that model equals G0 and is reported as "no feasible parameters"). Confirmation ONCE on validation (pooled and VitalDB-train -> MOVER validation; paired
1,000 patient-bootstrap vs G0 and vs old G2/G4). Sets, learners and hyperparameters are then frozen into `configs/hemodynamics.json` key `independence`.
- **S1 (beats current MAP)**: pooled test paired Delta AUROC of G0 + IND-EA (hgb) vs G0 (hgb) has 95% CI > 0 AND Delta AUPRC > 0, AND VitalDB-train -> all MOVER
  Delta AUROC >= 0.
- **S2 (better than old)**: G0 + IND-EA non-inferior to old G4 (Delta AUROC >= -0.003, pooled test, full windows) AND lower multivariate domain AUROC AND
  VitalDB -> MOVER AUROC >= old G4.
- **S3 (descriptive)**: IND-EA without G0 vs G0.

## 8. PPG decision rule
PPG stays in the main model only if G0 + IND-EAP beats G0 + IND-EA on validation (paired Delta AUROC CI > 0) in pooled AND VitalDB -> MOVER, with every P parameter of
IND-EAP passing F5. Otherwise PPG is removed from the main model and kept only for the secondary noninvasive study (Q8: IND-E + IND-P, no ART, no MAP).

## 9. Analysis order
Q3 build and QC (no outcomes) -> Q4 correlation atlas (training, every 3rd window) -> Q5 feasibility screen (training) -> Q6 selection (training), confirmation
(validation, once), freeze -> Q7 + Q8 single test pass (1,000 patient-bootstrap, paired) -> Q9 literature -> Q10 documents.
Learners: `eda_models._model("lr" | "hgb")`, `fit_predict(subsample=3)`. Metrics: `labels._metrics`; patient-cluster bootstrap with the same resampled patients for paired
differences.

## 10. Deviations
Recorded 2026-10-09 during the Q3 build, BEFORE any outcome (y, delta_map) of the full table was examined:
1. Lead-time bins: an event needs >= 60 s inside the 300 s horizon, so onset is at most 240 s ahead; the 240-300 s bin is empty. H3 compares 0-60 s with
   180-240 s (the farthest bin) and Q5/Q7 lead-time tables use 0-60 ... 180-240 s (`eda_models.LEAD_BINS`).
2. DFA alpha1: F(n) is divided by sqrt(1 - 4/n^2) (exact white-noise small-n bias of linear detrending); a constant shift, ranks unchanged (unit test: 0.5 +- 0.1).
3. Pooled bootstraps resample patients within dataset (stratified), so MOVER is never absent from a resample.
4. `split_pooled` and `split_within_dataset` are independent patient splits (e.g. 90,925 VitalDB pooled-TRAIN windows are within-dataset TEST). Screening,
   selection and confirmation use `split_pooled` only. In Q7 the `_settings` transfer settings (VitalDB within-train -> all MOVER; MOVER within-train -> VitalDB
   within-test) are kept for comparability with the old numbers but are NOT clean (their evaluation windows include pooled training/validation windows used
   in screening/selection). Added clean settings: VitalDB pooled-train -> MOVER pooled-test and MOVER pooled-train -> VitalDB pooled-test. The S1/S2 transfer
   clauses are evaluated on the pre-registered setting AND the clean one; if they disagree the claim is reported as not robust.
5. Q8(d) noninvasive model = IND-E + IND-P; if that union has < 3 parameters, all E and P candidates passing F1 and F5 are used instead (unselected,
   descriptive, labelled as such). Q8(a) raw vs self-normalized PPG = G0 + IND-EA + all raw P candidates passing F1 vs G0 + IND-EA + all `p_*_sn` passing F1.
6. Boosting check "subsample=3" is realized by the every-3rd-window training frame (no further subsampling inside the 3-fold out-of-fold fits).
7. Figure 01 orders each signal by its own clustering; F6 uses the pre-registered clustering over all 147 candidates.
8. Figure 11 example events: drawn at random (seed 5052) among training events with >= 18 windows in the 600 s before onset (no model-based choice); the
   alarm threshold is the 90%-specificity threshold of G0 + IND-EA (hgb) on pooled validation; the risk trace is in-sample (training case), illustration only.
9. L1 cross-check uses `LogisticRegression(l1_ratio=1, solver="liblinear")` (sklearn 1.9 deprecates `penalty=`).
10. (Q6, after selection) Greedy selection added no PPG parameter to IND-EAP, so IND-EAP = IND-EA; the PPG rule was therefore also checked with the Q8a models
    "G0 + IND-EA + PPG raw" and "+ PPG self-normalized" (all P parameters passing F1). The decision is unchanged (PPG removed).
11. (Reporting) `criteria()` marks H3 "supported" when MAP variance rises in both datasets; lag-1 autocorrelation rose in VitalDB only, so the report states H3 as PARTIALLY supported.
12. (Q3) One window (VitalDB case 5669) has missing G0 and is excluded by `load_ladder_frame` exactly as in the old ladder: 904,740 analysed windows (904,741 built).
