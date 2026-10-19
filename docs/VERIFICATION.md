# Submission Verification

Verified on 19 October 2026 using the project's Python 3.12 environment.

## Results

- Focused test runner: **196 tests run, 191 passed, 5 skipped**, no failures or errors.
- Skips: three MOVER archive checks requiring excluded source archives, one stored development-subset check, and one generated EDA-summary check. These require local clinical artifacts not distributed with the repository.
- All 80 notebook code cells parse, and saved outputs/execution counters are cleared.
- Submission structure validation passed for 74 Python modules/tests, 11 notebooks, 64 aggregate CSV tables, and 14 full-width report figures.
- The report compiled successfully from `report/main.tex` using the local Tectonic compiler. No external figure files are required.
- The data-free result viewer reproduced the recorded feature-comparison and neural-trial metrics from the committed CSVs.

## Change control

Executable syntax trees are unchanged in 73 of the 74 copied processing/model/test modules after documentation nodes are removed. In `bshi_audit.py`, only the generated explanatory report text changed; audit computations are unchanged. Sixty-eight API docstrings were added, and decorative comment banners and task-queue annotations were removed from comments and docstrings.

Notebook changes clear outputs, improve introductory explanations, make cohort root discovery portable, and make the environment check tolerate optional packages. The three bulk-processing permissions are disabled in the submission configuration; the machine-specific Java path is unset. Scientific thresholds and the archived feature-selection definitions are preserved.

The selected eight-feature context is documented separately and has not been trained. No new cohort evaluation or model training was performed while preparing this submission.

## Repository hygiene

Only selected code, configuration, aggregate evidence, documentation, and LaTeX source are included. Clinical data, identifiers, model weights, notebook outputs, logs, archives, machine probes, development queues, and caches are excluded. Newly generated result files are ignored unless explicitly reviewed and added to the aggregate-evidence allowlist. A credential-pattern check found no matching credentials in the curated files.

The original research workspace remains unchanged. The grading repository is an isolated copy, preserving the source work and dataset stores.
