# Reproducing the work together

The setup uses a Python virtual environment (venv); `r0/f0` means repeat 0 / outer fold 0. CACHE and RUN denote the authorized private cache and run-output roots; keep them together.

These are the steps we use ourselves. Start with patient-free synthetic tests to learn the interfaces, then check real-data authorization, input versions and the shared evaluation contract together. The repository maintainer can help locate existing configurations; research decisions are discussed together.

We all share obligations under the data use agreement (DUA), institutional review board (IRB) requirements and the principal investigator's (PI) data agreement. Patient data, weights and predictions stay in authorized private directories; see [EXCLUDED](EXCLUDED.md) for unpublished-code boundaries.

## Choose the contract to reproduce

Time notation: postoperative day (POD), counted from zero on surgery day.

|Contract|Final handoff models|Primary metric|Repeats / outer evaluation|
|---|---|---|---|
|local cleaned-table contract (PI72-CLEAN)|encoding-enhanced Light Gradient Boosting Machine reference (GSAFE-LGB) (629 columns); full availability-gated tree reference for the local cleaned-table contract (T8-D5SAFE-LGB) (604 columns)|Post-op POD≥0 area under the receiver operating characteristic curve (AUROC)|one-based local-contract repeat identifiers (r1–5), fixed held-out-stay split per repeat|
|daily rolling prediction task (Task A) outcome-conditioned development cohort (formal) v2.6 / harness v3|daily reference (A-D5-LGB) (604 columns, this round's reference)|Total AUROC; post-op is key secondary|zero-based daily-task repeat identifiers (r0–4), 5-fold stay out-of-fold predictions (OOF) per repeat|

## Set up and run

Follow [ENVIRONMENT](ENVIRONMENT.md) to install a fresh Mac / Linux venv, then run the commands below from the repository root with that environment's `python`. See [DATA_LAYOUT](DATA_LAYOUT.md) for all 20 raw tables, clean-column order, cohort flow and output tree. Write only to authorised private roots.

Names used in the example: interpreter module-search-path environment variable (`PYTHONPATH`); interpreter environment variable disabling bytecode caches (`PYTHONDONTWRITEBYTECODE`); environment variable for the authorized read-only data root (`PF_DATA_ROOT`); environment variable for the authorized private cache root (`PF_CACHE_ROOT`); environment variable for the authorized private run-output directory (`PF_RUN_DIR`).

```sh
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export PF_DATA_ROOT=/authorized/pc4
export PF_CACHE_ROOT=/private/pf-nec-cache
export PF_RUN_DIR=/private/pf-nec-cache/runs
python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
python -m pf_nec.verify
python -m pf_nec.cli build-data
for r in 1 2 3 4 5; do
  python -m pf_nec.cli train --model GSAFE-LGB --repeat "$r"
  python -m pf_nec.cli train --model T8-D5SAFE-LGB --repeat "$r"
done
for r in 0 1 2 3 4; do
  for f in 0 1 2 3 4; do
    python -m pf_nec.cli train --model A-D5-LGB --repeat "$r" --fold "$f"
  done
done
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model T8-D5SAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --descriptive-ci
```

For comparable runs, we run sequentially with 2 numeric threads per process and resident set size (RSS)≤8 gibibyte in binary units (GiB); if a limit is reached, preserve the record and review the cause together. Build the cache once; resumption requires unchanged inputs, versions, absolute paths and paired CACHE/RUN, not migration of RUN alone. See [EVALUATION](EVALUATION.md) for single-arm descriptive confidence interval (CI), paired references with explicit family, 400-draw harness versus 2000-draw stratified bootstrap, JavaScript Object Notation (JSON) shape and tolerances.

## Scope already checked

The first code handoff rebuilt 13 matching tables and checked real-data equivalence for A-D5-LGB r0/f0 and GSAFE-LGB r1. The tiny A prediction difference is consistent with rounding; see [ENVIRONMENT](ENVIRONMENT.md). These checks do not establish full-repeat, Linux or Windows reruns or independent validation. Synthetic tests cover final-model paths, causal time invariance, evaluation and card application programming interface (API).

These limits keep resource conventions comparable and help avoid incomplete results from memory exhaustion. If a limit looks insufficient, we can agree a new research version and budget together while preserving the settings of the contract being reproduced. Fixed menus and comparison families reduce choices made after seeing results; new seeds do not make previously used splits into an untouched holdout.

For new work, see [TEAM_TASKS](TEAM_TASKS.md) and [TEAM_USAGE](TEAM_USAGE.md).
