# Project Passion Fruit: NEC final-model code handoff

This repository contains the research code, frozen rules, synthetic tests and summary results for predicting necrotizing enterocolitis (NEC) in neonates with congenital heart disease, using the PC4 registry. **It contains no data, trained weights or predictions.** You need authorised access to the same PC4 data (same DUA/IRB) to reproduce anything. All results are cross-validated on the development data; **there is no independent or external validation yet.**

The other documents are in Chinese (`README.md`, `AGENTS.md`, `docs/`). This page is the English entry point and summarises them; where the two differ, the Chinese documents and the code are authoritative.

## What is included

| Contract | Models kept | Primary metric | Repeats / outer validation |
|---|---|---|---|
| PI72-CLEAN (local clean-table contract) | `GSAFE-LGB` (G-safe, 629 columns); `T8-D5SAFE-LGB` (single-day reference, 604 columns) | AUROC on POD ≥ 0 days (post-op) | r1–5, one fixed held-out stay split per repeat |
| Task A formal (v2.6 / harness v3) | `A-D5-LGB` (single day, 604 columns) | Total AUROC; post-op AUROC is the key secondary | r0–4, 5-fold stay-level out-of-fold per repeat |

- **The two contracts differ** in rows, labels, event definition, sampling and validation scheme. Never compare AUROC values across them.
- **PI72-CLEAN** rebuilds the received clean table's rules locally. It does **not** recover the final scoring design behind the original manuscript's AUROC 0.79.
- **Task B** was not retrained in this round.
- **Non-promoted methods** are under `src/explore/` (see `explore/README.md`). They did not pass the promotion gate.
  - T3 SHAP interpretation by post-op day and lead time.
  - I7 diagnostics: the 8-background-variable model and the learning-curve method.
  - Aim 1b target-trial-emulation scaffolding: measurement, eligibility, time zero and positivity diagnostics. No treatment effects are estimated.

## Accepted results (aggregates only)

| Contract / source report | Model | Primary metric [descriptive 95% interval] | Supplementary |
|---|---|---|---|
| PI72-CLEAN, R9 (5 repeats) | G-safe-LGB | Post-op AUROC .7329 [.7020, .7622] | Total .8095; post-op AP .0639 |
| PI72-CLEAN, R9 (5 repeats) | D5-safe LightGBM | Post-op AUROC .7145 [.6828, .7455] | Total .7962; post-op AP .0568 |
| Task A, R11 (5 repeats) | A-D5-LGB | Total AUROC .7175 [.6929, .7417] | Post-op .7059; AP total/post .0093/.0100 |

**How to read these numbers.**
- G-safe-LGB has the highest point estimate in PI72-CLEAN. It is not a promoted improvement: promotion requires a paired ΔAUROC ≥ max(0.01, MDD) and a simultaneous lower bound > 0.
- No Task A program improved on A-D5-LGB.
- In PI72-CLEAN, pre-operative days are all negative. As a result, total AUROC ≈ f_pre × (cross-stage AUC ≈ 1) + (1 − f_pre) × post-op AUROC, with f_pre ≈ 0.28.
- In Task A, about 10% of positive days are pre-operative. The total AUROC therefore has to be decomposed into four cells; see `docs/EVALUATION.md`.
- Daily prevalence is about 0.35%, which is why AP is low. AUROC alone does not establish clinical usefulness.

## Quick start

Run from the repository root with an interpreter set up from `env/`:

```sh
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export PF_DATA_ROOT=/authorized/pc4          # folder with "NEC Cleaned Data.csv" and "Raw CSV Files/"
export PF_CACHE_ROOT=/private/pf-nec-cache   # private, never inside the repo
export PF_RUN_DIR=/private/pf-nec-cache/runs
python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"   # synthetic tests only
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
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5
python -m pf_nec.cli evaluate --model T8-D5SAFE-LGB --repeats 1 2 3 4 5
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --harness
```

**How the commands behave**
- Run the commands sequentially. Each process uses 2 numeric threads.
- If resources run out, stop. Do not reduce repeats or change configurations.
- Completed training resumes with the same command.

**Expected times on our Mac (M4, 16 GB, Python 3.9)**
- Feature build: about 3 min.
- One Task A context: about 2.5 min, so about 1 h for all 25.
- G-safe-LGB: about 15 s per repeat.

On macOS, import `torch` before LightGBM to load libomp; the entry points already do this. Details are in `docs/ENVIRONMENT.md`.

## Reproducibility check performed before release

**Synthetic tests:** 112 passed inside this repository.

**Real-data check:** run read-only against the authorised data, with these results.
- The fresh feature build matched the project's existing 13 tables value-for-value.
- A-D5-LGB r0/f0 (61,386 predictions): maximum absolute difference 6.9×10⁻¹⁸. That is floating-point rounding between Python 3.12 and 3.9, with byte-identical model text.
- G-safe-LGB r1 (6,787 predictions): identical.

**Not covered:**
- The full repeats were not re-run.
- Linux was not re-run.
- Windows was not run at all. The resource guard uses the Unix `resource` module.

## Deliberately excluded (see `docs/EXCLUDED.md`)

**PI-derived material.** None of the following is shared, and the final LightGBM paths do not depend on any of it:
- the PI's unpublished code and trained model;
- the code that loads or checks that model;
- the XGBoost learner whose hyperparameters come from that model;
- the PI-29 feature family and every menu that depends on it;
- the lineage audit that cites PI source lines.

**Code tried but not promoted.** Sequence (TCN/GRU), M1 and GPU models were not promoted. They are listed in the docs, but their code is not shipped.

**Internal and private material.** Data, data dictionary, caches, weights, predictions, row-level logs, host or credential configuration and internal agent tooling are excluded.

## Rules for people and coding agents (summary of `AGENTS.md`)

**Data handling**
- Data are read-only.
- Patient-level tables, caches, models and predictions never go into Git. They are written only under `PF_CACHE_ROOT` or `PF_RUN_DIR`.
- All external paths resolve through `src/pf_nec/config.py`.

**Frozen rules**
- Do not edit the frozen spec, feature rules, seeds, risk sets or evaluation functions to match or improve a number.
- To propose a change, create a new version and state its impact.

**Leakage and metrics**
- A prediction day uses only information known by the end of that day.
- Operative variables are masked on pre-operative rows.
- G encodings are re-cross-fitted inside each training pool.
- For Task A, pool out-of-fold predictions within a repeat first, then weight repeats equally.
- Report both post-op and total AUROC for PI72-CLEAN.

**Interpretation**
- SHAP values and altered-input scores are prediction contributions, not treatment effects.
- Re-scoring the same development data is not independent validation.
- Report measured results, inferences and unfinished work separately.

## Layout

```
src/pf_nec/     final models, data build, frozen evaluator (harness_v3 for Task A)
src/explore/    non-promoted methods (T3, I7, Aim 1b)
tests/          synthetic-data tests only
docs/           tasks, evaluation contract, environment, results, exclusions, data layout (Chinese)
env/            requirements snapshots (macOS venv; Linux server direct dependencies)
MANIFEST.json   every file with sha256 and source
```
