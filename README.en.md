# Project Passion Fruit: NEC research code handoff

中文：[README.md](README.md). Bilingual convention: `docs/<NAME>.md` is Chinese and `docs/en/<NAME>.md` its English counterpart; root README / AGENTS and explore/README use `.en.md` for English. Both languages carry the same facts and numbers, with no priority language; builder tests check numeric-token parity.

[fact] This package contains final-model code, frozen rules, synthetic tests, aggregate results and exploratory scaffolding. It contains no data, weights or predictions. Users need existing DUA / IRB authorisation for the same PC4 data. These are retrospective internal development-data evaluations, with no independent or external validation yet. See [GLOSSARY](docs/en/GLOSSARY.md) for abbreviations.

|Contract|Final handoff models|Primary metric|Repeats / outer evaluation|
|---|---|---|---|
|PI72-CLEAN local clean-table contract|GSAFE-LGB (629 columns); T8-D5SAFE-LGB (604 columns)|Post-op POD≥0 AUROC|r1–5, fixed held-out-stay split per repeat|
|Task A formal v2.6 / harness v3|A-D5-LGB (604 columns, this round's reference)|Total AUROC; post-op is key secondary|r0–4, 5-fold stay OOF per repeat|

Contracts differ in rows, labels and validation scheme; never compare AUROC across them. PI72-CLEAN is a local rebuild, not the final scoring design behind the manuscript's 0.79, which remains unresolved. A-D5-LGB does not replace the historical champion; Task B was not retrained this round.

## Where we left off

[inference] Intended use remains undefined: who sees risk, when, and what changes afterward? [suggestion] Prioritize A1 use-specific evaluation, A2 surgical-score versus daily-model comparison, and A3 centre-held-out evaluation. [Approved decision] DEC-019 pauses main-line architecture search. Any one condition can support restarting: a distinguishable new mechanism with a frozen metric, budget and stopping rule; the use discussion with the PI changes the primary metric; or a new data or information source provides credible evidence. A new architecture name or new seeds alone is insufficient. T-2/T-3 are allowed under their own frozen cards; see [TEAM_TASKS](docs/en/TEAM_TASKS.md). See [STATUS_AND_NEXT](docs/en/STATUS_AND_NEXT.md).

[suggestion] Aim 1b prioritizes discussion of sternal closure, with peripheral arterial-line removal as a conditional backup; freeze after the PI answers 7 questions. [fact] B1 literature scouting is complete (not a systematic review): without a linkage key, help is limited to score recomputation or within-signal-cohort distillation; individual fusion requires authorized linkage at an institution holding both sources. No Aim 2 modelling was done; next steps are synthetic interface specification / prototyping and governance preparation only. See [AIM2_LINKAGE_NOTE](docs/en/AIM2_LINKAGE_NOTE.md). Retain B2, the mixed-phase evaluation lead.

[fact] Read three sets separately: **final models** above; **candidates that did not pass the pre-specified promotion gate** (T8, T4, M1 and Task A programs) with effect sizes / intervals in [RESULTS](docs/en/RESULTS.md); **exploratory analyses and scaffolding** (T3, I7, Aim 1b) in [explore](explore/README.en.md). Importance and altered-input scores are not treatment effects.

[fact] Positive-day prevalence is contract- and phase-specific: about 0.35% in Task A; about 1.0% overall and 1.4–1.5% post-op in PI72-CLEAN held-out data. All-negative pre-op labels in PI72-CLEAN (the local clean table) establish a decomposition identity; cross-stage AUC≈1 is measured for that contract, not guaranteed by label composition or attributed to the manuscript design. Task A's A-D5-LGB corresponding field is about 0.92, but compares all positives (including pre-op positives) with pre-op negatives, not the post-op-positive-only cell; see the four-cell definitions in [EVALUATION](docs/en/EVALUATION.md), without comparing performance across contracts. Sources: `r9/report.md`, `r11/report.md` (internal reports, not shipped; key numbers in RESULTS).

## From installation to reproduction

[suggestion] Follow [ENVIRONMENT](docs/en/ENVIRONMENT.md) to install a fresh Mac / Linux venv, then run the commands below from the repository root with that environment's `python`. See [DATA_LAYOUT](docs/en/DATA_LAYOUT.md) for all 20 raw tables, clean-column order, cohort flow and output tree. Write only to authorised private roots.

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

Run sequentially, with 2 numeric threads per process and RSS≤8 GiB; stop on limits without changing configuration. Build the cache once; resumption requires unchanged inputs, versions, absolute paths and paired CACHE/RUN, not migration of RUN alone. See [EVALUATION](docs/en/EVALUATION.md) for single-arm descriptive CIs, paired references with explicit family, 400-draw harness versus 2000-draw stratified bootstrap, JSON shape and tolerances.

## Check scope and reading order

[fact] Historical H1 rebuilt 13 matching tables and checked real-data equivalence for A-D5-LGB r0/f0 and GSAFE-LGB r1. The tiny A prediction difference is consistent with rounding; see ENVIRONMENT. These checks do not establish full-repeat, Linux or Windows reruns or independent validation. Synthetic tests cover final-model paths, causal time invariance, evaluation and card APIs.

Read [TASKS](docs/en/TASKS.md) and [RESULTS](docs/en/RESULTS.md), then STATUS_AND_NEXT; teammates start with [TEAM_TASKS](docs/en/TEAM_TASKS.md). Sharing exclusions are in [EXCLUDED](docs/en/EXCLUDED.md); bibliography and internal-source explanations are in [REFERENCES](docs/en/REFERENCES.md). MANIFEST source_path is provenance, not a runtime dependency. People and coding agents follow [AGENTS](AGENTS.md) / [AGENTS.en](AGENTS.en.md); CLAUDE points to the same rules.
