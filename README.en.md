# Project Passion Fruit: intestinal risk after neonatal heart surgery

[中文](README.md)

We use Pediatric Cardiac Critical Care Consortium (PC4) registry data to study prediction of necrotizing enterocolitis (NEC) after neonatal heart surgery. The first research aim (Aim 1) covers risk prediction and target-trial emulation to investigate treatment-improvement questions; the second research aim (Aim 2) will study continuous physiological signals, which are not yet available. This repository provides code, synthetic tests and aggregate evidence for research partners joining the work.

## Where things stand

As of 2026-10-07, within the same daily task and risk set, adding current-day condition improved discrimination; accumulated history showed no detectable additional gain. Alert detection and clinical benefit require their own evaluation. Point estimates for this round's information-ladder models are close to the [fixed daily tree-model reference](docs/en/TASKS.md) (A-D5-LGB); they did not meet promotion criteria, and equivalence has not been established.

Current scores are better at ranking which hospital stays have higher risk than tracking deterioration as onset approaches within a stay. Retrospective equal-capacity alert comparisons look more favourable, while the implementable training-fixed threshold rule shows no clear detection gain; these need separate interpretation.

All evidence is retrospective, internal cross-validation in development data, without external validation. Conclusions are conditional on availability by prediction time of 64 unresolved columns; independence is per stay, not per patient, because reliable cross-stay patient identifiers are absent; the population is the outcome-conditioned development cohort (formal), not a deployment population.

The common evaluation addendum is installed and frozen, so the fixed-landmark task is no longer blocked by addendum review. Alert levels are predeclared research choices, not clinical thresholds endorsed by the principal investigator (PI). The first-day prognostic-stratification step supporting treatment-improvement research has been run and checked locally: richer inputs showed no clear discrimination gain over fixed information plus clock; treatment effects were not estimated. The shorter endpoint has few events and unreliable estimates. [Status and next steps](docs/en/STATUS_AND_NEXT.md) and [results and limitations](docs/en/RESULTS.md) give the details.

## Getting started

- If you want to reproduce results, start with [the steps we use](docs/en/REPRODUCE.md): synthetic tests first, then data and configuration checks in an authorized environment. This package runs training/evaluation for the three handoff tree models and synthetic interface examples. The latest ladder, alert and first-day-stratification findings are summaries; complete reproduction drivers are not shipped. Authorized partners can obtain execution settings, version hashes and run arrangements from the repository maintainer (open an issue in this repo or get in touch directly).
- If you want to pick up a task, explore the [open task cards](docs/en/TEAM_TASKS.md) and [runnable synthetic examples](docs/en/TEAM_USAGE.md). Each card describes what is prepared, the next step and contact and the research boundaries.
- If you want the details, see [task definitions](docs/en/TASKS.md), [evaluation](docs/en/EVALUATION.md), [results](docs/en/RESULTS.md) and the [glossary](docs/en/GLOSSARY.md); signal preparation is described in the [linkage note](docs/en/AIM2_LINKAGE_NOTE.md).

The repository maintainer maintains this package and records the shared evaluation contract so results remain comparable and changes can be reviewed.

The data use agreement (DUA), institutional review board (IRB) requirements and the PI's data agreement are shared obligations. See the [exclusions](docs/en/EXCLUDED.md) for data and unpublished intellectual-property boundaries; patient-level data, caches, predictions, weights and unpublished PI code must not enter Git; patient-level artifacts stay in authorized private environments.

## For artificial intelligence (AI) coding agents and machine-readable entry points

Artificial intelligence (AI) coding agents can use these materials to understand interfaces and shared conventions:

- [AGENTS.md](AGENTS.md) / [English](AGENTS.en.md): collaboration, data governance and research conventions; [CLAUDE.md](CLAUDE.md): an entry pointing to the same rules.
- [MANIFEST.json](MANIFEST.json): file inventory, integrity hashes and source identities; source paths are provenance, not in-package runtime dependencies.
- [Model specification](src/pf_nec/spec_v1.json) and [exploration interface specification](src/explore/team/spec.json): fixed model settings and independent exploration constraints in JavaScript Object Notation (JSON).
- [Machine terminology](docs/terminology.json): bilingual definitions and code index. The exploration specification retains unreleased gate values from handoff; these do not establish current project status. The original implementation does not specify whether they are per-run defaults; see [interface scope](docs/en/TEAM_USAGE.md). Freezing a research addendum does not establish clinical endorsement; real runs still require agreed review and data authorization.
- [SCORER_SCHEMA](docs/en/SCORER_SCHEMA.md): fields and open items for the synthetic scoring interface, not yet a complete runnable scorer.

Documents are maintained in pairs at `docs/<NAME>.md` and `docs/en/<NAME>.md`; root and exploration entry points use `.en.md` for English. Both languages carry the same facts and numbers.
