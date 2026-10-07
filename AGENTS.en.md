# Shared conventions for people and artificial intelligence (AI) coding agents

[中文](AGENTS.md). Start with [README](README.en.md), [task definitions](docs/en/TASKS.md) and [current status](docs/en/STATUS_AND_NEXT.md).

The repository maintainer maintains this package and records the shared evaluation contract so results remain comparable and changes can be reviewed.

- We all share obligations under the data use agreement (DUA), institutional review board (IRB) requirements and the principal investigator (PI)'s data agreement. Raw inputs are read-only; patient-level tables, caches, weights and predictions must never enter Git and stay in the authorized private cache root (PF_CACHE_ROOT) or run directory (PF_RUN_DIR). See [EXCLUDED](docs/en/EXCLUDED.md) for unpublished intellectual-property boundaries.
- Patient-level use outside registered hosts still requires prior confirmation from the repository maintainer and the PI under those agreements; task cards and default paths are not transfer authorization. This obligation applies to everyone.
- External paths use the environment variables in `src/pf_nec/config.py`. Set the authorized data root (PF_DATA_ROOT) explicitly for real runs to preserve input identity; default paths do not confer authorization.
- Reproduction keeps fixed model specifications, feature rules, seeds, risk sets and evaluators so results remain comparable. We can discuss new ideas and give them separate versions; previously used holdouts do not regain independence through new seeds.
- Synthetic tests catch interface and timing problems early. For the daily task (Task A), pool 5 out-of-fold (OOF) predictions within each repeat, then weight repeats equally; the local cleaned-table contract (PI72-CLEAN) reports both total and postoperative metrics. Compare area under the receiver operating characteristic curve (AUROC) only within the same contract, risk set and validation scheme.
- Each prediction day uses information known by its end; gate historical days before constructing windows. Enhanced encodings (G) are cross-fitted within their training pools to keep outer-test information out of training.
- Use 2 numerical threads per process and a resident set size (RSS) cap of 8 gibibytes (GiB), running large jobs sequentially. Preserve the record and review causes together if a cap is reached; silently shrinking menus or repeats breaks the planned comparison.
- On Mac, import torch before the Light Gradient Boosting Machine (LightGBM) to load the numerical runtime correctly. See [DATA_LAYOUT](docs/en/DATA_LAYOUT.md) for environment differences and resumption requirements.
- `src/pf_nec/` provides final models and fixed evaluation; `src/explore/` provides exploratory analyses and scaffolding. Importance and altered-input scores do not establish treatment effects; reports distinguish measured facts, interpretations and suggestions in plain words.
- Keep Chinese and English documents synchronized with identical numbers and explain abbreviations on first use. The file inventory (MANIFEST) `source_path` field records the source location in the original project, not an in-package runtime path; review versions, hashes, synthetic tests and documentation together after changes.

These are the entry points we use to check the code:

Names used in the example: interpreter module-search-path environment variable (`PYTHONPATH`); interpreter environment variable disabling bytecode caches (`PYTHONDONTWRITEBYTECODE`).

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp="$PF_RUN_DIR/test-temp"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pf_nec.verify
```
