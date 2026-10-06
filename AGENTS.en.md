# Handoff rules for people and coding agents

中文：[AGENTS.md](AGENTS.md). Read README, TASKS and STATUS_AND_NEXT first; README defines bilingual paths.

- Inputs are read-only and require existing DUA / IRB authorisation. Patient-level tables, caches, models and predictions must never enter Git; write only under PF_CACHE_ROOT or PF_RUN_DIR.
- Resolve external paths only through the environment variables in `src/pf_nec/config.py`. Set PF_DATA_ROOT explicitly for real runs; a default path does not confer authorisation.
- Do not change frozen spec, feature rules, seeds, risk sets or evaluators to improve agreement or metrics. Version new research separately; do not reuse the consumed locked set as independent validation.
- Run synthetic tests first. Pool 5 OOF folds within each Task A repeat, then weight repeats equally; report both total and post-op PI metrics. Never compare AUROC across contracts.
- Each prediction day uses only information known by its end; gate historical days before window construction. Cross-fit G encodings within the appropriate training pool only.
- Use 2 numeric threads per process and an 8 GiB RSS cap; run large jobs sequentially, stop and report on limits, and never silently reduce menus or repeats.
- On Mac import torch before LightGBM. Record environment differences honestly; see DATA_LAYOUT for migration / resumption constraints.
- `src/pf_nec/` contains final models and frozen evaluation; `src/explore/` contains exploratory analyses and scaffolding. Do not classify interpretation tools as non-promoted performance programs.
- Label evidence as [fact] / [inference] / [suggestion] or 〔事实〕／〔推断〕／〔建议〕. Importance and altered-input scores are not treatment effects; do not claim independent / external validation that has not occurred.
- Maintain Chinese and English counterparts with identical numbers when changing docs. MANIFEST source_path identifies provenance, not an available in-package dependency.

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pf_nec.verify
```
