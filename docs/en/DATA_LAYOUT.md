# Inputs, cohort flow, outputs and resumption

[fact] `PF_DATA_ROOT` points to the same authorised read-only input version. It must contain the following 20 raw tables and 1 clean table; do not put extra CSVs in the raw directory. Preserve filenames and field-name case, encoding, date text, missing markers and numeric precision; do not recast dates or stay keys to integers. The raw reader uses UTF-8 with cp1252 fallback.

```text
PF_DATA_ROOT/
  NEC Cleaned Data.csv
  Raw CSV Files/
    AllOperations.csv
    Arrest.csv
    ArterialLine.csv
    CABSI.csv
    Catheterizations.csv
    Complications.csv
    DSSI.csv
    ECMO.csv
    IndexSurgHosp.csv
    IntracardLine.csv
    MechVent.csv
    NEC.csv
    PreopRiskFactor.csv
    Procedures.csv
    RiskSurgVIS.csv
    Sternum.csv
    Stroke.csv
    Surgdiag.csv
    Therapies.csv
    UTI.csv
```

## Column order and input identity

[fact] The clean table **must retain its original column order**. `legacy/rebuild_v26.py` selects `cols[25:58]` and `cols[109:124]`; do not alphabetise columns or drop apparently unused ones. Key linking fields include raw `hospitalizationidNEW`, `cardsurgdtSHIFT`, and clean `hospitalizationidnew`, `cardsurgdtshift_index`, `day`, `outcome_3d`, `necbelldtshift`. The full header and code mappings are rebuilt from local clean inputs; the clinical dictionary and data-derived crosswalks are not shipped.

Run `python -m pf_nec.cli build-data` in a fresh cache. It records each input SHA-256, checks rebuilt column names / order and populations, and rechecks input hashes on completion. Retain private `core/manifest.json` to compare input versions; identical filenames do not establish identical data. Source locations: `src/pf_nec/data.py:build_frames`, `src/pf_nec/legacy/rebuild_v26.py:Rebuilder`.

## Cohort flow (counts are stays, not independent infants)

[fact] There are two branches; PI72-CLEAN must not be obtained by an inner join to formal.

|Step / branch|Accepted count|Rule and source|
|---|---|---|
|Raw master|11,938 stays, 61 centres|Complete source cohort; `a2/report.md`|
|Original clean table|320,143 rows, 11,931 stays|Unmodified input; `01_review/数据全貌与分组说明.md`; H1 equivalence receipt|
|Raw → backing cache|462,721 rows; v3 split table has 11,907 stays|Rebuild, then filter before first NEC / discharge; retain formal and auxiliary sets; H1 equivalence receipt, harness v3|
|Backing → Task A formal|301,468 days, 11,674 stays, 359 cases, 1,066 positive days|Clean-linked a/b/none, still at risk, original POD≤31; `r11/report.md`|
|Complete raw calendar + clean → PI72-CLEAN|318,992 labelled days, 11,931 stays, 356 cases / positive days|Reconcile clean dates and row keys, drop missing labels; do not intersect formal; `ROUND_REDO_summary_zh.md`|
|PI72-CLEAN sampling per repeat|Train 1,116 stays; test 279 stays|Train cases / controls 285/831; test 71/208; `ROUND_REDO_summary_zh.md`, `r9/report.md`|

These are internal reports / receipts, not shipped; key numbers are summarised here. Backing includes auxiliary rows, so 11,907 is not formal's 11,674; the branches are not one funnel whose counts can be successively subtracted. See [GLOSSARY](GLOSSARY.md) for groups.

[fact] Separate exclusion rules from counts excluded for each reason. The clean table has 7 fewer stays than raw; the original inventory identifies 4 with no NEC-table record, 2 in its late group (POD≥31), and 1 with NEC dated before admission. Those historical groups are not the formal groups in this package; this composition does not explain why clean omitted them, and the accepted inventory does not document the actual removal reasons: **reason not documented in shipped sources**. Do not attribute all omissions to NEC or missing values. Shipped `legacy/build_v26.py:assemble_rows` removes rows on / after first NEC and on / after discharge, then retains only clean-linked formal or auxiliary sets; non-clean-linked rows, err and rows meeting none of the set rules are dropped. `harness_v3.make_splits` creates splits only for retained stays. Thus 11,907 comes from backing, not a simple deletion of a known group directly from clean’s 11,931; the per-reason count reconciliation between these stay totals is **not documented in shipped sources**. Backing-to-formal further excludes c/d auxiliary populations and rows outside the formal window; PI72-CLEAN separately uses retained clean days with nonmissing labels. Sources: `01_review/数据全貌与分组说明.md` (internal report, not shipped; key counts summarised here) and the shipped code above; no new patient-level derivation was performed to explain the differences.

## Private output tree

[fact] These are path patterns: N is a repeat and F a fold. Neither root may be inside the input data directory; `config.py` resolves actual locations. G-safe weights and predictions live in CACHE; retaining RUN alone is insufficient.

```text
PF_CACHE_ROOT/
  reconstruction_mappings.json
  v26_rows.parquet
  splits_v3.parquet
  core/
    manifest.json
    rows.parquet, stays.parquet, source_metadata.parquet
    D5.parquet, S.parquet, raw_codes.parquet
    split_01.parquet ... split_05.parquet
  gsafe/rN/gsafe/
    predictions.parquet
    outer/model.txt, outer/result.json, outer/encoding_full.json
    innerN/ ...
  tmp/, mpl/
PF_RUN_DIR/
  versions.json
  GSAFE-LGB/rN-f-1.json
  T8-D5SAFE-LGB/rN-f-1.json
  A-D5-LGB/rN-fF.json
  <model>/<context>/fit_ledger.json, jobs/ ...
  <model>/evaluation.json
  reference_pair.json
```

[suggestion] Reissue the same training command only with unchanged inputs / software and the paired CACHE and RUN at their original absolute paths. Retain models, encodings, predictions, job receipts and fit_ledger; checkpoints check training bindings and output existence, not arbitrary cache repair. Do not pair old G-safe CACHE with empty RUN: checkpoints may return while the new ledger cannot represent the original fits. Once `core/manifest.json` exists, rebuilding is refused; interrupted `.partial` files are not successful caches and batch-level build resumption is not guaranteed, so use a new CACHE.

[suggestion] For machine / path migration or package upgrades, start a fresh paired CACHE/RUN, rerun and compare; old absolute paths are not automatically relocated. Do not discard CACHE while retaining RUN or bypass version-freeze files. Stop for ambiguous input mappings or population changes; do not “repair” them by dropping rows or changing keys. Check the environment and synthetic tests before real execution.
