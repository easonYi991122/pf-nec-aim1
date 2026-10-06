# Glossary, aliases and timeline

[fact] This page explains code contracts, not the clinical data dictionary. Exact columns are in `src/pf_nec/spec_v1.json`; clinical codes, units and recording times still require the authorised dictionary and clinical review.

|Term|Meaning|
|---|---|
|NEC / PC4|Necrotizing enterocolitis; Pediatric Cardiac Critical Care Consortium|
|PI / DUA / IRB|Principal investigator; data use agreement; institutional review board|
|AUROC / AP / Brier|Area under the ROC curve (positive-negative ranking); average precision (prevalence-dependent); mean squared probability error (sampled days here)|
|OOF / repeat / fold|Out-of-fold predictions / repeat / fold; repeats add no new patients; folds partition stays within a repeat|
|stay / case / y|Stay key / case-stay indicator / daily label; a case stay can also have days with y=0|
|row_key / source_row_id / __harness_row_index|Daily-row identity; the latter fields identify source / harness rows in different contracts, not interchangeable stay keys|
|C_h|PI72-CLEAN source-case stay indicator, distinct from the daily label|
|D5 / D5SAFE / D5-safe|Historical feature-bank label; safe version has 604 columns; D5 does not mean a 5-day window|
|G / GSAFE-LGB / G-safe-LGB|This round's 629-column encoding-enhanced reference; old G-LGB did not repeat this round's pre-op masking and is different|
|v2.6 / harness v3|Row / feature data version versus evaluation / split version: separate version axes|
|r0–4 / r1–5|Task A inherits zero-based harness repeat IDs; PI72-CLEAN inherits one-based source-split IDs; these are not the same random splits|
|screen / MDD|3-repeat diagnostic screen / minimum detectable difference, frozen here at 0.01 AUROC; confirmation requires 5 complete repeats|
|paired Δ / simultaneous CI|Metric difference on matched rows / stays; intervals covering a predeclared complete family jointly|
|SHAP|SHapley Additive exPlanations, model attribution on the raw margin / log-odds scale; not an effect or action direction|
|TCN / GRU / GRU-D / M1|Temporal Convolutional Network; Gated Recurrent Unit; GRU with decay / missingness handling; historical GPU candidate group (TabM, GRU-D, CausalMS-CNN)|
|B / C / S slices|B=8 background / clock variables; C=cumulative variables; S=13 current-state variables. B is not Task B, and the G-safe support-state S plugin is a different column set|
|w1 / w7 / R1 / R3 / R4|1 / 7-day windows; flattened lag values / per-variable summaries plus structural counts / sequence values, missingness and row-availability bits. R1 is not repeat r1|
|GAIN / SHADOW / ACTION16 / PI10|Training-only gain ranking / historical shadow ranking / fixed action slice / fixed 10-variable interpretation group. Retained column definitions do not expose every training menu|
|PI-29 / source bank / CORE|Excluded model-derived family / its feature domain / historical name for the local clean-row and sampling interface; these are not external dependencies to open|
|STAT / VIS / IS / ICU|Congenital cardiac surgery risk score / vasoactive-inotropic score / inotropic score / intensive care unit|
|D/I/N/S risk|Decision candidate / indirect marker / non-intervenable category / severity-proxy risk; labels do not certify intervenability|
|DAG / CCW / ESS / SMD / MAR|Directed acyclic graph / clone–censor–weight / effective sample size / standardised mean difference / medication administration record|
|OR / PACU / Bell / positivity|Operating room / post-anaesthesia care unit / NEC severity criteria requiring clinical confirmation / support for both strategies given population and confounders|
|keep / maybe / time zero|Discussion candidate rating / tentative rating / aligned eligibility, strategy assignment and follow-up start; not a synonym for prediction day|
|z_ / supp_ / dxg_ / lnb_ / evr_ / raw_ / site_eb|Time and other derived values / supplemental measurements / diagnosis groups / line semantic block / event block / raw-code encoding / centre empirical-Bayes shrinkage; code defines each field|

## Timeline and groups

[fact] `DOA0=admission day → POD0=index surgery day → prediction at end of POD k → future event`; lead=event date−prediction date. Inputs use information known by prediction time; lead attribution retrospectively conditions on a future event.

|Group|First recorded NEC relative to index surgery|Formal|
|---|---|---|
|a|POD4–31|Include when clean-linked and still at risk|
|b|POD0–3|Include when clean-linked and still at risk; can create pre-op y3 positives|
|c|Before surgery, but not before admission|Exclude; pre-op auxiliary rows c_aux|
|d|POD>31|Exclude; within-window auxiliary rows d_window|
|none|No known first-event date|Include when clean-linked and row rules hold; not certified NEC-free|
|err|Date before admission|Exclude|

late_aux means late auxiliary days; locked means the already-used historical temporal holdout; late_era marks a later, potentially truncated period. Task B / B30 y30 / yB30 target first NEC within 30 days after the landmark; Bd is the daily version. Old yB targets events after the landmark through POD31 and is not interchangeable. No Task B training path is shipped this round.

## Supplement: historical stops and labels

|Term|Specific meaning here|
|---|---|
|C2 / STOP|Historical input-timing sensitivity work; here, the 3-repeat screen dropping 65 timing-uncertain columns. STOP means the original continuation gate was not met and screening stopped, without automatic continuation or reversal of historical results; these columns are not established leakage|
|DEC-008–010|Historical decision chain: DEC-008 defined the bounded search and fixed combination, DEC-009 audited component G, and DEC-010 required closing the AUC search after that batch; not a reopened menu for this round|
|v2.5 locked evaluation|A one-time evaluation of pre-separated later-period stays, already consumed; former held-out stays subsequently entered the harness v3 development pool, leaving no untouched holdout now|
|Original abc target|The original plan starts rolling prediction of first NEC at admission-day end and includes positive groups a/b/c, including pre-op-onset c; current clean-linked a/b/none formal excludes c/d and does not complete that target|
|y3|Task A daily label: whether first recorded NEC occurs 1–3 days after prediction; pre-op days can be positive|
|outcome_3d|The daily label retained from the local clean table; the observed rule retains 1 positive day per eligible case at max(0, NEC day−3). Not an alias for y3, and the name does not certify exactly 72 hours of lead time|

Sources: `r11/report.md` §T5, `STATE.md`, historical DEC decisions and the original task plan (internal reports, not shipped; key meanings summarised here); see TASKS and shipped frozen code for label implementation.

## Internal ID map

These identify internal tasks / reports, not extra software dependencies. Internal reports are not shipped; key results are in RESULTS and status in STATUS_AND_NEXT.

|ID|Question and destination|
|---|---|
|R8 / R9|PI72-CLEAN T8 3-repeat screen / 5-repeat confirmation; the latter reuses the former repeats|
|R10 / R11|Final PI72-CLEAN T3 / Task A confirmation, T3 and T5 closeout|
|T2 / T3a / T3b / T3c|Rescoring old fixed predictions / overall importance / case-control trajectories / POD and lead attribution|
|T4 / T8 / T5 / T6|History / order selection; feature-count selection; A/B closeout; action clue cards|
|I7 / M1|Slice, window, order and learning-curve diagnostics / historical GPU candidate confirmation|
|Aim 1b A1 / A2|Stages 1–2 measurement screening / stages 3–4 DAG and overlap diagnostics; distinct from next-step A1–A3 labels|
|RX-D1 / RX-I1 / RX-I2b / RX-I11|Frozen evaluation contract / implementation interface / G-safe encoding hook / fast T3 aggregation refactor|
|RX-H1 / RX-H2 / RX-H3|Initial handoff export / blind-reader and semantic review / this revision|
|T007 / DEC / WP0|Internal manuscript-table extraction / historical decisions / local clean-row and sampling reconciliation; not shipped|
|R9 raw-code columns|Alias for G's 9 raw-code input columns, not the R9 report|

If an internal report is unavailable, use the self-contained summary and concrete source entry points here; MANIFEST source_path is provenance only.
