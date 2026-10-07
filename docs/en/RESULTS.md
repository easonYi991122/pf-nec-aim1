# Results, attempted candidates and evidence limits

All numbers here are [fact]: aggregate excerpts from accepted internal results; each section identifies its contract. All internal reports are **not shipped; key numbers are summarised here**. Intervals condition on fitted models and prior selection, excluding retraining uncertainty. Do not compare AUROC across contracts, risk sets or validation schemes.

## Final handoff models

|Contract / internal report|Model|Primary AUROC [descriptive 95% interval]|Supplementary metrics|
|---|---|---|---|
|PI72-CLEAN, `r9/report.md`|GSAFE-LGB / G-safe-LGB|Post-op 0.7329 [0.7020, 0.7622]|Total 0.8095; post-op AP 0.06391|
|PI72-CLEAN, `r9/report.md`|T8-D5SAFE-LGB / D5-safe|Post-op 0.7145 [0.6828, 0.7455]|Total 0.7962; post-op AP 0.05676|
|Task A formal, `r11/report.md`|A-D5-LGB|Total 0.7175 [0.6929, 0.7417]|Post-op 0.7059; total/post AP 0.00933/0.00995|

These are equal-weight means over 5 repeats; Task A first pools 5 OOF folds per repeat. G-safe has the highest single-program point estimate in this PI72-CLEAN round; A-D5-LGB is this round's reference, not the historical champion. Candidates did not pass the pre-specified promotion gate; that does not establish equivalence or absence of useful information. Descriptive intervals do not replace simultaneous intervals.

## Attempted, not promoted: PI72-CLEAN

These are **post-op** ΔAUROC over 5 repeats. The R9 table gives all 10 frozen T8/T4 contrasts with that family’s simultaneous 95% intervals. The M1 table gives all 6 contrasts, displaying both the 6-contrast simultaneous and 16-contrast joint 95% intervals; promotion for the combined menu uses the 16-contrast joint family, without mixing endpoints across families. Sources: `r9/report.md`, `m1run/report.md` (internal reports, not shipped; key numbers summarised here).

### R9

|Program|Reference|Δ|10-contrast simultaneous 95% interval|
|---|---|---|---|
|T8-SELECT-LGB|T8-D5SAFE-LGB|-0.01605|[-0.0430, 0.0109]|
|T8-SELECT-LGB|GSAFE-LGB|-0.03439|[-0.0614, -0.0074]|
|T8-SELECT-XGB|T8-D5SAFE-XGB|+0.00038|[-0.0266, 0.0274]|
|T8-SELECT-XGB|GSAFE-LGB|-0.02124|[-0.0482, 0.0057]|
|T4-SELECT-TREE|T4-MATCHED-W1|-0.00593|[-0.0329, 0.0210]|
|T4-SELECT-TREE|GSAFE-LGB|-0.04013|[-0.0671, -0.0131]|
|T4-SELECT-TCN|T4-SELECT-TREE|+0.00543|[-0.0215, 0.0324]|
|T4-SELECT-TCN|GSAFE-LGB|-0.03470|[-0.0617, -0.0077]|
|T4-SELECT-GRU|T4-SELECT-TREE|+0.01050|[-0.0165, 0.0375]|
|T4-SELECT-GRU|GSAFE-LGB|-0.02963|[-0.0566, -0.0027]|

### M1

|Program|Reference|Δ|6-contrast simultaneous 95% interval|16-contrast joint 95% interval|
|---|---|---|---|---|
|M1-SELECT-TABM|T4-SELECT-TREE|0.0119|[-0.0161, 0.0399]|[-0.0171, 0.0409]|
|M1-SELECT-TABM|GSAFE-LGB|-0.0282|[-0.0562, -0.0002]|[-0.0572, 0.0008]|
|M1-SELECT-GRUD|T4-SELECT-TREE|0.0163|[-0.0117, 0.0443]|[-0.0127, 0.0453]|
|M1-SELECT-GRUD|GSAFE-LGB|-0.0238|[-0.0518, 0.0042]|[-0.0528, 0.0052]|
|M1-SELECT-CMSCN|T4-SELECT-TREE|0.0064|[-0.0216, 0.0344]|[-0.0226, 0.0354]|
|M1-SELECT-CMSCN|GSAFE-LGB|-0.0337|[-0.0617, -0.0057]|[-0.0627, -0.0047]|

Every simultaneous / joint lower bound above is ≤ 0; none passed the pre-specified promotion gate, including the positive M1 point differences against the local tree reference. The historical selection menus or learners are excluded here, so the original programs cannot be rerun or reassessed after reducing their menus.

GRU versus TREE reached a point difference of 0.01, but its simultaneous lower bound was negative. None of these programs passed the pre-specified promotion gate. Existing PI-29-dependent results remain valid, but cannot be rerun from this repository; no menu was silently reduced and reassessed. T8 showed no stable monotonic benefit from more columns within a fixed source menu; comparing PI-29 with D5 also changes the source, so it is not a pure feature-count experiment.

## Attempted, not promoted: Task A formal

`r11/report.md`: 5 repeats, total ΔAUROC; A-D5-LGB is the reference throughout, with this contract's 4-contrast simultaneous 95% intervals.

|Program|Δ|Simultaneous interval|
|---|---|---|
|A-T8-GAIN|-0.00409|[-0.0160, 0.0078]|
|A-T4-TCN|-0.01011|[-0.0220, 0.0018]|
|A-T4-GRU|-0.01686|[-0.0287, -0.0050]|
|A-T4-TREE|-0.01840|[-0.0303, -0.0065]|

None passed the pre-specified promotion gate. These selection programs are not shipped. These are completed historical comparisons, not unfinished runs.

## Exploratory analyses and scaffolding

[fact] I7's single-day model with 8 background / clock variables: post-op AUROC 0.693 [0.650, 0.731]; adding 13 current-state variables gives 0.717 [0.677, 0.753]. G-safe scored 0.737 on the same 3 repeats. **This is a 3-repeat screen, with no paired test against G-safe.** The paired post-op B→B+S Δ is +0.0241 [0.0055, 0.0449]; that is a separate descriptive comparison, not a treatment effect. Sources: `DEEPER_QUESTION_zh.md`, `i7/report.md`, `r8/report.md`.

[fact] The 8 variables in B are listed below; “background” names a slice, not universal pre-op availability. `statscore` and `xclamptime` are masked pre-op and still require the frozen surgery-completion availability gate post-op. Sources: frozen spec `diagnostics.feature_slices.B` and the clock / masking rules in `legacy/build_v26.py`.

|Field|Meaning|
|---|---|
|gestagewks|Gestational age in weeks|
|birthwt|Birth weight|
|z_dol|Age in days on the prediction date|
|z_doa|Days since admission|
|z_post_surg|Whether the current day is post-op|
|z_pod|Post-op day; missing pre-op|
|statscore|STAT surgical risk score; masked pre-op|
|xclamptime|Cross-clamp time; masked pre-op|

[fact] I7 learning curves: descriptive mean post-op AUROC slopes per doubling of training cases are LGB 0.0305 [0.0115, 0.0489] and TCN 0.0295 [0.0165, 0.0419]. History-window and order gains did not reach the detectable threshold; this does not prove history is uninformative or guarantee benefit from more data. At POD0–2, TCN versus an orderless MLP gave +0.0416 [0.0092, 0.0752] and versus the w7-R1 tree gave −0.0026; the former is not the largest of 16 sequence cells, with +0.0462 versus the w7-R1 tree at POD8–14. None has across-POD multiplicity correction. PI-29 curve / order results remain valid, but only sampling / statistical methods can be reused here, not those model fits. Sources: `i7/report.md`, `ROUND_REDO_summary_zh.md`.

[fact] T3: the two reported Task A models emphasise STAT, cross-clamp time and ventilation; arterial lines are absent from their post-op top five concepts. Evidence of arterial-line reliance comes from PI72-CLEAN (R10); drug and VIS permutation losses of approximately 0 also refer only to PI72-CLEAN, because Task A’s design did not perform permutation analysis. Code encodings matter for G-safe; STAT is not first in every model. In PI72-CLEAN, the historical PI29 surgical-location feature `AllOperations_procloc_9` (ProcLoc 9 = Cardiac OR) had a 16.3% share at POD0–2. Importance measures model reliance, not treatment effects; altered-input scores are not effects either. Comparison with the manuscript's importance ranking remains incomplete. Historical PI-29 attributions remain valid but cannot be rerun here. Sources: `r10/report.md`, `r11/report.md`, `ROUND_REDO_summary_zh.md`.

## Historical evidence and corrections (aggregates only)

Each entry retains its original contract and validation identity; no patient values are shown. Describe site-held-out and stay-split results separately without subtraction; POD cells are not new independent confirmation.

|Historical configuration / cell|Original value or correction|Source section name|
|---|---|---|
|E9-A-D5-site; harness v3, 1×5 leave-site-out|AUROC 0.6950 [0.6732, 0.7174]; distinct from stay splits|Run registry: E9-A-D5-site, site sensitivity (DEC-007 step 6)|
|TCN − ORDERLESS_MLP; POD0–2|+0.0416; not the largest sequence cell|I7 “POD-matched gains”|
|TCN − w7-R1 tree; POD0–2|−0.0026|I7 “POD-matched gains”|
|TCN − w7-R1 tree; POD8–14|+0.0462; 16 sequence cells inspected, without across-POD correction|I7 “POD-matched gains”|
|M1 menu; r1 GRU-D|Included GAIN50; selected GAIN50-w7|M1 addendum profile menu; r1 choices / GRUD-WINDOW|

I7 rows use the historical source-feature recipe, not results achieved by the new GAIN50 program; trimming the M1 menu also does not confirm the original program.

## Phase composition and historical identities

[fact] R9 held out 279 stays per repeat (71 cases, 208 controls); each repeat has 71 positive days, all post-op.

|Repeat|Total days|Pre-op days|Post-op days|
|---|---|---|---|
|1|6787|2020|4767|
|2|6958|1962|4996|
|3|6900|2025|4875|
|4|6861|1831|5030|
|5|6761|1876|4885|

PI72-CLEAN held-out positive-day prevalence is about 1.0% overall and 1.4–1.5% post-op; pre-op prevalence is 0. Task A has 1,066 positive days among 301,468 days (about 0.35%): 77,351 pre-op days / 110 positive days and 224,117 post-op days / 956 positive days. These are day proportions, not stay incidence. Sources: `r9/report.md`, `r11/report.md`.

[fact] R8 post-op AUROC was 0.7366/0.7282 for G-safe / D5-safe; R9 reused r1–3 and added r4–5, not independent validation. T2's old G-LGB did not repeat the pre-op masking and is not G-safe. Sources: `r9/report.md`, `t2/report.md`.

[fact] Historical identities (their own contracts; no subtraction from this round): Task A ensemble with CNN approximately 0.718, with centre information approximately 0.730; Task B POD3 combination F 0.778, component G 0.784. Search closed under DEC-008–010. A-D5-LGB does not replace the historical champions; Task B was not retrained this round. The v2.5 locked evaluation has already been used; there is no untouched holdout now. C2's 3-repeat screen dropping 65 columns remains STOP: their timing needs confirmation, not established leakage. Sources: `r11/report.md` §T5, `STATE.md` (internal reports, not shipped; key numbers summarised here).
