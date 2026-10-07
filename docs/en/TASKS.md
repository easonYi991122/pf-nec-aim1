# Task contracts: population, time, labels and evaluation

Group notation: a = first onset on postoperative days 4–31; b = surgery day through postoperative day 3; c = preoperative first onset; d = first onset after postoperative day 31; none = no known first-onset date, not proven absence of necrotizing enterocolitis. abc names the historical a/b/c target; the current outcome-conditioned development cohort (formal) uses a/b/none. Preoperative first onset and positive labels on preoperative rolling days are different definitions.

These tables use the frozen spec, row builder, `r9/report.md` and `r11/report.md`. Internal reports are not shipped; key numbers are summarised here. A contract fixes rows, labels, splits and the primary metric; do not subtract or rank area under the receiver operating characteristic curve (AUROC) across contracts.

## Population

Time notation: postoperative day (POD), counted from zero on surgery day.

|Item|local cleaned-table evaluation contract (PI72-CLEAN)|daily rolling prediction task (Task A) outcome-conditioned development cohort (formal)|
|---|---|---|
|Source / risk set|Local retrospective clean-table sampling; 318,992 labelled days, 11,931 stays, 356 source cases|v2.6 clean-linked a/b/none; 301,468 days, 11,674 stays, 359 case stays|
|Eligibility|Reconcile clean rows against the complete raw calendar, then exclude missing labels; retain clean-table days|Before first recorded necrotizing enterocolitis (NEC) and discharge, within original POD≤31 bound; retrospective inclusion by eventual event group|
|Main limitation|Not the paper's final scoring design; sampled days are not the deployment distribution|Formal differs from the original abc target; c/d excluded; none does not prove complete NEC-free follow-up|

## Time

|Concept|Rule|
|---|---|
|Shared timeline|days since admission (DOA)=current date−admission date; POD=current date−index surgery date; POD0 is post-op; end of day (EOD) means end of day|
|PI72-CLEAN|Date=index surgery date+clean day; integer days do not guarantee exactly 72 hours of lead time|
|Task A|At end of day t, predict t+1 through t+3; a pre-op day can be positive for future NEC|
|Availability|Mask surgery information pre-op; gate each historical day as of that day before constructing windows; gate intensive care unit (ICU) / first 2-hour fields accordingly|

## Labels

|Item|PI72-CLEAN|Task A formal|
|---|---|---|
|Event|First eligible post-op NEC, POD1–31|First recorded NEC during the stay|
|Daily label|Retain clean local cleaned-table single-positive-day label (outcome_3d); 1 positive day per case at max(0, NEC day−3)|first-outcome label for the next three days (y3)=first NEC in 1–3 days; at most 3 positive days per case|
|Observed composition|356 positive days; pre-op days all negative|1,066 positive days, including 110 pre-op days|

The single-positive-day rule is observed in the local clean table; it is not stated in the manuscript. The final scoring rows / phases behind the paper's 0.79 remain unresolved; this package neither supplies nor describes principal investigator (PI) code. The historical shorthand for the local cleaned-table contract (PI72) name is not an attribution claim.

## Features

|Model|Construction|Training boundary|
|---|---|---|
|daily availability-gated registry feature bank (D5-safe) / daily reference (A-D5-LGB)|Remove identifier of the removed historical preoperative-risk-factor field (PreopRiskFactor_330) from 605 rebuilt columns, leaving 604; mask 310 columns pre-op|Single day, fixed hyperparameters; choose stopping rounds in 3 inner folds and refit with their integer median|
|availability-gated encoding-enhanced feature scheme (G-safe) / encoding-enhanced Light Gradient Boosting Machine reference (GSAFE-LGB)|Same 604 columns + 9 raw-code encodings + 15 support-state columns + 1 centre shrinkage rate = 629|Refit encodings by cross-fitting within each training pool; no held-out selection|

## Splits

|Contract|Repeats and outer evaluation|Source-case / control train and test counts|
|---|---|---|
|PI72-CLEAN|one-based local-contract repeat identifiers (r1–5); one fixed held-out-stay split per repeat; seed=20260925+1000×r|Sample 1,039 controls; train 285/831, test 71/208; total train 1,116, test 279|
|Task A formal|zero-based daily-task repeat identifiers (r0–4), 5-fold stay out-of-fold predictions (OOF) per repeat; uses splits_v3|Same row set each repeat; do not multiply denominators by 5; v3 has no untouched holdout|

## Statistics

|Contract|Primary metric|Supplementary / key secondary metrics|
|---|---|---|
|PI72-CLEAN|POD≥0 AUROC|Total AUROC, phase composition / decomposition, average precision (AP) / sampled mean squared error of predicted probability against a binary outcome (Brier)|
|Task A formal|Total AUROC|Post-op AUROC is the key secondary; phase composition and four-cell decomposition|
|Shared rule|Pool target rows within repeat, then weight repeats equally|minimum detectable difference (MDD)=0.01; a 3-repeat screen cannot promote; require 5 complete repeats and simultaneous lower bound>0 from the complete family; see EVALUATION|

Stay-level splits do not establish independent infants across stays; linkage is unverified. The later-data-era flag with possible follow-up truncation (late_era) flag for truncated late-period data is for sensitivity analysis only. The new landmark risk-stratification task (Task B) driver is not shipped here; postoperative-first-day stratification has run locally and been verified; see the current update and has no current training entry point here; see [GLOSSARY](GLOSSARY.md) for historical contracts / aliases and [RESULTS](RESULTS.md) for historical identities.
