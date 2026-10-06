# Task contracts: population, time, labels and evaluation

[fact] These tables use the frozen spec, row builder, `r9/report.md` and `r11/report.md`. Internal reports are not shipped; key numbers are summarised here. A contract fixes rows, labels, splits and the primary metric; do not subtract or rank AUROC across contracts.

## Population

|Item|PI72-CLEAN|Task A formal|
|---|---|---|
|Source / risk set|Local retrospective clean-table sampling; 318,992 labelled days, 11,931 stays, 356 source cases|v2.6 clean-linked a/b/none; 301,468 days, 11,674 stays, 359 case stays|
|Eligibility|Reconcile clean rows against the complete raw calendar, then exclude missing labels; retain clean-table days|Before first recorded NEC and discharge, within original POD≤31 bound; retrospective inclusion by eventual event group|
|Main limitation|Not the paper's final scoring design; sampled days are not the deployment distribution|Formal differs from the original abc target; c/d excluded; none does not prove complete NEC-free follow-up|

## Time

|Concept|Rule|
|---|---|
|Shared timeline|DOA=current date−admission date; POD=current date−index surgery date; POD0 is post-op; EOD means end of day|
|PI72-CLEAN|Date=index surgery date+clean day; integer days do not guarantee exactly 72 hours of lead time|
|Task A|At end of day t, predict t+1 through t+3; a pre-op day can be positive for future NEC|
|Availability|Mask surgery information pre-op; gate each historical day as of that day before constructing windows; gate ICU / first 2-hour fields accordingly|

## Labels

|Item|PI72-CLEAN|Task A formal|
|---|---|---|
|Event|First eligible post-op NEC, POD1–31|First recorded NEC during the stay|
|Daily label|Retain clean outcome_3d; 1 positive day per case at max(0, NEC day−3)|y3=first NEC in 1–3 days; at most 3 positive days per case|
|Observed composition|356 positive days; pre-op days all negative|1,066 positive days, including 110 pre-op days|

The single-positive-day rule is observed in the local clean table; it is not stated in the manuscript. The final scoring rows / phases behind the paper's 0.79 remain unresolved; this package neither supplies nor describes PI code. The PI72 name is not an attribution claim.

## Features

|Model|Construction|Training boundary|
|---|---|---|
|D5-safe / A-D5-LGB|Remove PreopRiskFactor_330 from 605 rebuilt columns, leaving 604; mask 310 columns pre-op|Single day, fixed hyperparameters; choose stopping rounds in 3 inner folds and refit with their integer median|
|G-safe / GSAFE-LGB|Same 604 columns + 9 raw-code encodings + 15 support-state columns + 1 centre shrinkage rate = 629|Refit encodings by cross-fitting within each training pool; no held-out selection|

## Splits

|Contract|Repeats and outer evaluation|Source-case / control train and test counts|
|---|---|---|
|PI72-CLEAN|r1–5; one fixed held-out-stay split per repeat; seed=20260925+1000×r|Sample 1,039 controls; train 285/831, test 71/208; total train 1,116, test 279|
|Task A formal|r0–4, 5-fold stay OOF per repeat; uses splits_v3|Same row set each repeat; do not multiply denominators by 5; v3 has no untouched holdout|

## Statistics

|Contract|Primary metric|Supplementary / key secondary metrics|
|---|---|---|
|PI72-CLEAN|POD≥0 AUROC|Total AUROC, phase composition / decomposition, AP / sampled Brier|
|Task A formal|Total AUROC|Post-op AUROC is the key secondary; phase composition and four-cell decomposition|
|Shared rule|Pool target rows within repeat, then weight repeats equally|MDD=0.01; a 3-repeat screen cannot promote; require 5 complete repeats and simultaneous lower bound>0 from the complete family; see EVALUATION|

Stay-level splits do not establish independent infants across stays; linkage is unverified. The late_era flag for truncated late-period data is for sensitivity analysis only. Task B was not retrained this round and has no current training entry point here; see [GLOSSARY](GLOSSARY.md) for historical contracts / aliases and [RESULTS](RESULTS.md) for historical identities.
