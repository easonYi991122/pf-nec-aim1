# research use of identifying treatment-improvement space (U2) and postoperative-first-day stratification task (T-4): shared evaluation-contract summary

**The common evaluation addendum is installed and frozen: v1.2, 2026-10-07, after blind reviews.** T-4 is no longer blocked by the addendum. This self-contained summary does not replace the executable addendum; the repository maintainer can help verify matching configuration and hashes for reproduction. U2 is the research use of identifying treatment-improvement space; this step stratifies prognostic risk without estimating treatment effects. Postoperative-first-day stratification has run and been verified; results are below.

Time notation: postoperative day (POD), counted from zero on surgery day.

|Contract item|Primary row|Supplementary row|
|---|---|---|
|Prediction time and eligibility|End of POD1; hospitalized, alive and no previous necrotizing enterocolitis (NEC), one row per stay|Same eligibility and time|
|Endpoint|first in-hospital-outcome label for the next thirty days (yB30): first in-hospital NEC in the next 30 days, POD2–31|first-outcome label for the next three days (y3): next 3 days; fit separately for its own endpoint|
|Calibration, deciles, decision curve analysis (DCA)|Primary row only; methods and thresholds follow the frozen addendum|Not performed; relabelling a y3 model is not “recalibration”|
|Inputs and Light Gradient Boosting Machine learner code (LGB) recipe|Surgery-fixed columns, columns adding POD1 condition, and the LGB recipe: check against the installed addendum; the repository maintainer can help locate the execution files|Freeze separately for the corresponding endpoint; do not inherit an unconfirmed configuration|
|Observation rules|Follow the installed addendum for discharge / death, missing rows, late truncation, event deduplication and unknown labels; incomplete follow-up is not a negative label|Check the addendum observation rules for the corresponding endpoint|
|Evaluation scope|T-4 shares this primary-row contract; absolute risk and positive predictive value (PPV) are interpreted only within the outcome-conditioned development cohort (formal) contract|Fit and report separately from the primary row|

**Outcome selection in formal:** group d (NEC recorded after POD31) is excluded, so inclusion does not use only information known at prediction time. Calibration, decile risks, DCA and PPV therefore apply only within this formal contract, not as deployment-population absolute risks, net benefit or PPV. Disclose later-data-era flag with possible follow-up truncation (late_era) truncation; do not carve an independent holdout from existing development data.

If the use discussion with the principal investigator (PI) calls for the next 3 days, we jointly version the contract, make y3 primary and fit separately; predeclaration avoids choosing endpoints from favourable results. The addendum's 5 alerts per 100 at-risk patient-days, minimum 5-percentage-point detection gain and 3-day cooldown are research choices, not PI-endorsed clinical thresholds; daily alert capacity does not directly apply to landmark flags per stay.

Current evidence is conditional on availability by prediction time of 64 unresolved columns; independence is per stay, not per patient, with no reliable cross-stay patient identifier; the outcome-conditioned development cohort (formal). All results are retrospective internal cross-validation, without external validation. Source: verified research summary dated 2026-10-07; tasks and prepared code are in [TEAM_TASKS](TEAM_TASKS.md).

## Verified results from this round

Postoperative-first-day risk stratification (U2) has run as preregistered and been verified: the next 30 days are primary and the next 3 days supplementary. This run proceeded with the preregistered horizons without waiting for the PI's horizon response; this does not establish PI confirmation of clinical use.

This is prognostic-stratification preparation for treatment-improvement research; treatment effects have not been estimated.

Primary-horizon area under the receiver operating characteristic curve (AUROC) is 0.734 for fixed information plus clock and 0.736 for the richest model, a difference of 0.002 with an interval including 0: no clear gain at the first postoperative day. Raw scores show agreement in total event counts, with observed/expected ratios about 1.02–1.03; inner recalibration instead makes the richest model over-predict, with observed/expected 0.88. Decision-curve net benefit exceeds treat-all (the strategy of intervening in every eligible stay) at thresholds of about 2–5%; this applies only to the outcome-conditioned development cohort and cannot establish clinical net benefit. The endpoint covering the next 3 days has only 40 events and unreliable estimates. Source: this round's verified first-day stratification summary, replacing the previous awaiting-verification status. Observed/expected event-count ratios describe total event-count agreement in this development cohort; they alone do not establish calibration across risk levels. This decision-curve comparison does not establish any treatment effect or deployment net benefit.
