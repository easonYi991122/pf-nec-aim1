# U2 and T-4: shared evaluation-contract summary

**BLOCKED: the common evaluation addendum is under review; T-4 execution requires the controller to install an approved, versioned and hashed addendum.** This self-contained summary does not replace it. Prepare protocols and synthetic boundary tests only. U2 means identifying treatment-improvement space; this stage stratifies prognostic risk without estimating treatment effects.

|Contract item|Primary row|Supplementary row|
|---|---|---|
|Prediction time and eligibility|End of POD1; hospitalized, alive and no previous NEC, one row per stay|Same eligibility and time|
|Endpoint|yB30: first in-hospital NEC in the next 30 days, POD2–31|y3: next 3 days; fit separately for its own endpoint|
|Calibration, deciles, DCA (decision curves)|Primary row only; methods and thresholds arrive with the approved addendum|Not performed; relabelling a y3 model is not “recalibration”|
|Inputs and LGB recipe|Surgery-fixed columns, columns adding POD1 condition, and the LGB recipe: to be supplied by the controller with the installed addendum|Freeze separately for the corresponding endpoint; do not inherit an unconfirmed configuration|
|Observation rules|The installed addendum must specify discharge / death, missing rows, late truncation, event deduplication and unknown labels; incomplete follow-up is not a negative label|Observation rules also require freezing|
|Evaluation scope|T-4 shares this primary-row contract; absolute risk and PPV are interpreted only within the formal contract|Fit and report separately from the primary row|

**Outcome selection in formal:** group d (NEC recorded after POD31) is excluded, so inclusion does not use only information known at prediction time. Calibration, decile risks, DCA and PPV therefore apply only within this formal contract, not as deployment-population absolute risks, net benefit or PPV. Disclose late_era truncation; do not carve an independent holdout from existing development data.

If the PI establishes that POD1 decisions concern only the next 3 days, the controller must version the contract, make y3 primary and fit separately; never switch endpoints based on favourable results. Until the addendum is installed, do not invent column names, LGB parameters, calibrators or thresholds. The project lead supplies its installed location and hash through [TEAM_TASKS](TEAM_TASKS.md).
