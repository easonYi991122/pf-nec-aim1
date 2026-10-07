# T-6: interface specification and open-items table

This is a patient-free synthetic interface proposal, not the complete input list of an existing scorer. Stage one can specify schemas and open items now; complete score recomputation remains BLOCKED until the controller freezes scorer identity, target, field mapping and generation conventions. The table specifies synthetic field names, units and availability times without copying real records, dictionary descriptions or weights.

|Field name|Unit / type|As-of time and role|
|---|---|---|
|synthetic_subject_key|Synthetic string, dimensionless|Row identity and splitting only, never a predictor or a real identifier|
|score_at|Timezone-aware timestamp|Current scoring-day end; a fixed cutoff, not inferred from a future event|
|gestational_age_weeks|Weeks, numeric|Birth record entered by cutoff; anchor-only input|
|diagnosis_code|Synthetic categorical code, dimensionless|Diagnosis first available by cutoff; anchor-only input|
|surgery_type|Synthetic categorical code, dimensionless|Visible only after surgery-day end and recorded surgical information; mask preoperatively; anchor-only input|
|pod|Days, integer|Calculated from index surgery already known by cutoff; POD0 is surgery day, missing preoperatively; anchor-only input|
|heart_rate_daily_mean|beats/min, numeric|Aggregate synthetic heart-rate observations from that day with availability_time no later than cutoff; signal-arm candidate, excluded from anchor-only|
|spo2_daily_mean|%, numeric|Aggregate synthetic oxygen-saturation observations from that day with availability_time no later than cutoff; signal-arm candidate, excluded from anchor-only|
|`registry_inputs.<field>.value`|Type declared in required_fields|Per-field mapping of the complete registry scorer input; names, units and availability gates await freezing, without substituting anchors for missing fields|
|`registry_inputs.<field>.unit`|Canonical unit string|Must match the frozen input definition; unknown or incompatible means invalid, without guessing conversions|
|`registry_inputs.<field>.timestamp` / `registry_inputs.<field>.availability_time`|Timezone-aware timestamp|Measurement / event time and recording availability respectively; neither may be later than cutoff|
|quality / missing_reason|Synthetic quality label / missing reason|Use only status known by cutoff; delayed or missing does not mean normal|
|scorer_id / scorer_version / target_id / required_fields|Interface metadata|Identity, target and required-field specifications frozen before calls, never chosen from validation outcomes|

Call contract: `score_registry(request) -> {synthetic_subject_key, score_at, scorer_id, scorer_version, target_id, score, valid, missing_inputs}`. Inputs are the metadata and predictors above, without labels. A valid score is a probability in 0–1; missing required fields, invalid units / availability or identity mismatch yields valid=false, score=null and explicit reasons, never a silent zero or anchor-score substitute. Test identical scores for identical aligned inputs, and no changes to past scores from future / late observations.

Target relationship: T-6’s simulated label for the next 30 days is an interface-testing convention, distinct from PI72-CLEAN outcome_3d, Task A y3 and U2’s real yB30. Define a separate synthetic scorer and fit its simulated target. Any later application of an existing scorer retains its original target identity as a baseline / prior input, without relabelling it as a new risk probability. Simulated labels require sufficient generated follow-up; unknown observation is not negative.

|Controller decision / delivery pending|Work allowed before receipt|
|---|---|
|Scorer identity, target, required_fields and field definitions / units / as-of mappings|Prepare the per-field table; mark unmatched inputs unavailable without claiming recomputation|
|Stay, event, death / discharge and missingness generation processes plus synthetic seeds|List boundary tests first; never select generating parameters by simulation performance|
|Missingness rates, arrival delays and sample sizes for 3 complete / missing / delayed scenarios|Specify configuration keys and open items without inventing numbers; retain caps of 4 CPU hours, 0 GPU hours and 0 real-outcome fits|
|Common rows and training / validation separation for anchor-only, score-alone, signals-alone and signals-plus-score|Design synthetic_subject_key separation tests; anchor-only uses exactly the fields marked above|
|Data holder, honest broker, IRB / DUA answers and validation-contamination audit|Complete governance questions without contacting institutions or assuming authorized linkage|

This page specifies an interface without implementing a scorer. See [TEAM_TASKS](TEAM_TASKS.md) for the task and null deliverables, and [AIM2_LINKAGE_NOTE](AIM2_LINKAGE_NOTE.md) for evidence and method boundaries.
