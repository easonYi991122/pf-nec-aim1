# Where we left off and what to resolve next

Interpretations and priorities on this page are labelled [inference] or [suggestion], not approved new studies or established conclusions. Sources: `ROUND_REDO_next_steps_zh.md`, `DEEPER_QUESTION_zh.md`, `ROUND_REDO_summary_zh.md` (internal reports, not shipped; key conclusions and numbers summarised here). Measured results are in [RESULTS](RESULTS.md).

[inference] The central open question is the undefined intended use: who sees the risk, when, and what changes afterward? Discrimination, AP and importance do not establish alert burden, bedside benefit or treatment effects. Surgical risk stratification, daily warning and causal intervention research require different evaluations.

|Priority|Suggested next step|Question / boundary|
|---|---|---|
|A1 Use-specific evaluation|[suggestion] Define use, then use existing out-of-fold predictions for calibration, risk-group incidence, sensitivity / lead time at fixed alert volume, and decision curves|These clinical-utility results have not been computed|
|A2 Simple surgical score versus daily model|[suggestion] Formally pair background / surgical models with G-safe and test incremental dynamic information|The 8-variable finding is only a 3-repeat screen without a paired test against G-safe|
|A3 Leave whole centres out|[suggestion] Freeze a leave-centers-out design and describe dispersion in centre NEC recording rates|Test transportability and coding reliance; centre differences cannot yet be attributed to recording differences|

[fact] Shipped `src/pf_nec/harness/harness_v3.py` already provides `fold_site`: 5 centre groups, keeping stays from each centre together. This round’s reported results use stay-based `fold_r*` splits; `fold_site` was not used to derive its conclusions. [suggestion] A3 can start from this field, but requires a separately frozen training / evaluation plan; an available split field is not a completed centre-held-out evaluation.

[suggestion] Pause architecture searches. Reopen only if the use discussion changes the primary metric, or new data / information plus a pre-frozen hypothesis justify a new study; a favourable subgroup point estimate alone is insufficient. Complete A1–A3 before deciding on new independent evaluation, expansion and causal analysis. The existing locked evaluation has been used; new independent evaluation needs a new-period or new-centre design.

[inference] I7 learning curves suggest sample size may be one limitation; they do not rule out architecture, optimisation or measurement limitations or guarantee benefit from expansion.

## Aim 1b: suggested ranking and questions for the PI

[suggestion] Sternal-closure timing is the primary candidate; peripheral arterial-line removal is the conditional backup; keep umbilical lines, defer ventilation, and treat drugs as snapshot comparisons only. This is a discussion ranking; no primary question is frozen and no treatment effect has been estimated.

[inference] The measured feasibility evidence informing this ranking is from `a2/report.md` (internal report, not shipped; key numbers summarised here), displayed at POD3 with a 1-day grace period. Sternal closure has at least 5 stays in each arm in 42/61 centres, maximum SMD falls from 0.24 to 0.07, and available events are 92–93; peripheral arterial-line propensity tails cover 45.5%, its maximum SMD remains 0.67 after weighting, with 177–183 available events; residual imbalance limits it to a conditional backup. Every candidate has severity-proxy risk; readiness, haemodynamic trends and intestinal perfusion are unmeasured. Event counts do not establish power, and low SMD does not establish exchangeability; this table must not select an optimal intervention day.

[suggestion] Ask the PI / clinicians these 7 questions before freezing (from `a2/report.md` §4):

1. Which infants and recorded readiness assessment allow both choices? Can circulation / perfusion, sedation, respiratory parameters and symptom onset be added?
2. Remove a specified line or all invasive monitoring? Are replacement / reinsertion allowed? Does a ventilation endpoint mean planned extubation and a sternal endpoint true closure?
3. What clinical times do calendar day, surgery completion, ICU admission and 24h mean? Can location, recording availability and same-day ordering be confirmed?
4. What are arm execution, at most 2 grace periods, rescue thresholds and deviation rules? What dose steps / shared drug history are required? The first 2h snapshot cannot substitute for assignment.
5. What are the Bell criteria, first / recurrent NEC definition, in-hospital POD31 target, death / discharge handling and minimum meaningful difference?
6. What restricted population is acceptable where centres / indications lack both arms? Complex propensity models or weight clipping cannot repair unmeasured confounding.
7. How will formal effect analysis define independent confirmation and multiplicity boundaries? The PI / controller freezes the final 1 primary + 1 backup question.

[suggestion] Remain at protocol / feasibility work until freezing; scores from altered prediction-model inputs are not treatment effects.

## Methods lead and Aim 2

[suggestion] B2 could become a discussion of perioperative evaluation methods: in PI72-CLEAN the measured cross-stage ranking makes total AUROC about 0.08 higher than post-op AUROC; Task A's pre-op-positive versus post-op-negative component offsets part of the opposite contribution. The direction depends jointly on composition and **actual ranking**, not inevitably on all-negative pre-op labels. See [EVALUATION](EVALUATION.md) for the identity and four cells. Sources: `ROUND_REDO_next_steps_zh.md`, `r9/report.md`, `r11/report.md`.

[inference] Aim 2 continuous physiological signals might inform dynamic prediction and readiness confounding, but overlapping patients currently cannot be linked; it cannot directly serve as Aim 1 external validation. This round did no new Aim 2 work, and the prior plan assigned low priority. [suggestion] Independently study the incremental value of dynamic signals over surgical stratification in its own data, then let intended use determine investment; federated learning is only an option. Sources: `ROUND_REDO_summary_zh.md`, `ROUND_REDO_next_steps_zh.md` (internal reports, not shipped; key status summarised here).
