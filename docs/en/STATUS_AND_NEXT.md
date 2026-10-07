# Where we left off and what to resolve next

This page distinguishes facts, approved planning decisions and research suggestions awaiting freeze; planning approval is not acceptance of real runs or scientific conclusions. Sources: `ROUND_REDO_next_steps_zh.md`, `DEEPER_QUESTION_zh.md`, `ROUND_REDO_summary_zh.md` (internal reports, not shipped; key conclusions and numbers summarised here). Measured results are in [RESULTS](RESULTS.md).

[inference] The central open question is the undefined intended use: who sees the risk, when, and what changes afterward? Discrimination, AP and importance do not establish alert burden, bedside benefit or treatment effects. Surgical risk stratification, daily warning and causal intervention research require different evaluations.

|Priority|Suggested next step|Question / boundary|
|---|---|---|
|A1 Use-specific evaluation|[suggestion] Freeze intended use and inner threshold / calibration rules before reporting risk stratification and alert policies|Historical alert-burden descriptions exist; their threshold and formal population do not represent deployment burden|
|A2 Simple surgical score versus daily model|[suggestion] Formally pair background / surgical models with G-safe and test incremental dynamic information|The 8-variable finding is only a 3-repeat screen without a paired test against G-safe|
|A3 Leave whole centres out|[suggestion] Freeze the new study separately from historical site-held-out evaluation to test transportability and coding reliance|E9-A-D5-site already completed historical site-held-out evaluation; do not subtract stay-split results or attribute centre differences to recording differences|

[fact] Shipped harness v3 provides `fold_site`, assigning centres to 5 fixed groups. Historical E9-A-D5-site used harness v3 1×5 leave-site-out evaluation, AUROC 0.6950 [0.6732, 0.7174]; this historical sensitivity evaluation is complete. Current final-model reports use stay-based `fold_r*`; the validation schemes differ and must not be subtracted. A3 still needs its new plan frozen separately. Source: run registry E9-A-D5-site; see [RESULTS](RESULTS.md).

[Historical fact] XGB-A-D5-phase_split described alerts in the formal cohort: about 5.66 per 100 patient-days and case detection 0.25, using a threshold based on negative-day 5% FPR. This is not deployment alert burden: formal was selected using future outcomes and the threshold was not frozen for clinical capacity. New evaluation must report actual burden, detection, PPV, alerts per 1 detected case and missed cases, separating retrospective capacity curves from inner-threshold policies. Sources: historical run registry; harness v3 validation, with alert calculations inherited from harness_v2.

[Approved decision] A7 pauses main-line architecture search under DEC-019 as an opportunity-cost decision: tested structural hypotheses failed the frozen promotion gate, further selection on reused splits accumulates forking paths, and intended use remains unfixed with the PI so the primary metric may change. It does not invoke the saturation stopping rule: the rule requires slope <0.005 per doubling of cases, whereas the descriptive measured slope is about +0.03, so the first condition is not met; the observed range is 71–285 training cases, 3 repeats and a single historical recipe. The pause proves neither that architectures are useless nor that data volume is the sole bottleneck. Restart requires a distinguishable mechanism with frozen metric / budget / stopping rule, a use-driven primary-metric change, or credible new information; a new architecture name or seeds alone are insufficient. Teammates may develop T-2/T-3 under separately frozen cards.

[Approved intent] D1: do not carve an “untouched holdout” from data already used for development. Record only the intent to isolate future new data, sealing access and training / selection logs before the controller freezes an independent evaluation contract. Existing r1–5 / r0–4 results are replication / development evidence.

[Dictionary fact] ProcLoc 9 = Cardiac OR, not a centre code; G-safe’s centre encoding is site_eb. Source: ProcLoc in the authorized data dictionary.

See [TEAM_TASKS](TEAM_TASKS.md) for teammate contracts and prerequisites; D3 retains PI10 as an interpretation group without exposing its learner.

[Historical correction] At POD0–2, TCN versus the w7-R1 tree is −0.0026; +0.0416 versus the orderless MLP is not the largest of 16 sequence cells, with +0.0462 versus the w7-R1 tree at POD8–14. These are descriptive cells without across-POD multiplicity correction, not grounds for choosing a maximal-cell primary endpoint. The M1 menu included GAIN50; r1 GRU-D chose GAIN50-w7. Source section names are in the [RESULTS historical-correction table](RESULTS.md).

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

[fact] B1 literature scouting is complete (not a systematic review): without a linkage key, help is limited to score recomputation or within-signal-cohort distillation; individual fusion requires authorized linkage at an institution holding both sources. No Aim 2 modelling was done; next steps are synthetic interface specification / prototyping and governance preparation only. See [AIM2_LINKAGE_NOTE](AIM2_LINKAGE_NOTE.md). Uses are U1 dynamic warning and U2 treatment-improvement space; B1 priority rises if dynamic gain proves small.
