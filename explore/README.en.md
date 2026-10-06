# Exploratory analyses and scaffolding

[fact] `src/explore/` shares frozen interfaces with final models, but interpretation, diagnostics and causal drafts are not performance candidates to be classified by the same promotion gate. Freeze a separate plan for future analyses; see [STATUS_AND_NEXT](../docs/en/STATUS_AND_NEXT.md) for the architecture-search pause and reopening conditions.

|Entry|Supported scope|Work still required|
|---|---|---|
|T3 `explore.t3.interpret` / `t3_fast`|Grouped SHAP, time attribution, trajectory statistics; `gsafe_t3.packets(engine, output)` checks fixed predictions against saved models and creates in-memory packets|Assemble all Task A folds / repeats; lead needs independently checked event_day, not merely packet day; save aggregates only|
|I7 `explore.i7.diagnostics`|8-background-variable and B+C / B+S / B+C+S slices, windows 1 or 7; learning_curve_stays and learning_slopes retain methods|Original PI-29 curves cannot be rerun; freeze new-model curves separately; see GLOSSARY for B/C/S|
|Aim 1b `stage12` / `stage34`|Measurement, eligibility, positivity, aggregate card evidence, trial / DAG drafts and discussion ranking|PI review of readiness, time zero, actions, unmeasured confounding and effect protocol; no effect estimation|

I7 executable entry: `python -m explore.i7.diagnostics --repeat 1 --arm B-only-w1`, after build-data. Receipts are under `PF_RUN_DIR/explore-i7/<context>/`. Run each of the 3 screen repeats separately; this command only trains and saves predictions, not an accepted report or promotion conclusion. Use `pf_nec.inference.single_arm_intervals` for evaluation, taking targets independently from slice_provider test metadata.

TCN / GRU, orderless-control and M1/GPU training code are excluded; attempted results are in RESULTS. [suggestion] Pause architecture searches; consider reopening only if the use discussion changes the primary metric, or new data / information plus a pre-frozen hypothesis justify a new study. Importance alone cannot justify reopening; the conditions match [STATUS_AND_NEXT](../docs/en/STATUS_AND_NEXT.md).

## Aim 1b synthetic demonstration

[fact] These two CLIs output synthetic aggregates only.

```sh
python -m explore.aim1b.stage12 --synthetic
python -m explore.aim1b.stage34 --synthetic
```

Run the following with the configured interpreter to call the card, trial, DAG and ranking APIs end to end. Titles and protocol fields are **synthetic placeholders**; empty model evidence means not supplied, not zero importance. Real cards must come from an authorised protocol.

```python
from explore.aim1b import stage34 as s
cards = [dict(id=cid, title="Synthetic " + cid, rating="maybe",
              time_zero="Synthetic decision", eligibility="Synthetic eligibility",
              rescue="Synthetic rescue", additional_confounds="Synthetic confounders")
         for cid in s.RANK_ORDER]
summary = {"repeats": [1, 2, 3, 4, 5], "T6_candidates": [], "models": {}}
enriched = s.stage3_cards(cards, summary, [], [], [])
diagnostics = s.run_diagnostics(s.synthetic_tables(120), pods=(3,), graces=(1,))
ranked = s.rank_cards(enriched, diagnostics["diagnostics"])
trials = [s.trial_skeleton(c) for c in enriched if c["stage4_included"]]
dags = [s.draft_dag(c) for c in enriched if c["stage4_included"]]
assert len(ranked) == 13 and len(trials) == len(dags) == 9
```

## Card and aggregate-evidence schema

|Input|Required structure|
|---|---|
|cards|13 distinct IDs covering RANK_ORDER; each has title, rating, time_zero, eligibility, rescue and additional_confounds. stage3 retains extra original fields|
|summary|repeats must be [1,2,3,4,5]; models dictionary and T6_candidates list; empty lists mean no supplied evidence|
|Each models item|overall.post.top20; optional lag_profile; each pod / lead layer has top20 and rank_ci_support_ok|
|top20 / T6 contributions|feature, mean_abs, normalized_share, selection_frequency, repeat_rank_range, top10_frequency, top20_frequency, per_repeat, included_in_any_model; untabulated entries remain null|
|T6_candidates and t6_csv|Matching candidate_id sets; features, contribution_and_stability, POD_lead_profile and trajectory_support are JSON strings in CSV and must match summary exactly|
|trajectories|feature, alignment, day; eligible_rows, nonmissing, unknown_codes, nan_rows, unavailable_rows, adjacent_calendar_pairs, available_pairs, missingness_switches; measurement denominators only|
|permutation|model, group, features (JSON string), mean_auc_loss, uncertainty (JSON string containing ci95), status|
|diagnostics|The diagnostics list returned by run_diagnostics; rank_cards reads it without optimising ranking on outcome effects|

Pass aggregates, never patient rows, to stage3_cards. Call trial_skeleton and draft_dag only for stage4_included cards (9 questions); rank_cards takes all 13 cards. The historical R10 assembled report is not shipped; independently produced safe-model aggregates with this schema can use the functions, with missing attribution left missing.

[fact] `S_RISK` and `RANK_REASONS` retain their original Chinese strings; their meaning is: discuss sternal closure first because its endpoint is observable and measured overlap is better, keep peripheral arterial-line removal conditional on residual imbalance, retain umbilical lines, defer ventilation and operating-room extubation, and limit drugs to snapshot comparisons; every candidate has high or very high severity-proxy risk from unmeasured readiness, indications and treatment response, so this ranking establishes neither exchangeability nor treatment effects.

## Adapting authorised raw tables

[suggestion] Complete synthetic tests first, then run this measurement / overlap example privately. It reads raw tables using the same date parser and displays fixed POD3 with a 1-day grace period; it is not formal effect analysis. It fits treatment-proxy propensity models, **not NEC outcome models**. Apply the AGENTS resource-monitoring and stop rules.

```python
from pf_nec import config, contract, run
from explore.aim1b import stage12 as a1, stage34 as s
config.require_data()
config.initialize()
contract.reset_threads()
tables = {name: a1.read_table(name)[0] for name in
          ("IndexSurgHosp", "MechVent", "Sternum", "ArterialLine", "RiskSurgVIS", "NEC")}
tables["PreopRiskFactor"] = s.read_plain_table("PreopRiskFactor")[0]
result = s.run_diagnostics(tables, pods=(3,), graces=(1,))
run.write_json(config.writable(config.RUN_ROOT / "aim1b_diagnostics.json"), result)
```
