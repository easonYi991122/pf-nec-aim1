import json
import numpy as np
import pandas as pd
import pytest
from explore.aim1b import stage12, stage34
from explore.i7 import diagnostics
from pf_nec import selectors
from test_gsafe import sample


def card_inputs():
    cards = [dict(id=cid, title="Synthetic candidate " + cid, rating="maybe",
                  time_zero="Synthetic decision", eligibility="Synthetic eligibility",
                  rescue="Synthetic rescue", additional_confounds="Synthetic confounders")
             for cid in stage34.RANK_ORDER]
    feature = dict(feature="ArterialLine_1", mean_abs=.1, normalized_share=.2,
                   top20_frequency=1., included_in_any_model=True)
    layer = dict(top20=[feature], rank_ci_support_ok=True)
    summary = dict(repeats=[1, 2, 3, 4, 5], T6_candidates=[], models={
        "T8-D5SAFE-LGB": dict(overall={"post": layer}, pod={"POD0-2": layer}, lead={"3": layer})})
    trajectory = dict(feature="ArterialLine_1", alignment="surgery", day="3",
                      eligible_rows="2", nonmissing="2", unknown_codes="0", nan_rows="0",
                      unavailable_rows="0", adjacent_calendar_pairs="1", available_pairs="1",
                      missingness_switches="0")
    permutation = dict(features='["ArterialLine_1"]', group="arterial", model="T8-D5SAFE-LGB",
                       mean_auc_loss="0.01", uncertainty='{"ci95": [-0.01, 0.03]}', status="synthetic")
    return cards, summary, [], [trajectory, trajectory], [permutation]


def test_aim1b_cards_trials_dags_and_ranking_end_to_end():
    cards = stage34.stage3_cards(*card_inputs())
    peripheral = next(c for c in cards if c["id"] == "arterial_1")
    umbilical = next(c for c in cards if c["id"] == "arterial_5")
    assert peripheral["prediction_contribution_and_stability"]["T8-D5SAFE-LGB"][0]["normalized_share"] == .2
    assert umbilical["prediction_contribution_and_stability"]["T8-D5SAFE-LGB"][0]["normalized_share"] is None
    assert peripheral["trajectory_measurement_support_pooled"][0]["eligible_rows"] == 4
    trials = [stage34.trial_skeleton(c) for c in cards if c["stage4_included"]]
    dags = [stage34.draft_dag(c) for c in cards if c["stage4_included"]]
    diagnostics = stage34.run_diagnostics(stage34.synthetic_tables(120), pods=(3,), graces=(1,))
    ranked = stage34.rank_cards(cards, diagnostics["diagnostics"])
    assert len(cards) == 13 and len(trials) == len(dags) == 9
    assert ranked[0]["id"] == "sternal_closure" and ranked[1]["id"] == "arterial_1"
    assert ranked[0]["representative_diagnostic"]["decision_POD"] == 3
    assert all(c["S_risk"] for c in cards)
    json.dumps(dict(trials=trials, dags=dags, cards=ranked), allow_nan=False)


def test_aim1b_cards_reject_inconsistent_aggregate_evidence():
    cards, summary, _, trajectory, permutation = card_inputs()
    summary["T6_candidates"] = [dict(candidate_id="x", features=[])]
    with pytest.raises(ValueError, match="R10CSVSummaryMismatch"):
        stage34.stage3_cards(cards, summary, [dict(candidate_id="x", features='["wrong"]')], trajectory, permutation)


def test_aim1b_synthetic_pipeline_and_embedded_causality_tests():
    result = stage12.synthetic_run()
    assert result["synthetic_only"]
    stage12.test_causality_future_changes_leave_prefix_unchanged()
    stage12.test_first_interval_selected_without_future_end()
    result = stage34.run_diagnostics(stage34.synthetic_tables(120), pods=(1,), graces=(1,))
    assert result["index_stays"] == 120
    assert len(result["diagnostics"]) > 0
    json.dumps(result, allow_nan=False)


def test_i7_nested_case_control_subsets_and_background_definition():
    _, meta = sample(80)
    ctx = selectors.FitContext()
    a = set(diagnostics.learning_curve_stays(meta, ctx, .25))
    b = set(diagnostics.learning_curve_stays(meta, ctx, .5))
    d = set(diagnostics.learning_curve_stays(meta, ctx, 1.))
    assert len(a) == 20 and len(b) == 40 and len(d) == 80 and a < b < d
    background = next(arm for arm in diagnostics.slice_arms() if arm.recipe == "B")
    assert background.k == 8 and background.window == 1


def test_i7_background_model_synthetic_end_to_end(tmp_path):
    train, test = sample(120), sample(30, 200)
    loader = lambda recipe, partition: test[0] if partition == 'test' else train[0]
    metadata = lambda partition: test[1] if partition == 'test' else train[1]
    engine = diagnostics.SliceEngine(selectors.FitContext(), loader, metadata, tmp_path / 'i7')
    arm = next(arm for arm in diagnostics.slice_arms() if arm.recipe == 'B')
    output = engine.apply(arm)
    predictions = pd.read_parquet(output['prediction_path'])
    assert len(predictions) == len(test[1]) and np.isfinite(predictions.probability).all()
    assert len(engine.ledger.records) == 4
