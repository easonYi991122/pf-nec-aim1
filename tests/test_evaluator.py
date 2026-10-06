import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from pf_nec import evaluate as ev
from explore.t3 import interpret as it


def target_rows(repeats=(1, 2, 3, 4, 5), n_stays=40, pods=(-1, 1, 3, 8, 16)):
    records = []
    for repeat in repeats:
        for h in range(n_stays):
            case = h < n_stays // 2
            for pod in pods:
                records.append(dict(repeat=repeat, row_key=f"{h}:{pod}", stay=h + .125,
                                    y=int(case and pod == 3), pod=pod, case=int(case)))
    return pd.DataFrame(records)


def prediction_table(rows, probability):
    return rows.assign(probability=np.broadcast_to(probability, len(rows)))


def family_predictions(rows, frame="PI72-CLEAN"):
    family = ev.frozen_family(frame)
    refs = {c["reference"] for c in family}
    programs = {c["arm"] for c in family}
    predictions = {a: prediction_table(rows, .5) for a in refs | programs}
    # T8 passes local and GSAFE; tree/sequence dependencies intentionally remain
    # tied so one passing local claim cannot promote unrelated programs.
    arm = "T8-SELECT-LGB" if frame == "PI72-CLEAN" else "A-T8-GAIN"
    predictions[arm] = prediction_table(rows, .1 + .8 * rows.y.to_numpy())
    return predictions


def layout():
    return pd.DataFrame([
        dict(column="signal_lag0", feature="signal", lag=0, kind="value"),
        dict(column="signal_lag1", feature="signal", lag=1, kind="value"),
        dict(column="signal_missing1", feature="signal", lag=1, kind="missingness"),
        dict(column="other_lag0", feature="other", lag=0, kind="value"),
    ])


def packets(rows, *, values=True, chunk=53):
    result = []
    for _, part in rows.groupby("repeat", sort=True):
        for start in range(0, len(part), chunk):
            block = part.iloc[start:start + chunk].reset_index(drop=True)
            # Opposing lags expose abs(sum) vs sum(abs); one missingness column.
            phi = np.column_stack([1 + .01 * block.pod, -np.ones(len(block)), .2 * np.ones(len(block)), .1 * np.ones(len(block))])
            packet = dict(rows=block, phi=phi, base=np.zeros(len(block)), margin=phi.sum(axis=1), layout=layout(),
                          model_id=f"synthetic-r{int(block.repeat.iloc[0])}")
            if values:
                packet["values"] = np.ones_like(phi)
            result.append(packet)
    return result


@pytest.mark.parametrize("mask", [None, np.array([True, True, True, False, True, True])])
def test_cached_weighted_metrics_match_sklearn_with_ties(mask):
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([.2, .2, .8, .9, .8, .2])
    w = np.array([3., 1., 2., 0., 4., 2.])
    choose = w > 0 if mask is None else (w > 0) & mask
    got = ev.BinaryMetric(y, p, mask).score(w)
    assert got["auc"] == pytest.approx(roc_auc_score(y[choose], p[choose], sample_weight=w[choose]))
    assert got["ap"] == pytest.approx(average_precision_score(y[choose], p[choose], sample_weight=w[choose]))
    assert got["brier"] == pytest.approx(brier_score_loss(y[choose], p[choose], sample_weight=w[choose]))


def test_probability_validation_and_empty_scores():
    for probability in ([np.nan], [np.inf], [-.1], [1.1]):
        with pytest.raises(ev.EvaluationError):
            ev.BinaryMetric([0], probability)
    metric = ev.BinaryMetric([0, 1], [.2, .8])
    with pytest.raises(ev.EvaluationError):
        metric.score([-1, 2])
    assert all(np.isnan(v) for v in metric.score([0, 0]).values())


def test_stage_decomposition_and_direction_not_assumed():
    rows = target_rows(repeats=(1,), n_stays=8)
    # Raising preop scores lowers cross-stage ordering without changing post AUC.
    p = .2 + .4 * rows.y.to_numpy()
    good = ev.score_rows(rows, p, frame="PI72-CLEAN")
    p[rows.pod.lt(0)] = .9
    bad = ev.score_rows(rows, p, frame="PI72-CLEAN")
    assert good["metrics"]["AUROC_post"] == bad["metrics"]["AUROC_post"] == 1
    assert bad["metrics"][ev.CROSS] == 0
    assert bad["metrics"]["AUROC_total"] < bad["metrics"]["AUROC_post"]
    assert good["metrics"]["AUROC_preop_within_phase"] is None
    assert bad["decomposition"]["identity_absolute_error"] < 1e-12
    assert bad["counts"]["preop_negative_fraction"] == pytest.approx(8 / 36)
    assert "AUROC_pre" not in bad["metrics"]
    assert json.loads(json.dumps(bad, allow_nan=False))["metrics"][ev.CROSS] == 0


@pytest.mark.parametrize("subset", ["post", "negative", "positive", "empty"])
def test_empty_stage_or_class_returns_null_and_zero_mass_identity(subset):
    rows = target_rows(repeats=(1,), n_stays=8)
    mask = {"post": rows.pod.ge(0), "negative": rows.y.eq(0), "positive": rows.y.eq(1), "empty": np.zeros(len(rows), bool)}[subset]
    rows = rows.loc[mask]
    result = ev.score_rows(rows, .1 + .8 * rows.y.to_numpy(), frame="PI72-CLEAN")
    json.dumps(result, allow_nan=False)
    if subset == "post":
        assert result["metrics"][ev.CROSS] is None
        assert result["decomposition"]["components"][0]["contribution"] == 0
        assert result["decomposition"]["identity_absolute_error"] == 0
    else:
        assert result["metrics"]["AUROC_total"] is None


def test_task_a_four_component_identity_with_preop_cases():
    rows = target_rows(repeats=(0,), n_stays=8)
    rows.loc[(rows.stay < 2) & rows.pod.lt(0), "y"] = 1
    p = np.linspace(.01, .99, len(rows))
    result = ev.score_rows(rows, p, frame="A-formal")
    assert len(result["decomposition"]["components"]) == 4
    assert result["decomposition"]["reconstructed_total_auc"] == pytest.approx(roc_auc_score(rows.y, p))
    with pytest.raises(ValueError):
        ev.score_rows(rows, p, frame="PI72-CLEAN")


def test_exact_alignment_restores_shuffled_keys_and_fractional_stays():
    rows = target_rows(repeats=(1,))
    prediction = prediction_table(rows, np.linspace(0, 1, len(rows))).sample(frac=1, random_state=42)
    actual, aligned = ev.align_predictions(rows, {"arm": prediction}, frame="PI72-CLEAN")
    assert actual.stay.equals(rows.stay)
    assert np.array_equal(aligned["arm"], np.linspace(0, 1, len(rows)))


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "label", "stay", "pod", "case", "phase", "frame"])
def test_alignment_never_intersects_or_repairs_metadata(defect):
    rows = target_rows(repeats=(1,))
    prediction = prediction_table(rows, .5)
    if defect == "missing":
        prediction = prediction.iloc[:-1]
    elif defect == "extra":
        prediction = pd.concat([prediction, prediction.iloc[:1].assign(row_key="new")])
    elif defect == "duplicate":
        prediction.loc[1, "row_key"] = prediction.loc[0, "row_key"]
    elif defect == "label":
        prediction.loc[0, "y"] = 1
    elif defect == "phase":
        prediction["post_surg"] = 0
    elif defect == "frame":
        prediction["frame"] = "A-formal"
    else:
        prediction.loc[0, defect] += 1
    with pytest.raises(ev.EvaluationError):
        ev.align_predictions(rows, {"arm": prediction}, frame="PI72-CLEAN")


def test_target_validation_refuses_lossy_keys_wrong_repeats_and_oof_leaks():
    rows = target_rows(repeats=(1,))
    with pytest.raises(ev.EvaluationError, match="float32"):
        ev.validate_targets(rows.assign(stay=rows.stay.astype("float32")), "PI72-CLEAN")
    with pytest.raises(ev.EvaluationError, match="numbering"):
        ev.validate_targets(rows.assign(repeat=0), "PI72-CLEAN")
    a = target_rows(repeats=(0,)).assign(fold=lambda r: r.stay.astype(int) % 5)
    ev.validate_targets(a, "A-formal")
    with pytest.raises(ev.EvaluationError, match="all five"):
        ev.validate_targets(a[a.fold != 4], "A-formal")
    a.loc[0, "fold"] = 1
    with pytest.raises(ev.EvaluationError, match="multiple OOF"):
        ev.validate_targets(a, "A-formal")


def test_bootstrap_shares_multiplicity_for_same_stay_across_repeats_and_phases():
    rows = target_rows(repeats=(1, 2))
    # Partially overlapping test sets; union is not a stack of independent rows.
    rows = rows.loc[~((rows.repeat == 2) & (rows.stay < 10))].reset_index(drop=True)
    plan = ev.StayBootstrap(rows)
    rng = np.random.default_rng(31)
    for _ in range(10):
        draw = plan.draw(rng)
        weight = plan.row_weights(draw)
        assert pd.Series(weight).groupby(rows.stay).nunique().max() == 1
        assert draw[plan.strata[0]].sum() == len(plan.strata[0])
        assert draw[plan.strata[1]].sum() == len(plan.strata[1])
    broken = rows.copy()
    broken.loc[(broken.repeat == 2) & (broken.stay == 10.125), "case"] = 0
    with pytest.raises(ev.EvaluationError, match="conflicting"):
        ev.StayBootstrap(broken)
    with pytest.raises(ev.EvaluationError):
        plan.row_weights(np.ones(len(plan.stays) - 1))


@pytest.mark.parametrize("size", [1, 3, 4, 7, 10])
def test_arbitrary_contrast_family_exact_centered_max_error(size):
    estimate = np.linspace(.02, .2, size)
    samples = estimate + np.linspace(-.03, .04, 2000)[:, None] * np.arange(1, size + 1)
    actual = ev.bootstrap_intervals(estimate, samples, list(range(size)))
    q = np.quantile(np.max(np.abs(samples - estimate), axis=1), .95, method="linear")
    assert actual["family_size"] == size
    assert actual["simultaneous"]["q"] == pytest.approx(q)
    assert np.asarray(actual["simultaneous"]["ci95"]) == pytest.approx(np.column_stack([estimate - q, estimate + q]))


def test_ci_common_validity_uses_whole_declared_family_no_redraw_or_shrinking():
    estimate = np.array([.05, .06, .07])
    samples = np.tile(estimate, (2000, 1))
    samples[:6, 0], samples[6:12, 1] = np.nan, np.nan
    result = ev.bootstrap_intervals(estimate, samples, [0, 1, 2])
    assert result["ordinary"][0]["percentile_ci95"] is not None
    assert result["common_valid_draws"] == 1988
    assert result["simultaneous"] is None
    samples[10:12, 1] = .06
    assert ev.bootstrap_intervals(estimate, samples, [0, 1, 2])["common_valid_draws"] == 1990
    with pytest.raises(ev.EvaluationError):
        ev.bootstrap_intervals(estimate, samples[:1999], [0, 1, 2])
    for family in ([0, 0], [-1], [3]):
        with pytest.raises(ev.EvaluationError):
            ev.bootstrap_intervals(estimate, samples, family)


@pytest.mark.parametrize("stage,repeats,lower,delta,complete,expected", [
    ("confirm", [1, 2, 3, 4, 5], .001, .01, True, True),
    ("screen", [1, 2, 3, 4, 5], .4, .5, True, False),
    ("screen", [1, 2, 3], .4, .5, True, False),
    ("confirm", [1, 2, 3], .4, .5, True, False),
    ("confirm", [1, 2, 3, 4, 5], 0., .5, True, False),
    ("confirm", [1, 2, 3, 4, 5], .001, .0099, True, False),
    ("confirm", [1, 2, 3, 4, 5], .001, .5, False, False),
])
def test_promotion_requires_five_repeats_effect_size_and_simultaneous_lower_bound(stage, repeats, lower, delta, complete, expected):
    result = ev.promotion_rule(delta, [lower, .9], frame="PI72-CLEAN", repeats=repeats,
                               stage=stage, family_complete=complete, common_valid_draws=2000)
    assert result["passes"] is expected
    if stage == "screen":
        assert "screen_never_promotes" in result["reasons"]


def test_mdd_cannot_be_lowered_and_higher_mdd_applies():
    args = dict(frame="PI72-CLEAN", repeats=[1, 2, 3, 4, 5], stage="confirm", family_complete=True, common_valid_draws=2000)
    assert not ev.promotion_rule(.015, [.002, .03], mdd=.02, **args)["passes"]
    with pytest.raises(ev.EvaluationError):
        ev.promotion_rule(.02, [.002, .03], mdd=.005, **args)


def test_end_to_end_ten_contrast_family_and_claims():
    rows = target_rows(pods=(-1, 3))
    result = ev.evaluate(rows, family_predictions(rows), frame="PI72-CLEAN", stage="confirm")
    assert result["attempted_draws"] == result["common_valid_draws"] == 2000
    assert len(result["comparisons"]) == 10
    assert result["claims"]["T8-SELECT-LGB"]["PI_champion_candidate"]
    assert not result["claims"]["T4-SELECT-GRU"]["PI_champion_candidate"]
    json.dumps(result, allow_nan=False)


def test_equal_repeat_mean_not_pooled_and_descriptive_family_cannot_promote():
    rows = target_rows(repeats=(1, 2), pods=(3,))
    rows = rows.loc[(rows.repeat == 1) | ((rows.stay >= 10) & (rows.stay < 30))].reset_index(drop=True)
    p = np.where(rows.repeat.eq(1), .1 + .8 * rows.y, .9 - .8 * rows.y)
    contrast = [dict(arm="new", reference="reference", metric="AUROC_post")]
    result = ev.evaluate(rows, {"new": prediction_table(rows, p), "reference": prediction_table(rows, .5)},
                         frame="PI72-CLEAN", stage="screen", contrasts=contrast, family=contrast)
    assert result["comparisons"][0]["delta"] == pytest.approx(0.)
    assert result["arm_intervals"]["new"]["AUROC_post"]["estimate"] == pytest.approx(.5)
    assert roc_auc_score(rows.y, p) != pytest.approx(.5)
    assert not result["comparisons"][0]["promotion"]["passes"]
    with pytest.raises(ev.EvaluationError, match="missing"):
        ev.evaluate(rows, {"new": prediction_table(rows, p), "reference": prediction_table(rows, .5)},
                    frame="PI72-CLEAN", stage="confirm", contrasts=contrast)


def test_task_a_pools_five_oof_folds_and_separate_family():
    rows = target_rows(repeats=(0, 1, 2, 3, 4), pods=(-1, 3)).assign(fold=lambda r: r.stay.astype(int) % 5)
    result = ev.evaluate(rows, family_predictions(rows, "A-formal"), frame="A-formal", stage="confirm")
    assert result["primary_metric"] == "AUROC_total"
    assert len(result["comparisons"]) == 4
    assert result["claims"]["A-T8-GAIN"]["local_improvement"]
    assert "PI_champion_candidate" not in result["claims"]["A-T8-GAIN"]


def test_stratified_metrics_support_threshold_and_empty_strata():
    rows = target_rows(repeats=(1,), pods=(-1, 3), n_stays=24)
    result = ev.time_stratified_metrics(rows, {"arm": prediction_table(rows, .1 + .8 * rows.y)}, frame="PI72-CLEAN")
    assert result["strata"]["POD3-7"]["arms"]["arm"]["auc"]["estimate"] == 1
    assert result["strata"]["preop"]["arms"]["arm"]["auc"]["estimate"] is None
    assert result["strata"]["POD8-14"]["arms"]["arm"]["brier"]["estimate"] is None
    few = rows.loc[rows.stay < 18]
    result = ev.time_stratified_metrics(few, {"arm": prediction_table(few, .1 + .8 * few.y)}, frame="PI72-CLEAN")
    assert result["strata"]["POD3-7"]["arms"]["arm"]["auc"]["estimate"] is None


def test_shap_group_cancellation_additivity_lags_and_unselected():
    phi = np.array([[3., -2., .5, -.25], [-1., 2., -.5, .25]])
    margin = phi.sum(axis=1) + .75
    it.check_shap_additivity(phi, .75, margin)
    grouped = it.group_shap(phi, layout(), feature_universe=["signal", "other", "unselected"])
    assert grouped["abs_sum"][0, 0] == 5.5
    assert grouped["net_abs"][0, 0] == 1.5
    assert grouped["observation_abs"][0, 0] == .5
    assert grouped["lag_profile"]["signal", 1][0] == 2.5
    assert not grouped["included"][2]
    assert np.all(grouped["abs_sum"][:, 2] == 0)
    with pytest.raises(ev.EvaluationError, match="additivity"):
        it.check_shap_additivity(phi, .75, 1 / (1 + np.exp(-margin)))
    wrong = {**grouped, "abs_sum": grouped["net_abs"]}
    with pytest.raises(ev.EvaluationError, match="group-sum"):
        it.check_group_sums(phi, wrong)


def test_r3_grouping_all_summaries_and_observation_component():
    columns = ["current", "mean", "min", "max", "delta", "changes", "n_observed"]
    meta = [dict(column=f"a__{c}", feature="a", lag=None, kind="observation" if c == "n_observed" else "value") for c in columns]
    grouped = it.group_shap(np.arange(7)[None, :], meta)
    assert grouped["abs_sum"][0, 0] == 21
    assert grouped["observation_abs"][0, 0] == 6
    assert grouped["lag_profile"] == {}


@pytest.mark.parametrize("defect", ["duplicate", "width", "future_lag", "forbidden", "forbidden_dependency"])
def test_shap_lineage_negative_fixtures(defect):
    meta = layout()
    if defect == "duplicate":
        meta.loc[1, "column"] = meta.loc[0, "column"]
    elif defect == "width":
        meta = meta.iloc[:-1]
    elif defect == "future_lag":
        meta.loc[1, "lag"] = -1
    elif defect == "forbidden":
        meta.loc[0, "feature"] = "necbelldtshift"
    else:
        meta["dependencies"] = [[], ["hospdischstat"], [], []]
    with pytest.raises(ev.EvaluationError):
        it.group_shap(np.ones((2, 4)), meta)


def test_shap_overall_time_strata_uncertainty_and_coverage():
    rows = target_rows(repeats=(1, 2), n_stays=40, pods=(1, 3, 8))
    rows = rows.assign(day=rows.pod, event_day=np.where(rows.case, 12., np.nan))
    result = it.shap_importance(rows, packets(rows), frame="PI72-CLEAN", feature_universe=["signal", "other", "unselected"], view="pod")
    group = result["strata"]["POD3-7"]
    signal = group["features"][0]
    assert signal["mean_abs"] == pytest.approx(2.23)
    assert signal["net_mean_abs"] == pytest.approx(.23)
    assert signal["rank_interval"]["ci95"] == [1, 1]
    assert signal["selection_frequency"] == signal["top10_frequency"] == 1
    assert group["features"][2]["mean_abs"] == 0
    assert not group["features"][2]["included_in_any_model"]
    assert result["strata"]["POD15-31"]["features"][0]["mean_abs"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ev.EvaluationError, match="coverage"):
        it.shap_importance(rows, packets(rows)[:-1], frame="PI72-CLEAN", feature_universe=["signal", "other"])


def donor_rows(n=12):
    return pd.DataFrame(dict(stay=np.arange(n) + .125, pod=3,
                             row_observed_pattern="11", surgery_pattern="11", two_hour_pattern="01"))


def test_donors_match_structural_patterns_no_self_outcome_or_order_dependency():
    rows = donor_rows()
    rows.loc[6:, "two_hour_pattern"] = "11"
    donor, coverage = it.matched_donors(rows, np.random.default_rng(17))
    assert coverage["coverage"] == 1
    assert (rows.stay.to_numpy()[donor] != rows.stay).all()
    for key in ("pod", "row_observed_pattern", "surgery_pattern", "two_hour_pattern"):
        assert (rows[key].to_numpy()[donor] == rows[key]).all()
    noisy = rows.assign(y=np.arange(len(rows)) % 2, case=np.arange(len(rows)) > 3, event_day=np.arange(len(rows)) * 999)
    assert np.array_equal(it.matched_donors(noisy, np.random.default_rng(17))[0], donor)
    shuffled = rows.sample(frac=1, random_state=7).reset_index(drop=True)
    other = it.matched_donors(shuffled, np.random.default_rng(17))[0]
    assert dict(zip(shuffled.stay, shuffled.iloc[other].stay)) == dict(zip(rows.stay, rows.iloc[donor].stay))
    small, coverage = it.matched_donors(rows.iloc[:4], np.random.default_rng(17))
    assert np.array_equal(small, np.arange(4)) and coverage["coverage"] == 0
    with pytest.raises(ev.EvaluationError):
        it.matched_donors(rows.drop(columns="surgery_pattern"), np.random.default_rng(1))


def trajectory_fixture():
    rows = target_rows(repeats=(1,), n_stays=16, pods=(-1, 0, 1, 3, 8)).drop(columns="repeat")
    rows["day"] = rows.pod
    rows["event_day"] = np.where(rows.case, 6., np.nan)
    rows["y"] = (rows.case.eq(1) & rows.pod.eq(3)).astype(int)
    values = pd.DataFrame({"dose": rows.pod.to_numpy(float), "support": (rows.pod.to_numpy() > 0).astype(float)}, index=rows.row_key)
    values.loc[rows.loc[rows.pod.eq(0), "row_key"], "support"] = 9
    events = pd.concat([rows.loc[rows.case.eq(1), ["stay", "event_day"]].drop_duplicates(),
                        rows.loc[rows.case.eq(1), ["stay", "event_day"]].drop_duplicates().assign(event_day=-2)])
    definitions = {"dose": dict(kind="continuous", unknown_codes=[], counter=True),
                   "support": dict(kind="binary", unknown_codes=[9])}
    return rows, values, events, definitions


def test_trajectory_label_boundary_unknowns_gaps_and_future_perturbation():
    rows, values, events, definitions = trajectory_fixture()
    result = it.trajectories(rows, values, definitions, frame="PI72-CLEAN", alignment="surgery")
    summaries = {(x["feature"], x["day"], x["cohort"]): x for x in result["summaries"]}
    assert summaries["dose", 8, "case"]["eligible_rows"] == 0
    assert summaries["dose", 8, "control"]["eligible_rows"] == 8
    assert summaries["dose", 3, "case"]["available_pairs"] == 0  # calendar gap 1 -> 3
    assert summaries["dose", 1, "case"]["counter_advances"] == 8
    assert summaries["support", 0, "case"]["unknown_codes"] == 8
    assert summaries["support", 0, "case"]["known_positive_fraction"] is None
    changed = values.copy()
    changed.loc[rows.loc[rows.case.eq(1) & rows.day.gt(3), "row_key"], "dose"] = 1e12
    repeated = it.trajectories(rows, changed, {"dose": definitions["dose"]}, frame="PI72-CLEAN", alignment="surgery")
    assert [x for x in result["summaries"] if x["feature"] == "dose"] == repeated["summaries"]
    json.dumps(result, allow_nan=False)


def test_event_anchor_uses_first_eligible_postop_not_first_any_and_keeps_missing_days():
    rows, values, events, definitions = trajectory_fixture()
    anchors = it.validate_event_anchors(rows, events, frame="PI72-CLEAN")
    assert (anchors == 6).all()  # previous -2 NEC is not the eligible anchor
    bad = events.copy()
    bad.loc[bad.event_day.eq(6), "event_day"] = 5
    with pytest.raises(ev.EvaluationError, match="disagrees"):
        it.validate_event_anchors(rows, bad, frame="PI72-CLEAN")
    with pytest.raises(ev.EvaluationError, match="independently"):
        it.validate_event_anchors(rows, None, frame="PI72-CLEAN")
    result = it.trajectories(rows, values, {"dose": definitions["dose"]}, frame="PI72-CLEAN", alignment="event", eligible_events=events)
    by_day = {(x["day"], x["cohort"]): x for x in result["summaries"]}
    assert by_day[-1, "case"]["estimate"] is None
    assert by_day[-2, "case"]["estimate"] is None
    assert by_day[-3, "case"]["eligible_rows"] == 8
    assert by_day[-3, "standardized_control"]["estimate"] == 3


def test_case_pod_control_standardization_recomputed_in_every_bootstrap(monkeypatch):
    rows, _, _, _ = trajectory_fixture()
    first_cases = rows.case.eq(1) & rows.stay.lt(4)
    rows.loc[first_cases, "event_day"] = 4
    rows.loc[first_cases, "y"] = rows.loc[first_cases, "day"].eq(1).astype(int)
    events = rows.loc[rows.case.eq(1), ["stay", "event_day"]].drop_duplicates()
    values = pd.DataFrame({"binary": rows.pod.ge(3).astype(float).to_numpy()}, index=rows.row_key)

    def concentrate_cases(self, rng):
        counts = np.ones(len(self.stays), dtype=int)
        counts[self.stays.case.eq(1)] = 0
        counts[0] = int(self.stays.case.sum())
        return counts

    monkeypatch.setattr(ev.StayBootstrap, "draw", concentrate_cases)
    result = it.trajectories(rows, values, {"binary": {"kind": "binary", "unknown_codes": []}}, frame="PI72-CLEAN", alignment="event", eligible_events=events)
    control = next(x for x in result["summaries"] if x["day"] == -3 and x["cohort"] == "standardized_control")
    assert control["known_positive_fraction"] == pytest.approx(.5)
    assert control["estimate_uncertainty"]["ci95"] == [0., 0.]
    assert control["estimate_uncertainty"]["valid_draws"] == 1000


def test_missing_control_pod_yields_null_not_reweighted_complete_case_result():
    rows, values, events, definitions = trajectory_fixture()
    keep = ~(rows.case.eq(0) & rows.pod.eq(3))
    rows = rows.loc[keep]
    values = values.loc[rows.row_key]
    result = it.trajectories(rows, values, {"dose": definitions["dose"]}, frame="PI72-CLEAN", alignment="event", eligible_events=events)
    control = next(x for x in result["summaries"] if x["day"] == -3 and x["cohort"] == "standardized_control")
    assert control["estimate"] is None and not control["standardization_defined"]
    assert control["estimate_uncertainty"]["undefined_draws"] == 1000


def test_task_a_trajectory_stops_strictly_before_formal_first_nec():
    rows, values, _, definitions = trajectory_fixture()
    rows.loc[rows.case.eq(1), "event_day"] = 3
    result = it.trajectories(rows, values, {"dose": definitions["dose"]}, frame="A-formal", alignment="surgery")
    case_day3 = next(x for x in result["summaries"] if x["day"] == 3 and x["cohort"] == "case")
    assert case_day3["eligible_rows"] == 0


def test_task_a_auc_is_pooled_before_repeat_mean_not_fold_mean():
    rows = target_rows(repeats=(0,), n_stays=40, pods=(3,))
    rows["fold"] = rows.stay.astype(int) % 5
    # Every fold has perfect AUC, but score offsets create cross-fold inversions.
    p = rows.fold.to_numpy() * .18 + rows.y.to_numpy() * .05 + .05
    contrast = [dict(arm="arm", reference="ref", metric="AUROC_total")]
    result = ev.evaluate(rows, {"arm": prediction_table(rows, p), "ref": prediction_table(rows, .5)},
                         frame="A-formal", stage="screen", contrasts=contrast, family=contrast)
    got = result["arm_intervals"]["arm"]["AUROC_total"]["estimate"]
    assert got == pytest.approx(roc_auc_score(rows.y, p))
    assert got < np.mean([roc_auc_score(part.y, p[part.index]) for _, part in rows.groupby("fold")])


def test_named_reference_cannot_be_dropped_to_shrink_family():
    rows = target_rows(repeats=(1, 2, 3), pods=(3,))
    predictions = family_predictions(rows)
    predictions.pop(ev.MATCHED_W1)
    with pytest.raises(ev.EvaluationError, match="Unknown/missing"):
        ev.evaluate(rows, predictions, frame="PI72-CLEAN", stage="screen")


def test_low_common_bootstrap_count_cannot_promote_even_with_positive_interval():
    result = ev.promotion_rule(.3, [.2, .4], frame="PI72-CLEAN", repeats=[1, 2, 3, 4, 5],
                               stage="confirm", family_complete=True, common_valid_draws=1989)
    assert not result["passes"]
