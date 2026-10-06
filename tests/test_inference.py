import json
import numpy as np
import pytest
from pf_nec import cli, evaluate as ev
from pf_nec.inference import single_arm_intervals
from test_evaluator import target_rows, prediction_table


@pytest.mark.parametrize("frame", ["PI72-CLEAN", "A-formal"])
def test_single_arm_matches_frozen_paired_arm_intervals(frame):
    repeats = (0, 1, 2) if frame == "A-formal" else (1, 2, 3)
    rows = target_rows(repeats=repeats, n_stays=30)
    if frame == "A-formal":
        rows["fold"] = ((rows.stay - .125).astype(int) % 5)
        rows.loc[rows.pod.eq(-1) & rows.case.eq(1), "y"] = 1
    rng = np.random.default_rng(91)
    predictions = {"reference": prediction_table(rows, rng.uniform(.01, .99, len(rows))),
                   "other": prediction_table(rows, .5)}
    family = [dict(arm="reference", reference="other", metric="AUROC_total")]
    paired = ev.evaluate(rows, predictions, frame=frame, stage="screen", contrasts=family, family=family)
    result = single_arm_intervals(rows, predictions["reference"], frame=frame, model="reference")
    assert result["arm_intervals"] == paired["arm_intervals"]["reference"]
    assert result["attempted_draws"] == 2000 and result["promotion_assessed"] is False
    assert paired["family_complete"] is False
    assert not paired["comparisons"][0]["promotion"]["passes"]
    if frame == "PI72-CLEAN":
        assert result["arm_intervals"]["AUROC_preop_within_phase"]["percentile_ci95"] is None
    json.dumps(result, allow_nan=False)


def test_single_arm_rejects_missing_rows_and_nonfinite_probability():
    rows = target_rows(repeats=(1,), n_stays=10)
    pred = prediction_table(rows, .5)
    for bad in (pred.iloc[:-1], pred.assign(probability=np.nan)):
        with pytest.raises(ev.EvaluationError):
            single_arm_intervals(rows, bad, frame="PI72-CLEAN", model="reference")


def test_evaluate_cli_descriptive_flag_and_json(tmp_path, monkeypatch, capsys):
    rows = target_rows(repeats=(1,), n_stays=10)
    predictions = prediction_table(rows, .1 + .7 * rows.y)
    monkeypatch.setattr(cli, "load_evaluation_inputs", lambda model, repeats: (rows, predictions))
    monkeypatch.setattr(cli.config, "RUN_ROOT", tmp_path)
    cli.main(["evaluate", "--model", "GSAFE-LGB", "--repeats", "1", "--descriptive-ci"])
    result = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "GSAFE-LGB/evaluation.json").read_text())
    assert saved == result
    assert result["descriptive_ci"]["arm_intervals"]["AUROC_post"]["estimate"] == 1.
