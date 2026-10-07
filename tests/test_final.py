from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pf_nec import contract as c, features as f, windows, run, trees, gsafe, selectors, config, cli
from test_gsafe import sample, encoding, engine


def test_frozen_seed_and_excluded_arms():
    from pf_nec import evaluate as ev
    expected = int.from_bytes(hashlib.sha256(b"RX-D1-v1|PI72-CLEAN|-1|-1|bootstrap|shared|-1|-1|-1").digest()[:4], "little") % (2**31 - 1)
    assert ev.seed("PI72-CLEAN", "bootstrap") == expected
    assert len(ev.frozen_family("PI72-CLEAN")) == 10
    assert len(ev.frozen_family("A-formal")) == 4
    assert len(c.D5SAFE) == 604
    for recipe in ("PI10", "excluded-source-recipe", "SHADOW5"):
        with pytest.raises(c.ContractError):
            c.feature_columns(recipe)
    with pytest.raises(c.ContractError):
        cli.context("excluded-arm", 1, -1)


def test_strict_midnight_gates_and_prefix_invariance():
    rows, meta = sample()
    rows = f.select_features(rows, ("mechvent", "statscore"))
    values = rows.values.copy()
    future = rows.keys[c.DATE].gt(102)
    values.loc[future] = 54321
    changed = replace(rows, values=values)
    target = rows.keys.loc[~future]
    left = windows.calendar_windows(rows, target, window=3)
    right = windows.calendar_windows(changed, target, window=3)
    np.testing.assert_equal(left.values, right.values)
    assert not rows.gates.surgery_available[rows.keys[c.DATE].eq(100)].any()
    assert rows.values.loc[rows.keys[c.DATE].lt(100), "statscore"].isna().all()
    assert not np.isinf(rows.values.to_numpy()).any()


@pytest.mark.parametrize("model", ["GSAFE-LGB", "T8-D5SAFE-LGB", "A-D5-LGB"])
def test_end_to_end_training_isolation_and_resume(tmp_path, monkeypatch, model):
    frame = "A-formal" if model.startswith("A-") else "PI72-CLEAN"
    repeat, fold = (0, 0) if frame == "A-formal" else (1, -1)
    train = sample(120, frame=frame, repeat=repeat)
    test = sample(30, 200, frame=frame, repeat=repeat)
    if frame == "A-formal":
        train[1]["fold"] = 1
        test[1]["fold"] = 0
    first, calls = engine(tmp_path / "first", train, test, repeat=repeat, frame=frame, fold=fold)
    original = trees.fit_tree
    def checked(*args, **kwargs):
        assert not any(partition == "test" for _, partition in calls)
        return original(*args, **kwargs)
    monkeypatch.setattr(trees, "fit_tree", checked)
    execute = lambda obj: gsafe.gsafe(obj) if model == "GSAFE-LGB" else obj.apply(run.Arm("D5SAFE", frame=frame))
    result = execute(first)
    pred = pd.read_parquet(result["prediction_path"])
    assert len(pred) == len(test[1]) and np.isfinite(pred.probability).all()
    assert pred.row_key.tolist() == test[1].row_key.tolist()
    weights = Path(result["model_path"]).read_bytes()
    attempts = len(first.ledger.records)
    monkeypatch.setattr(trees, "fit_tree", original)
    assert execute(first) == result
    assert len(first.ledger.records) == attempts
    from explore.t3 import gsafe_t3
    packets = list(gsafe_t3.packets(first, result))
    assert sum(len(packet['rows']) for packet in packets) == len(pred)
    assert all(packet['phi'].shape[1] == (629 if model == 'GSAFE-LGB' else 604) for packet in packets)
    altered = replace(test[0], values=test[0].values * -15)
    metadata = test[1].copy()
    mask = metadata.case.eq(1) & metadata.pod.ge(0)
    metadata.loc[mask, "y"] = 1 - metadata.loc[mask, "y"]
    second, _ = engine(tmp_path / "second", train, (altered, metadata), repeat=repeat, frame=frame, fold=fold)
    other = execute(second)
    assert Path(other["model_path"]).read_bytes() == weights


def test_paths_never_write_data_or_unconfigured_output(tmp_path):
    with pytest.raises(ValueError):
        config.writable(config.DATA_ROOT / "derived.json")
    with pytest.raises(ValueError):
        config.writable(tmp_path.parent / "outside.json")
    assert config.writable(config.RUN_ROOT / "safe.json").is_relative_to(tmp_path)


def test_unknown_or_nonfinite_predictors_fail_closed():
    rows, _ = sample()
    with pytest.raises(c.ContractError):
        c.assert_predictors(["future_nec"], bank=c.D5_BANK)
    values = rows.values.copy()
    values.iloc[0, 0] = np.inf
    with pytest.raises(c.ContractError):
        trees.matrix(replace(rows, values=values), window=1, representation="R1")


def test_frozen_harness_v3_synthetic_evaluation():
    from pf_nec.harness import harness_v3 as hv
    rng = np.random.default_rng(921)
    y = np.tile([0, 0, 1, 0], 30)
    frame = pd.DataFrame({c.H: np.arange(len(y)) // 4, 'y3': y,
        'post_surg': np.tile([0, 1, 1, 1], 30), 'pod_raw': np.tile([-1, 0, 3, 5], 30),
        'doa': np.tile([0, 1, 4, 6], 30), 'group': np.where(np.arange(len(y)) % 3, 'a', 'b'),
        'lead': np.tile([5, 4, 3, 2], 30)})
    probability = np.column_stack([rng.uniform(.05, .8, len(y)), rng.uniform(.05, .8, len(y))])
    score = hv.evaluate_A(frame, probability, b=8)
    assert score['n_rows'] == len(frame) and score['n_cases'] == 30
    assert np.isfinite(score['point']['calib_slope'])
    assert 0 <= score['point']['auc'] <= 1
