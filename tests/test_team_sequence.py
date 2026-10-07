import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from pf_nec import contract as c, evaluate as ev, windows
from explore.team import sequence as s
from test_team import rows_fixture, extract, seq_config as config


@pytest.mark.parametrize("learner", ("TCN", "GRU"))
@pytest.mark.parametrize("training", (False, True))
def test_future_perturbation_with_controlled_dropout_rng(learner, training):
    torch.manual_seed(101)
    model = s.make_model(learner, 21, .2).train(training)
    x = torch.randn(3, 14, 21)
    future = x.clone()
    future[:, 7:] = torch.randn_like(future[:, 7:]) * 100
    torch.manual_seed(703)
    past = model.sequence_logits(x)
    torch.manual_seed(703)
    perturbed = model.sequence_logits(future)
    torch.testing.assert_close(past[:, :7], perturbed[:, :7], rtol=0, atol=0)
    assert not torch.allclose(past[:, 7:], perturbed[:, 7:])


@pytest.mark.parametrize("learner", ("TCN", "GRU"))
def test_last_target_gradient_cannot_see_future(learner):
    model = s.make_model(learner, 21, .2).eval()
    x = torch.randn(2, 14, 21, requires_grad=True)
    model.sequence_logits(x)[:, 6].sum().backward()
    assert (x.grad[:, 7:] == 0).all()
    assert x.grad[:, :7].abs().sum() > 0
    torch.testing.assert_close(model(x[:, :7]), model.sequence_logits(x)[:, 6], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("learner", s.LEARNERS)
@pytest.mark.parametrize("training", (False, True))
def test_other_stays_do_not_change_window_output(learner, training):
    model = s.make_model(learner, 21, .3).train(training)
    x = torch.randn(4, 7, 21)
    x[..., -1] = 1
    other = x.clone()
    other[1:] *= 400
    # Another stay's shorter observed history is represented INSIDE its own
    # fixed-width window, never via whole-stay batch padding or length inference.
    other[1, :4, :10] = 0
    other[1, :4, 10:20] = 1
    other[1, :4, -1] = 0
    torch.manual_seed(81)
    expected = model(x)[0]
    torch.manual_seed(81)
    actual = model(other)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not training:
        torch.testing.assert_close(model(x[:1])[0], expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("learner", ("TCN", "GRU"))
def test_other_window_length_and_appended_future_do_not_change_prefix_eval(learner):
    model = s.make_model(learner, 21, .3).eval()
    x = torch.randn(1, 7, 21)
    long = torch.cat((x, torch.randn(1, 7, 21) * 100), dim=1)
    expected = model(x)
    torch.testing.assert_close(model.sequence_logits(long)[:, 6], expected, rtol=1e-5, atol=2e-6)
    # A different length in a separate call cannot affect this window's state.
    model(torch.randn(9, 3, 21))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_orderless_history_permutation_and_no_history():
    model = s.make_model("ORDERLESS_MLP", 59, .2).eval()
    x = torch.randn(4, 7, 59)
    x[..., -1] = torch.tensor([1, 0, 1, 1, 0, 1, 1])
    reordered = x.clone()
    reordered[:, :-1] = x[:, [5, 3, 0, 2, 1, 4]]
    torch.testing.assert_close(model(x), model(reordered), rtol=1e-5, atol=2e-6)
    changed_padding = x.clone()
    changed_padding[:, [1, 4], :-1] = 10000
    torch.testing.assert_close(model(x), model(changed_padding), rtol=0, atol=0)
    x[:, :-1, -1] = 0
    captured = []
    handle = model.head.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    model(x)
    handle.remove()
    assert (captured[0][:, :32] == 0).all() and (captured[0][:, -1] == 0).all()


def test_unique_source_preprocessing_not_window_frequency_and_all_missing():
    rows, _, _ = rows_fixture()
    prep = s.Preprocessor.fit(rows.values.to_numpy())
    z = np.sign(rows.values.to_numpy()) * np.log1p(np.abs(rows.values.to_numpy()))
    for j in range(z.shape[1]):
        good = np.isfinite(z[:, j])
        median = np.median(z[good, j]) if good.any() else 0
        filled = np.where(good, z[:, j], median)
        assert prep.median[j] == pytest.approx(median)
        assert prep.mean[j] == pytest.approx(filled.mean(), abs=1e-7)
        assert prep.scale[j] == pytest.approx(filled.std() or 1, abs=1e-7)
    assert (prep.median[-1], prep.mean[-1], prep.scale[-1]) == (0, 0, 1)
    transformed = prep.transform(rows.values.to_numpy())
    assert np.isfinite(transformed).all() and transformed.dtype == np.float32
    assert (abs(transformed) <= 6).all()
    for length in (3, 7, 14):
        batch = windows.calendar_windows(rows, window=length)
        raw = windows.r4_tensor(batch)
        actual = prep.r4(raw)
        expected = windows.r4_tensor(batch, transform=prep.transform)
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
        assert (actual[~batch.row_observed, :len(rows.values.columns)] == 0).all()
        assert actual.shape[-1] == 101  # constant k masks, even all-finite apply inputs
        np.testing.assert_array_equal(actual[:, -1], prep.r4(windows.r4_tensor(
            windows.calendar_windows(rows, window=1)))[:, -1])
    with pytest.raises(s.SequenceError):
        s.Preprocessor.fit(np.array([[np.inf]]))
    with pytest.raises(s.SequenceError):
        s.Preprocessor.fit(np.empty((0, 10)))


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_calendar_future_rows_and_frozen_preprocessing_leave_past_predictions(learner):
    rows, _, _ = rows_fixture()
    prep = s.Preprocessor.fit(rows.values.to_numpy())
    changed = copy.deepcopy(rows)
    future = rows.keys[c.DATE] > 101
    changed.values.loc[future, :] = changed.values.loc[future, :] * 1e4
    original_x = prep.r4(windows.r4_tensor(windows.calendar_windows(rows, window=7)))
    changed_x = prep.r4(windows.r4_tensor(windows.calendar_windows(changed, window=7)))
    past = ~future.to_numpy()
    np.testing.assert_array_equal(original_x[past], changed_x[past])
    for training in (True, False):
        model = s.make_model(learner, 101, .2).train(training)
        torch.manual_seed(62)
        expected = model(torch.from_numpy(original_x))
        torch.manual_seed(62)
        actual = model(torch.from_numpy(changed_x))
        torch.testing.assert_close(actual[past], expected[past], rtol=0, atol=0)


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_seed_matches_evaluator_and_alias_identity(learner):
    cfg = config(learner=learner, recipe="GAIN50", stage="inner")
    for init in range(3):
        assert s.network_seed(cfg, init) == ev.seed(cfg["frame"], "network", repeat=1,
            fold=-1, arm=cfg["arm"], inner=0, trial=-1, init=init)
    bad = dict(cfg, arm="profile-slot-1")
    with pytest.raises(s.SequenceError, match="canonical"):
        s._validate_config(bad, require_features=False)


def test_epoch_summary_nine_epochs_fold_mean_and_rejection():
    receipts = []
    for fold, epochs in enumerate(([1, 2, 3], [4, 5, 6], [7, 8, 9])):
        cfg = config(stage="inner")
        cfg["inner_fold"] = fold
        receipts.append({"status": "completed", "config": cfg, "validation_auc": .6 + .1 * fold,
                         "members": [{"init": i, "best_epoch": e} for i, e in enumerate(epochs)]})
    result = s.summarize_inner(receipts)
    assert result["chosen_epochs"] == 5 and result["best_epochs"] == list(range(1, 10))
    assert result["mean_inner_auc"] == pytest.approx(.7)
    bad = copy.deepcopy(receipts)
    bad[0]["status"] = "synthetic_smoke"
    with pytest.raises(s.SequenceError, match="smoke"):
        s.summarize_inner(bad)
    bad = copy.deepcopy(receipts)
    bad[1]["config"]["inner_fold"] = 0
    with pytest.raises(s.SequenceError, match="folds"):
        s.summarize_inner(bad)
    bad = copy.deepcopy(receipts)
    bad[1]["config"]["arm"] = "SEQ-PI29-w7-TCN"
    with pytest.raises(s.SequenceError, match="contexts"):
        s.summarize_inner(bad)


@pytest.mark.parametrize("failure", ["overlap", "single_class", "post_single_class", "label_coverage", "label_vector", "extra_config", "refit_validation"])
def test_fit_package_boundaries_fail_closed(tmp_path, failure):
    train, y, _ = rows_fixture()
    val, vy, post = rows_fixture(offset=100)
    cfg = config(stage="inner")
    if failure == "overlap":
        val, vy, post = rows_fixture()
    elif failure == "single_class":
        vy[:] = 0
    elif failure == "post_single_class":
        post = vy == 0
    elif failure == "label_coverage":
        y = y.iloc[:-1]
    elif failure == "label_vector":
        y = y.to_numpy()
    elif failure == "extra_config":
        cfg["lead"] = 3
    elif failure == "refit_validation":
        cfg = config(stage="refit")
    with pytest.raises(s.SequenceError):
        s.build_fit_package(tmp_path / "bad.zip", train, y, cfg,
                             validation_rows=val, validation_labels=vy, validation_score_mask=post)


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_cpu_micro_smoke_fit_apply_ensemble_and_future_invariance(tmp_path, learner):
    recipe = "GAIN50"
    train, y, _ = rows_fixture(recipe=recipe)
    val, vy, post = rows_fixture(recipe=recipe, offset=100)
    cfg = config(stage="inner", learner=learner, recipe=recipe)
    s.build_fit_package(tmp_path / "inner.zip", train, y.sample(frac=1, random_state=1), cfg,
                         validation_rows=val, validation_labels=vy, validation_score_mask=post)
    inner_dir = extract(tmp_path / "inner.zip", tmp_path / "inner")
    receipt = s.fit_package(inner_dir, tmp_path / "inner_models", device="cpu", synthetic_smoke=True)
    assert receipt["status"] == "synthetic_smoke" and len(receipt["members"]) == 3
    assert all(m["epochs_ran"] == 2 and m["parameters"] < 100000 for m in receipt["members"])
    assert np.isfinite(receipt["validation_auc"]) and np.isfinite(receipt["AUROC_total"])
    cfg = config(stage="refit", learner=learner, recipe=recipe)
    s.build_fit_package(tmp_path / "refit.zip", train, y, cfg)
    refit_dir = extract(tmp_path / "refit.zip", tmp_path / "refit")
    fit = s.fit_package(refit_dir, tmp_path / "models", device="cpu")  # exactly chosen 2 epochs x 3 init
    assert fit["status"] == "completed" and all(m["best_epoch"] == 2 for m in fit["members"])
    prep = s.Preprocessor.fit(train.values.to_numpy())
    assert fit["preprocessing"] == prep.state()  # cannot see validation values
    app_cfg = config(learner=learner, recipe=recipe)
    s.build_apply_package(tmp_path / "apply.zip", val, app_cfg)
    app_dir = extract(tmp_path / "apply.zip", tmp_path / "apply")
    p = s.apply_package(app_dir, tmp_path / "models", tmp_path / "predictions.npz", device="cpu")
    _, arrays = s.load_package(app_dir)
    individual = []
    for init in range(3):
        saved = torch.load(tmp_path / "models" / f"init{init}.pt", weights_only=True)
        model = s.make_model(learner, arrays["X"].shape[-1], .5)
        model.load_state_dict(saved["state_dict"])
        individual.append(s._predict(model, arrays["X"], prep, torch.device("cpu")))
    np.testing.assert_array_equal(p, np.mean(individual, axis=0))
    changed = copy.deepcopy(val)
    future = changed.keys[c.DATE] > 101
    changed.values.loc[future] *= 1000
    s.build_apply_package(tmp_path / "changed.zip", changed, app_cfg)
    changed_dir = extract(tmp_path / "changed.zip", tmp_path / "changed")
    changed_p = s.apply_package(changed_dir, tmp_path / "models", tmp_path / "changed.npz", device="cpu")
    np.testing.assert_array_equal(changed_p[~future], p[~future])
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as result:
        assert set(result.files) == {"row_key", "stay", "actual_date", "repeat", "probability"}
        np.testing.assert_array_equal(result["row_key"], val.keys[c.SOURCE_ROW])
        np.testing.assert_array_equal(result["stay"], val.keys[c.H])
    # Apply data cannot affect weights/stats/seeds: hash every final checkpoint.
    hashes = [s.file_sha256(tmp_path / "models" / f"init{i}.pt") for i in range(3)]
    s.apply_package(changed_dir, tmp_path / "models", tmp_path / "changed2.npz", device="cpu")
    assert hashes == [s.file_sha256(tmp_path / "models" / f"init{i}.pt") for i in range(3)]
    # Stored feature order mismatch cannot silently score another arm.
    bad_config = json.loads((app_dir / "config.json").read_text())
    bad_config["features"] = bad_config["features"][::-1]
    (app_dir / "config.json").write_text(json.dumps(bad_config))
    with pytest.raises(s.SequenceError, match="context differs"):
        s.apply_package(app_dir, tmp_path / "models", tmp_path / "bad.npz", device="cpu")


def test_standalone_runner_numpy_torch_only_cpu(tmp_path):
    train, y, _ = rows_fixture(frame="A-formal", recipe="GAIN50", strings=True)
    val, _, _ = rows_fixture(frame="A-formal", recipe="GAIN50", strings=True, offset=100)
    cfg = config(frame="A-formal", recipe="GAIN50", stage="refit", learner="GRU")
    s.build_fit_package(tmp_path / "fit.zip", train, y, cfg)
    fit_dir = extract(tmp_path / "fit.zip", tmp_path / "fit")
    s.build_apply_package(tmp_path / "apply.zip", val, config(frame="A-formal", recipe="GAIN50", learner="GRU"))
    apply_dir = extract(tmp_path / "apply.zip", tmp_path / "apply")
    # Simulate absent Windows packages using an import hook before runpy execution.
    bootstrap = '''import sys,runpy
# A missing optional dependency must return None from find_spec (torch probes
# optional packages); sys.modules[name]=None also prevents actual import.
sys.modules.update({name:None for name in
 {'pandas','pyarrow','sklearn','xgboost','lightgbm','scipy','threadpoolctl'}})
script=sys.argv.pop(1)
sys.path.insert(0, __import__('os').path.dirname(script))
runpy.run_path(script,run_name='__main__')
'''
    for command in ([str(fit_dir / "sequence.py"), "fit", "--package", str(fit_dir),
                     "--output", str(tmp_path / "models"), "--device", "cpu"],
                    [str(apply_dir / "sequence.py"), "apply", "--package", str(apply_dir),
                     "--models", str(tmp_path / "models"), "--output", str(tmp_path / "p.npz"), "--device", "cpu"]):
        result = subprocess.run([sys.executable, "-B", "-c", bootstrap, *command], capture_output=True, text=True,
                                 cwd=tmp_path, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "completed"
    with np.load(tmp_path / "p.npz", allow_pickle=False) as predictions:
        assert set(predictions.files) == {"row_key", "stay", "actual_date", "repeat", "fold", "probability"}
        assert (predictions["fold"] == 2).all()
        np.testing.assert_array_equal(predictions["stay"], val.keys[c.H])


def test_join_predictions_exact_tree_keys_without_inner_join(tmp_path):
    targets = pd.DataFrame({"repeat": [1, 1, 1], "row_key": [4, 8, 9],
                            "stay": [123456789012.25, 123456789012.25, 987654321098.5],
                            "actual_date": [100, 101, 103], "y": [0, 1, 0],
                            "pod": [0, 1, 3], "case": [1, 1, 0]})
    predictions = targets[["repeat", "row_key", "stay", "actual_date"]].assign(probability=[.2, .8, .3])
    shuffled = predictions.iloc[[2, 0, 1]]
    np.savez(tmp_path / "p.npz", **{k: shuffled[k].to_numpy() for k in shuffled})
    actual = s.join_predictions(tmp_path / "p.npz", targets)
    pd.testing.assert_frame_equal(actual, targets.assign(probability=[.2, .8, .3]))
    ev.align_predictions(targets, {"SEQ-PI10-w7-TCN": actual}, frame="PI72-CLEAN")
    for bad in (predictions.iloc[:-1], pd.concat([predictions, predictions.iloc[[0]]]),
                predictions.assign(stay=predictions.stay + .125), predictions.assign(probability=np.nan)):
        np.savez(tmp_path / "bad.npz", **{k: bad[k].to_numpy() for k in bad})
        with pytest.raises(s.SequenceError):
            s.join_predictions(tmp_path / "bad.npz", targets)
