"""Synthetic integration and fixture contracts for the new team interfaces."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import torch
import numpy as np
import pandas as pd
import pytest
from pf_nec import contract as c, features as f, selectors as sel, windows
from explore.team import inputs, sequence as s, sequence_m1 as m
from test_gsafe import sample


def rows_fixture(*, recipe="GAIN50", frame="PI72-CLEAN", offset=0, strings=False):
    columns = c.D5SAFE if recipe == "D5SAFE" else c.D5SAFE[:50]
    identity = c.SOURCE_ROW if frame == "PI72-CLEAN" else c.HARNESS_ROW
    stays = [f"stay-{offset+j:03d}" if strings else 123456789012.25 + offset + j/4 for j in range(6)]
    dates = [99, 100, 101, 103]
    keys = pd.DataFrame({c.H: np.repeat(stays, 4), c.DATE: dates * 6,
                        identity: np.arange(offset*100, offset*100+24, dtype=np.int64)})
    values = pd.DataFrame(np.random.default_rng(53+offset).normal(size=(24, len(columns))).astype(np.float32),
                          columns=columns, index=pd.Index(keys[identity], name=identity))
    values.iloc[1] = np.nan
    values.iloc[4, 0] = np.nan
    values.iloc[:, -1] = np.nan
    anchors = pd.DataFrame({c.H: stays, "adm": 99., "surg": 100., "icu_serial": 100.5})
    rows = f.build_features(keys, values, anchors, bank=c.D5_BANK, frame=frame)
    y = pd.Series([int(j % 2 == 0 and d == 101) for j in range(6) for d in dates], index=keys[identity])
    post = pd.Series(keys[c.DATE].to_numpy() >= 100, index=keys[identity])
    return rows, y, post


def seq_config(*, stage="apply", learner="TCN", recipe="GAIN50", frame="PI72-CLEAN", **kwargs):
    return s.make_config(frame=frame, repeat=1 if frame == "PI72-CLEAN" else 0,
                         fold=-1 if frame == "PI72-CLEAN" else 2, recipe=recipe,
                         window=7, learner=learner, stage=stage,
                         inner_fold=0 if stage == "inner" else -1,
                         chosen_epochs=2 if stage == "refit" else None, synthetic=True, **kwargs)


def m1_config(learner="GRUD-WINDOW", stage="apply", recipe="GAIN50", window=7):
    return m.make_config(frame="PI72-CLEAN", repeat=4, recipe=recipe, window=window,
                         learner=learner, stage=stage, synthetic=True,
                         inner_fold=0 if stage == "inner" else -1,
                         chosen_epochs=2 if stage == "refit" else None)


def m1_configured(learner="GRUD-WINDOW", k=50):
    cfg = m1_config(learner)
    cfg["features"] = list(c.D5SAFE[:k])
    return cfg


def build_m1_apply(path, rows, cfg):
    train, _, _ = rows_fixture(recipe=cfg["recipe"])
    pp = m.Preprocessor.fit(train.values.to_numpy(np.float32))
    return m.build_apply_package(path, rows, cfg, preprocessing=pp.state(), fit_pool_sha256="0"*64)


def extract(path, target):
    target.mkdir()
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    return target


def test_new_identity_and_final_spec_separation():
    payload = Path(s.__file__).with_name("spec.json").read_bytes()
    assert s.SPEC_SHA256 == m.ADDENDUM_SHA256 == hashlib.sha256(payload).hexdigest()
    assert s.SPEC_SHA256 != c.SPEC_SHA256
    assert m.BASE_RUNNER_SHA256 == s.file_sha256(s.__file__)
    assert s.FORMAT == "NX-H4-SEQ-v1" and m.FORMAT == "NX-H4-M1-v1"
    for recipe in ("PI10", "excluded-source-recipe", "SHADOW50"):
        with pytest.raises(s.SequenceError):
            seq_config(recipe=recipe)
        with pytest.raises(m.SequenceError):
            m1_config(recipe=recipe)
    with pytest.raises(m.SequenceError):
        m1_config(recipe="D5SAFE")  # full-bank M1 violates the inherited input/parameter caps
    with pytest.raises(s.SequenceError):
        seq_config(recipe="D5SAFE")  # preserve the frozen full-bank/window restrictions
    assert json.loads(payload)["real_data_gate"]["automatic_real_run_authorization"] is False


@pytest.mark.parametrize("frame", ["PI72-CLEAN", "A-formal"])
def test_safe_bank_packaging_and_all_sequence_learner_shapes(tmp_path, frame):
    recipe = "GAIN50"
    rows, _, _ = rows_fixture(frame=frame, recipe=recipe)
    cfg = seq_config(frame=frame, recipe=recipe)
    info = s.build_apply_package(tmp_path/"apply.zip", rows, cfg)
    loaded, arrays = s.load_package(extract(info["path"], tmp_path/"apply"))
    assert loaded["features"] == list(rows.values)
    assert set(arrays) == {"X", "availability", "row_key", "stay", "actual_date"}
    pp = s.Preprocessor.fit(rows.values.to_numpy())
    tensor = torch.from_numpy(pp.r4(arrays["X"]))
    for learner in s.LEARNERS:
        model = s.make_model(learner, tensor.shape[-1], .2).eval()
        assert model(tensor).shape == (24,)
        assert torch.isfinite(model(tensor)).all()
        assert s.parameter_count(model) < s.PARAMETER_CAP
    with pytest.raises((s.SequenceError, c.ContractError)):
        s._pack_rows(replace(rows, bank=c.SOURCE_BANK), cfg)


@pytest.mark.parametrize("learner", m.LEARNERS)
def test_m1_masks_calendar_gap_and_exact_library_model(tmp_path, learner):
    rows, y, _ = rows_fixture()
    cfg = m1_config(learner, "refit")
    info = m.build_fit_package(tmp_path/"fit.zip", rows, y, cfg)
    loaded, arrays = m.b.load_package(extract(info["path"], tmp_path/"fit"))
    pp = m.Preprocessor.from_state(loaded["preprocessing"])
    expected = windows.tree_matrix(windows.calendar_windows(rows, window=7), "R1").values if learner == "TABM-R1" else windows.r4_tensor(windows.calendar_windows(rows, window=7))
    np.testing.assert_array_equal(arrays["train_X"], pp.r4(expected))
    assert np.isfinite(arrays["train_X"]).all()
    model = m.make_model(loaded, .2, pp).eval()
    shape = (24, 16) if learner == "TABM-R1" else (24,)
    assert model(torch.from_numpy(arrays["train_X"])).shape == shape
    assert m.parameter_count(model) <= m.CAPS[learner]


def test_fitted_gain_rank_and_downstream_training_on_synthetic_data(tmp_path):
    train, meta = sample(120)
    # A known synthetic baseline signal and enough rows for the frozen leaf size.
    train.values.loc[:, "gestagewks"] = 30 + 6 * meta.case.to_numpy()
    labels = pd.Series(meta.y.to_numpy(), index=meta.row_key)
    rows, selection = inputs.gain50(train, labels, sel.FitContext(), checkpoint_directory=tmp_path/"rank")
    assert len(rows.values.columns) == 50 and rows.bank == c.D5_BANK
    validation, vmeta = sample(12, 100)
    held = inputs.selected_rows(validation, selection)
    changed = copy.deepcopy(validation)
    changed.values.iloc[:] *= -100
    other = inputs.selected_rows(changed, selection)
    assert list(other.values) == list(held.values) == selection["columns"]
    assert selection["fit_binding"] == sel.pool_binding(*sel.sorted_rows(train, labels))
    cfg = seq_config(stage="refit", learner="GRU")
    info = s.build_fit_package(tmp_path/"fit.zip", rows, labels, cfg)
    s.fit_package(extract(info["path"], tmp_path/"fit"), tmp_path/"models", device="cpu", synthetic_smoke=True)
    app = s.build_apply_package(tmp_path/"apply.zip", held, seq_config(learner="GRU"))
    probability = s.apply_package(extract(app["path"], tmp_path/"apply"), tmp_path/"models", tmp_path/"predictions.npz", device="cpu")
    assert probability.shape == (len(vmeta),) and np.isfinite(probability).all()
    with pytest.raises(c.ContractError):
        inputs.gain50(rows, labels, sel.FitContext())
