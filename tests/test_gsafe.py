from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from pf_nec import contract as c, features as f, run, selectors as sel, gsafe as hook


def sample(n=24, start=0, frame="PI72-CLEAN", repeat=1):
    days = 6
    stay = np.repeat(np.arange(start, start + n, dtype=np.float64) + 120000.25, days)
    day = np.tile(np.arange(99, 105), n)
    identity = c.SOURCE_ROW if frame == "PI72-CLEAN" else c.HARNESS_ROW
    keys = pd.DataFrame({c.H: stay, c.DATE: day, identity: np.arange(start * days, (start + n) * days)})
    case = np.repeat(np.arange(start, start + n) % 2, days)
    y = ((day == 104) & (case == 1)).astype(np.int8)
    rng = np.random.default_rng(start + 143)
    values = pd.DataFrame(rng.normal(size=(len(day), 604)).astype(np.float32), columns=c.D5SAFE,
                          index=pd.Index(keys[identity], name=identity))
    values.loc[:, "mechvent"] = (day % 2).astype(np.float32)
    values.iloc[::11, 8] = np.nan
    anchors = pd.DataFrame({c.H: np.unique(stay), "adm": 98., "surg": 100., "icu_serial": 101.})
    rows = f.build_features(keys, values, anchors, bank=c.D5_BANK, frame=frame)
    meta = pd.DataFrame({"repeat": repeat, "row_key": keys[identity], "stay": stay,
                         "y": y, "pod": day - 100, "case": case})
    if frame == "A-formal":
        meta["fold"] = np.repeat(np.arange(start, start + n) % 5, days)
    return rows, meta


def encoding(rows):
    keys = rows.keys
    raw = pd.DataFrame({field: [(str(int(v) % 4),) for v in keys[c.H]] for field in hook.RAW_FIELDS})
    raw["_stay"] = keys[c.H].astype(str)
    raw["_admitted"] = True
    raw["_surgical"] = rows.gates.surgery_available.to_numpy()
    for field in hook.RAW_FIELDS[1:]:
        raw[field] = [v if ok else () for v, ok in zip(raw[field], raw._surgical)]
    support = pd.DataFrame(np.zeros((len(keys), 15), np.float32), columns=hook.S_COLUMNS)
    support.iloc[:, 0] = keys[c.DATE].mod(2)
    return hook.EncodingInputs(keys, raw, keys[c.H].mod(3), support)


def test_crossfit_held_labels_prior_counts_and_dependency_closure(tmp_path):
    rows, meta = sample()
    obj, _ = engine(tmp_path / "e", (rows, meta), sample(6, 100))
    inputs = encoding(rows)
    original, state, states = hook._crossfit(inputs, meta.y.to_numpy(), obj, obj.context, tmp_path / "reference")
    held_ids = np.sort(rows.keys[c.H].unique())[next(hook.KFold(3, shuffle=True,
                         random_state=obj.context.seed("model", hook.ARM)).split(np.arange(24)))[1]]
    held = rows.keys[c.H].isin(held_ids).to_numpy()
    changed_y = meta.y.to_numpy().copy()
    changed_y[held] = 1 - changed_y[held]
    other, _ = engine(tmp_path / "other", (rows, meta), sample(6, 100))
    changed, _, changed_states = hook._crossfit(inputs, changed_y, other, other.context, tmp_path / "changed")
    np.testing.assert_array_equal(original[held], changed[held])
    for name in ("prior", "rates", "site_rates", "binding"):
        assert states[0][name] == changed_states[0][name]
    assert state["fit_stays"] == 24
    assert len(obj.ledger.records) == 4
    assert all(record["component"] == "supervised_encoding" for record in obj.ledger.records)
    poisoned = replace(inputs, raw=inputs.raw.assign(necbelldtshift=900))
    with pytest.raises(c.ContractError, match="label-free"):
        hook._fit_state(poisoned, meta.y)
    bad = inputs.raw.copy()
    bad.at[3, "preop_risk"] = ("330",)
    with pytest.raises(c.ContractError, match="risk code"):
        hook._transform(replace(inputs, raw=bad), state)


def test_encoding_unseen_site_gates_and_future_invariance(tmp_path):
    training, meta = sample()
    inputs = encoding(training)
    state = hook._fit_state(inputs, meta.y)
    held, _ = sample(4, 100)
    apply = encoding(held)
    values = hook._transform(replace(apply, site=pd.Series(np.nan, index=range(len(held.keys)))), state)
    np.testing.assert_array_equal(values[:, -1], np.float32(state["prior"]))
    assert np.isnan(values[~held.gates.surgery_available, 1:9]).all()
    changed = apply.raw.copy()
    future = held.keys[c.DATE].gt(102)
    for field in hook.RAW_FIELDS:
        changed.loc[future, field] = pd.Series([("new",)] * int(future.sum()), index=changed.index[future])
    after = hook._transform(replace(apply, raw=changed), state)
    before = hook._transform(apply, state)
    np.testing.assert_array_equal(before[~future], after[~future])
    # Hidden raw surgery codes cannot reach a count/rate or count feature.
    poisoned = apply.raw.copy()
    poisoned.loc[~poisoned._surgical, "primary_proc"] = pd.Series(
        [("hidden",)] * int((~poisoned._surgical).sum()), index=poisoned.index[~poisoned._surgical])
    with pytest.raises(c.ContractError, match="Unavailable raw codes"):
        hook._transform(replace(apply, raw=poisoned), state)


def test_inner_validation_outcomes_are_not_encoder_inputs(tmp_path):
    rows, meta = sample()
    obj, _ = engine(tmp_path, (rows, meta), sample(6, 100))
    fit, validation = next(sel.inner_folds(rows.keys, meta.y.to_numpy(), obj.context))
    inputs = encoding(rows)
    state = hook._fit_state(inputs.take(fit), meta.y.iloc[fit])
    changed_labels = meta.y.copy()
    changed_labels.iloc[validation] = 1 - changed_labels.iloc[validation]
    other = hook._fit_state(inputs.take(fit), changed_labels.iloc[fit])
    assert state == other
    np.testing.assert_array_equal(hook._transform(inputs.take(validation), state),
                                  hook._transform(inputs.take(validation), other))
    # Official validation y is allowed to affect LGB stopping by the frozen
    # contract. Changing a projected-away copy must not alter stopping inputs.
    projected = inputs.take(validation)
    with pytest.raises(c.ContractError, match="label-free"):
        hook._transform(replace(projected, raw=projected.raw.assign(y=changed_labels.iloc[validation].to_numpy())), state)

def engine(path, training, testing, *, repeat=1, frame="PI72-CLEAN", fold=-1):
    calls = []
    def loader(recipe, partition):
        assert recipe == "D5SAFE"
        calls.append((recipe, partition))
        return training[0] if partition == "train" else testing[0]
    def metadata(partition):
        calls.append(("metadata", partition))
        return (training[1] if partition == "train" else testing[1]).assign(repeat=repeat)
    result = run.TreeEngine(sel.FitContext(frame, repeat, fold), loader, metadata, path)
    result.i2b_cache = path / "private"
    result.gsafe_inputs = encoding
    return result, calls
