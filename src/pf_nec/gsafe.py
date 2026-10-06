"""Frozen G-safe fit/encoding hook, extracted without historical report drivers."""
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from . import contract as c, evaluate as ev, features as f, run, selectors as sel, trees, windows, config
PRIVATE = config.CACHE_ROOT

def portable(path):
    return str(Path(path).resolve())

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else config.CACHE_ROOT / p


VERSION = "RX-I2b-v1.1"


ARM = "GSAFE-LGB"


G_COLUMNS = tuple(c.SPEC["feature_sets"]["GSAFE"]["columns"])


EXTRA = tuple(c.SPEC["feature_sets"]["GSAFE"]["extension_columns"])


R_COLUMNS, S_COLUMNS = EXTRA[:9], EXTRA[9:24]


RAW_FIELDS = ("fund", "primary_proc", "primary_surgdiag", "procedures", "preop_risk")


def _json(path):
    return json.loads(resolve(path).read_text())


def _write(path, value):
    def native(item):
        if isinstance(item, dict):
            return {str(k): native(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [native(v) for v in item]
        if isinstance(item, np.generic):
            return native(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    value = native(value)
    if "/i2b_smoke/" in str(Path(path)) and isinstance(value, dict):
        value = {**value, "execution_scope": "train_only_pipeline_probe"}
        if value.get("status") == "completed":
            value["status"] = "smoke_completed"
    run.write_json(path, value)


def _digest(*items):
    h = hashlib.sha256((c.SPEC_SHA256 + VERSION).encode())
    for item in items:
        if isinstance(item, pd.DataFrame):
            h.update(json.dumps(list(item.columns)).encode())
            h.update(pd.util.hash_pandas_object(item, index=False).values.tobytes())
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def _runtime(cache):
    """Legacy imports reset runtime settings; restore them before numerical work."""
    core = c.import_core()
    cache = Path(cache)
    for name in ("tmp", "mpl"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tempfile.tempdir = str(cache / "tmp")
    os.environ["MPLCONFIGDIR"] = str(cache / "mpl")
    c.reset_threads()
    return core


def _cache(engine):
    if hasattr(engine, "i2b_cache"):
        return Path(engine.i2b_cache)
    base = Path(engine.guard.cache) if engine.guard is not None else PRIVATE
    return base / "gsafe_t3" / _digest(portable(engine.directory))[:16]


def _guard(engine):
    if engine.guard:
        engine.guard()


@dataclass(frozen=True)
class EncodingInputs:
    keys: pd.DataFrame
    raw: pd.DataFrame
    site: pd.Series
    support: pd.DataFrame

    def take(self, positions):
        return EncodingInputs(self.keys.iloc[positions].reset_index(drop=True),
                              self.raw.iloc[positions].reset_index(drop=True),
                              self.site.iloc[positions].reset_index(drop=True),
                              self.support.iloc[positions].reset_index(drop=True))


def _check_inputs(inputs):
    c.validate_keys(inputs.keys)
    if set(inputs.raw) != {*RAW_FIELDS, "_stay", "_admitted", "_surgical"}:
        raise c.ContractError("Encoding accepts only the label-free raw-code schema")
    if tuple(inputs.support) != S_COLUMNS or not (len(inputs.keys) == len(inputs.raw) == len(inputs.site) == len(inputs.support)):
        raise c.ContractError("Misaligned G extension inputs")
    for field in RAW_FIELDS:
        if not all(isinstance(v, tuple) and all(isinstance(code, str) for code in v) for v in inputs.raw[field]):
            raise c.ContractError("Raw codes must be explicit string tuples")
        gate = inputs.raw._admitted if field == "fund" else inputs.raw._surgical
        if not gate.isin([True, False]).all() or inputs.raw.loc[~gate.astype(bool), field].map(len).gt(0).any():
            raise c.ContractError("Unavailable raw codes cannot enter encoding counts")
    if any({"320", "330"}.intersection(v) for v in inputs.raw.preop_risk):
        raise c.ContractError("Forbidden NEC risk code in supervised dependency closure")
    if np.isinf(inputs.support.to_numpy(float)).any():
        raise c.ContractError("Infinite support input")
    if inputs.site.groupby(inputs.keys[c.H]).nunique(dropna=False).gt(1).any():
        raise c.ContractError("Site changes within a hospitalization")


def _load_encoding_inputs(rows, cache):
    """Verified deterministic caches only; no old G rates or outcome columns."""
    core = _runtime(cache)
    from .legacy import line_R
    ids = rows.keys[c.SOURCE_ROW].tolist()
    book = core._validated_table("raw_codes.parquet").set_index("raw_stay_key")
    for field in RAW_FIELDS:
        book[field] = book[field].map(tuple)
    aligned = book.loc[rows.keys[c.H].map(line_R._code)].reset_index(drop=True)
    anchors = aligned[["adm", "surg", "icu"]].rename(columns={"icu": "icu_serial"})
    anchors[c.H] = rows.keys[c.H].to_numpy()
    gates, _ = f.availability_gates(rows.keys, anchors.drop_duplicates(c.H)[list(c.ANCHORS)])
    if not gates.equals(rows.gates) or not np.array_equal(aligned.adm, rows.admission):
        raise c.ContractError("G raw-code anchors differ from accepted D5 gates")
    raw = line_R._asof(rows.keys, book)
    # The old helper permits ICU exactly at midnight. The accepted strict
    # historical-day gate also controls which codes may enter rate counts.
    raw["_surgical"] = rows.gates.surgery_available.to_numpy()
    for field in RAW_FIELDS[1:]:
        raw[field] = [codes if ok else () for codes, ok in zip(raw[field], raw._surgical)]
    raw = raw[[*RAW_FIELDS, "_stay", "_admitted", "_surgical"]]
    identity = core._validated_table("rows.parquet", columns=[c.SOURCE_ROW, c.H, c.DATE, "site"],
                                     filters=[(c.SOURCE_ROW, "in", ids)]).set_index(c.SOURCE_ROW).loc[ids]
    if not np.array_equal(identity[c.H], rows.keys[c.H]) or not np.array_equal(identity[c.DATE], rows.keys[c.DATE]):
        raise c.ContractError("G site/input identity mismatch")
    support = core.load_features("S", ids).reset_index(drop=True)
    inputs = EncodingInputs(rows.keys.copy(), raw, identity.site.reset_index(drop=True), support)
    _check_inputs(inputs)
    c.reset_threads()
    return inputs


def _encoding_binding(inputs, y):
    return _digest(inputs.keys, inputs.raw, inputs.site.to_frame("site"), inputs.support,
                   np.asarray(y, int).tolist())


def _fit_state(inputs, y):
    """One count per stay/code; priors use only this encoding fit's labels."""
    from .legacy import line_R
    _check_inputs(inputs)
    y = np.asarray(y)
    if y.shape != (len(inputs.keys),) or not np.isin(y, [0, 1]).all():
        raise c.ContractError("Invalid encoding fit labels")
    target = pd.Series(y).groupby(inputs.keys[c.H].to_numpy()).max()
    raw = inputs.raw.assign(_label=inputs.keys[c.H].map(target).to_numpy())
    prior, rates = line_R._fit_rates(raw)
    sites = pd.DataFrame({"stay": inputs.keys[c.H], "site": inputs.site.map(line_R._code)})
    sites = sites.drop_duplicates("stay").assign(ever=lambda x: x.stay.map(target))
    counts = sites.loc[sites.site.ne("")].groupby("site").ever.agg(["sum", "count"])
    site_rates = ((counts["sum"] + 200 * prior) / (counts["count"] + 200)).to_dict()
    return {"prior": prior, "rates": rates, "site_rates": site_rates,
            "fit_stays": len(target), "target": "max_y_in_current_fit_pool",
            "strength_raw": 100, "strength_site": 200}


def _transform(inputs, state):
    from .legacy import line_R
    _check_inputs(inputs)
    raw = line_R._encode(inputs.raw, state["prior"], state["rates"])
    site = inputs.site.map(line_R._code).map(state["site_rates"]).fillna(state["prior"])
    values = np.column_stack([raw, inputs.support.to_numpy(), site]).astype(np.float32)
    if values.shape != (len(inputs.keys), 25) or np.isinf(values).any() or not np.isfinite(values[:, -1]).all():
        raise c.ContractError("Invalid encoded G629 extension")
    return values


def _fit_checkpoint(path, binding, engine, key, kind, compute):
    """A checkpoint per fitted state; the run owns its single resource receipt."""
    path = Path(path)
    if path.exists():
        saved = _json(path)
        if saved["binding"] != binding or saved["fit_key"] != key:
            raise c.ContractError("GSAFE checkpoint binding changed")
        for output in saved.get("output_paths", []):
            if not resolve(output).is_file():
                raise run.IncompleteJob("GSAFE checkpoint output missing; no silent refit")
        engine.ledger.recover(saved["fit_attempt_id"])
        return saved
    _guard(engine)
    with engine.ledger.fit(key, arm=ARM, kind=kind) as record:
        result = compute()
        result.update(binding=binding, fit_key=key, fit_attempt_id=record["attempt_id"])
        record.update(rows=result.get("rows"), rounds=result.get("rounds"),
                      component=result.get("component", "G629-LGB"))
        _write(path, result)
    _guard(engine)
    return result


def _crossfit(inputs, y, engine, context, directory):
    """Training encodings exclude a whole stay; application uses full fit state."""
    _check_inputs(inputs)
    stays = np.sort(inputs.keys[c.H].unique())
    if len(stays) < 3:
        raise c.ContractError("GSAFE crossfit requires at least three stays")
    encoded = np.full((len(y), 25), np.nan, np.float32)
    states = []
    split = KFold(3, shuffle=True, random_state=context.seed("model", ARM))
    for fold, (fit, held) in enumerate(split.split(stays)):
        a = np.flatnonzero(inputs.keys[c.H].isin(stays[fit]))
        b = np.flatnonzero(inputs.keys[c.H].isin(stays[held]))
        child = inputs.take(a)
        binding = _encoding_binding(child, np.asarray(y)[a])
        key = f"{context.key}/{binding}/{ARM}/encoding/crossfit{fold}"
        state = _fit_checkpoint(Path(directory) / f"encoding{fold}.json", binding, engine, key, "selector",
                                lambda: {**_fit_state(child, np.asarray(y)[a]), "rows": len(a),
                                         "component": "supervised_encoding"})
        encoded[b] = _transform(inputs.take(b), state)
        states.append(state)
    binding = _encoding_binding(inputs, y)
    state = _fit_checkpoint(Path(directory) / "encoding_full.json", binding, engine,
                            f"{context.key}/{binding}/{ARM}/encoding/full", "selector",
                            lambda: {**_fit_state(inputs, y), "rows": len(y), "component": "supervised_encoding"})
    return encoded, state, states


def _g_matrix(rows, extension):
    if tuple(rows.values) != c.D5SAFE or G_COLUMNS != c.D5SAFE + EXTRA:
        raise c.ContractError("GSAFE must be D5-safe604 + R9/S15/site1 in frozen order")
    c.assert_predictors(rows.values, bank=rows.bank, dependencies=rows.dependencies)
    values = np.column_stack([rows.values.to_numpy(np.float32), extension]).astype(np.float32)
    return windows.TreeMatrix(rows.keys.copy(), values, G_COLUMNS, G_COLUMNS, ())


def gsafe(engine):
    """Three inner stop fits then fixed median-round outer fit; never test-score."""
    if engine.context.frame != "PI72-CLEAN":
        raise c.ContractError("GSAFE is a PI reference, not a Task A arm")
    cache = _cache(engine) / "gsafe"
    _runtime(cache)
    rows = engine.base("D5SAFE")
    y = sel.aligned_labels(rows, engine.labels)
    load_inputs = getattr(engine, "gsafe_inputs", lambda data: _load_encoding_inputs(data, cache))
    inputs = load_inputs(rows)
    binding = _digest(sel.pool_binding(rows, y), _encoding_binding(inputs, y))
    rounds = []
    for inner, (fit, valid) in enumerate(sel.inner_folds(rows.keys, y, engine.context)):
        context = replace(engine.context, inner=inner, pool="inner-fit")
        directory = cache / f"inner{inner}"
        child = f.take_rows(rows, rows.keys[c.SOURCE_ROW].iloc[fit])
        held = f.take_rows(rows, rows.keys[c.SOURCE_ROW].iloc[valid])
        extra, state, _ = _crossfit(inputs.take(fit), y[fit], engine, context, directory)
        train, validation = _g_matrix(child, extra), _g_matrix(held, _transform(inputs.take(valid), state))

        def stop_fit():
            model = trees.fit_tree(train, y[fit], learner="LGB", context=context, arm=ARM,
                                   valid=validation, valid_y=y[valid],
                                   valid_post=engine.train_meta.pod.iloc[valid].ge(0), guard=engine.guard)
            return {"rounds": model.rounds, "rows": len(fit), "validation_rows": len(valid)}

        info = _fit_checkpoint(directory / "stop.json", binding, engine,
                               f"{context.key}/{binding}/{ARM}/model", "inner", stop_fit)
        rounds.append(info["rounds"])
        del train, validation, child, held, extra
        gc.collect()
    count = trees.median_rounds(rounds)
    context = replace(engine.context, inner=-1, pool="outer")
    directory = cache / "outer"
    extra, state, _ = _crossfit(inputs, y, engine, context, directory)

    def final_fit():
        train = _g_matrix(rows, extra)
        model = trees.fit_tree(train, y, learner="LGB", context=context, arm=ARM,
                               rounds=count, guard=engine.guard)
        path = directory / "model.txt"
        model.save(path)
        return {"arm": ARM, "learner": "LGB", "rounds": count, "inner_rounds": rounds,
                "columns": list(G_COLUMNS), "selected": list(G_COLUMNS), "parameters": model.parameters,
                "model_path": portable(path), "encoding_path": portable(directory / "encoding_full.json"),
                "rows": len(y), "input_columns": 629, "probe": bool(getattr(engine, "i2b_probe", False)),
                "output_paths": [portable(path)]}

    info_path = directory / "result.json"
    info = _fit_checkpoint(info_path, binding, engine, f"{context.key}/{binding}/{ARM}/model", "outer", final_fit)
    del extra, inputs, rows
    gc.collect()
    # Only after all stopping/state/model checkpoints exist may test be opened.
    test = engine.base("D5SAFE", "test")
    targets = run.context_targets(engine.metadata("test"), engine.context, "test")
    matrix = _g_matrix(test, _transform(load_inputs(test), state))
    model = trees.load_fit(resolve(info["model_path"]), info)
    prediction = targets.assign(probability=model.predict(matrix.values))
    ev.align_predictions(targets, {ARM: prediction}, frame=engine.context.frame)
    path = cache / "predictions.parquet"
    prediction.to_parquet(path, index=False)
    _guard(engine)
    return {"status": "smoke_completed" if getattr(engine, "i2b_probe", False) else "completed",
            "arm": ARM, "rows": len(targets), "columns": 629,
            "rounds": count, "inner_rounds": rounds, "complete_key_coverage": True,
            "model_path": info["model_path"], "fit_info_path": portable(info_path),
            "prediction_path": portable(path), "outer_test_metrics_computed": False,
            "output_paths": [portable(path), portable(info_path), info["model_path"]]}
