"""Frozen fit contexts, seeds, folds and optional GAIN ranking."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from . import contract as c, evaluate as ev


@dataclass(frozen=True)
class FitContext:
    frame: str = "PI72-CLEAN"
    repeat: int = 1
    fold: int = -1
    inner: int = -1
    pool: str = "outer"

    def seed(self, purpose, arm="shared", trial=-1):
        return ev.seed(self.frame, purpose, repeat=self.repeat, fold=self.fold,
                       arm=arm, inner=self.inner, trial=trial)

    @property
    def key(self):
        return f"{self.frame}/r{self.repeat}/f{self.fold}/{self.pool}/i{self.inner}"


def aligned_labels(rows, labels):
    """Labels are a separate exact row-ID-indexed Series, never model input."""
    identity = c.validate_keys(rows.keys, frame=rows.frame)
    if not isinstance(labels, pd.Series) or labels.index.has_duplicates:
        raise c.ContractError("Labels must be a unique row-ID-indexed Series")
    ids = pd.Index(rows.keys[identity])
    if len(labels) != len(ids) or not ids.isin(labels.index).all():
        raise c.ContractError("Label coverage differs from fit pool")
    y = labels.reindex(ids).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise c.ContractError("Nonbinary/missing fit labels")
    return y.astype(np.int8)


def sorted_rows(rows, labels):
    from .features import take_rows
    identity = c.validate_keys(rows.keys, frame=rows.frame)
    keys = rows.keys.sort_values([c.H, c.DATE, identity])
    ordered = take_rows(rows, keys[identity])
    return ordered, aligned_labels(ordered, labels)


def stay_table(keys, y):
    c.validate_stay_ids(keys[c.H])
    return pd.DataFrame({c.H: keys[c.H].to_numpy(), "y": y}).groupby(c.H, sort=True).y.max()


def inner_folds(keys, y, context):
    """Three grouped stratified folds; single-class partitions fail, no redraw."""
    stays = stay_table(keys, y)
    if stays.value_counts().reindex([0, 1], fill_value=0).min() < 3:
        raise c.ContractError("Insufficient stay strata for three inner folds")
    splitter = StratifiedKFold(3, shuffle=True, random_state=context.seed("inner-folds"))
    for fit, val in splitter.split(stays.index, stays):
        train = np.flatnonzero(keys[c.H].isin(stays.index[fit]))
        valid = np.flatnonzero(keys[c.H].isin(stays.index[val]))
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise c.ContractError("Single-class inner fold; no redraw")
        yield train, valid


def pool_binding(rows, y):
    """One content binding per fit pool; not a per-file hash registry."""
    digest = hashlib.sha256(c.SPEC_SHA256.encode())
    digest.update(json.dumps([rows.frame, rows.bank, list(rows.values), rows.dependencies],
                             sort_keys=True).encode())
    for start in range(0, len(rows.keys), 1024):
        sl = slice(start, start + 1024)
        digest.update(pd.util.hash_pandas_object(rows.keys.iloc[sl], index=False).values.tobytes())
        digest.update(rows.values.iloc[sl].to_numpy(np.float32).tobytes())
        digest.update(rows.available[sl].tobytes())
        digest.update(rows.gates.iloc[sl].to_numpy(bool).tobytes())
        digest.update(y[sl].tobytes())
    return digest.hexdigest()


def candidates(rows):
    c.assert_predictors(rows.values.columns, bank=rows.bank, dependencies=rows.dependencies)
    if rows.bank != c.D5_BANK or not set(rows.values) <= set(c.D5SAFE):
        raise c.ContractError("Rankers require registered D5-safe source columns")
    # Missing/constant filter uses only this current training child.
    columns = sorted((name for name in rows.values if rows.values[name].nunique(dropna=True) > 1),
                     key=lambda name: name.encode("utf-8"))
    if not columns:
        raise c.ContractError("No nonconstant fit-pool candidates")
    return columns


def _attempt(ledger, key, arm, work, checkpoint=None):
    if checkpoint is not None and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["fit_key"] != key:
            raise c.ContractError("Selector trial checkpoint binding changed")
        if ledger is not None:
            ledger.recover(saved["attempt_id"])
        return np.asarray(saved["values"], dtype=np.float64)

    def compute(attempt_id=None):
        result = work()
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".partial")
            temporary.write_text(json.dumps({"fit_key": key, "attempt_id": attempt_id,
                                              "values": np.asarray(result).tolist()}, allow_nan=False) + "\n")
            temporary.replace(checkpoint)
        return result

    if ledger is None:
        return compute()
    with ledger.fit(key, arm=arm, kind="selector") as record:
        return compute(record["attempt_id"])


def gain_rank(rows, labels, context, *, ledger=None, arm="T8-GAIN5-LGB", guard=None, checkpoint_directory=None):
    from .trees import lgb_params
    c.reset_threads()
    import lightgbm as lgb
    if rows.frame != context.frame:
        raise c.ContractError("Selector context/frame mismatch")
    rows, y = sorted_rows(rows, labels)
    columns = candidates(rows)
    binding = pool_binding(rows, y)
    params = lgb_params(context.seed("gain"))
    if guard:
        guard()

    def work():
        data = lgb.Dataset(rows.values[columns].to_numpy(np.float32), label=y,
                           feature_name=columns, free_raw_data=True)
        model = lgb.train(params, data, num_boost_round=c.SPEC["selectors"]["GAIN"]["trees"])
        return model.feature_importance(importance_type="gain")

    checkpoint = None if checkpoint_directory is None else Path(checkpoint_directory) / "fit.json"
    gain = _attempt(ledger, f"{context.key}/{binding}/gain", arm, work, checkpoint)
    if not np.isfinite(gain).all() or gain.sum() <= 0:
        raise c.ContractError("Zero/nonfinite total gain")
    records = [{"column": col, "gain": float(value)} for col, value in zip(columns, gain)]
    records.sort(key=lambda r: (-r["gain"], r["column"].encode("utf-8")))
    return {"method": "GAIN", "context": context.key, "binding": binding,
            "candidate_count": len(columns), "ranking": records}


class RankingCache:
    """Reuse only exact fit identity and content; persist compact rankings."""
    def __init__(self, directory=None):
        self.directory = None if directory is None else Path(directory)
        self.memory = {}

    def get(self, method, rows, labels, context, **kwargs):
        ordered, y = sorted_rows(rows, labels)
        binding = pool_binding(ordered, y)
        key = (context.key, method)
        path = None
        if self.directory is not None:
            path = self.directory / context.key / f"{method}.json"
        result = self.memory.get(key)
        if result is None and path is not None and path.exists():
            result = json.loads(path.read_text())
        if result is not None:
            if result["binding"] != binding:
                raise c.ContractError("Ranking cache fit-pool binding changed; use a new run directory")
            return result
        function = gain_rank if method == "GAIN" else None
        if function is None:
            raise c.ContractError("Unknown selector")
        result = function(ordered, pd.Series(y, index=ordered.keys[c.validate_keys(ordered.keys, frame=ordered.frame)]),
                          context, checkpoint_directory=None if path is None else path.parent / f"{method}_fits", **kwargs)
        self.memory[key] = result
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    def select(self, recipe, rows, labels, context, **kwargs):
        entry = c.SPEC["feature_sets"][recipe]
        if "selector" not in entry:
            return c.feature_columns(recipe, frame=context.frame)
        if context.frame == "A-formal" and entry["selector"] == "SHADOW":
            raise c.ContractError("SHADOW is outside Task A")
        ranking = self.get(entry["selector"], rows, labels, context, **kwargs)
        k = entry["nominal_k"]
        if len(ranking["ranking"]) < k:
            raise c.ContractError(f"Insufficient fit-pool candidates for {recipe}; no silent k change")
        return c.feature_columns(recipe, selected=[r["column"] for r in ranking["ranking"][:k]], frame=context.frame)
