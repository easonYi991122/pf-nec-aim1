"""Bounded calendar windows from masked source rows; no fill or learned state."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract as c


@dataclass(frozen=True)
class WindowBatch:
    keys: pd.DataFrame
    columns: tuple
    values: np.ndarray             # n, w, k; oldest to current, NaN preserved
    row_observed: np.ndarray       # n, w; row exists even when every value is NaN
    available: np.ndarray          # n, w, k; historical structural gate, not !isna
    gates: np.ndarray              # n, w, 3; post/surgery/two-hour; metadata only
    source_positions: np.ndarray   # n, w; -1 for padding/gaps; never model input
    bank: str
    dependencies: dict


@dataclass(frozen=True)
class TreeMatrix:
    keys: pd.DataFrame
    values: np.ndarray
    columns: tuple
    concepts: tuple                # base feature for grouping; None = row structure
    observation_columns: tuple     # counts/row indicators separately interpretable


def calendar_windows(rows, targets=None, *, window):
    """Build one bounded batch, preserving target order and exact stay identity.

    Source history must already be the retained frame's feature rows. Targets
    must be a subset with identical row/stay/date keys, so no unlabelled sentinel
    days or silently missing targets can enter. Use iter_windows for real pools.
    """
    if window not in c.WINDOW_LENGTHS:
        raise c.ContractError("Window length is not preregistered")
    if window > 1 and len(rows.values.columns) >= len(c.D5SAFE):
        raise c.ContractError("Full D5 multi-day concatenation is prohibited")
    identity = c.validate_keys(rows.keys, frame=rows.frame)
    c.assert_predictors(rows.values.columns, bank=rows.bank, dependencies=rows.dependencies)
    if len(rows.keys) != len(rows.values) or rows.available.shape != rows.values.shape:
        raise c.ContractError("Misaligned masked source rows")
    targets = rows.keys if targets is None else targets
    c.validate_keys(targets, frame=rows.frame)
    index = pd.Index(rows.keys[identity])
    positions = index.get_indexer(targets[identity])
    if (positions < 0).any():
        raise c.ContractError("Window target is absent from retained source rows")
    expected = rows.keys.iloc[positions].reset_index(drop=True)
    if not expected[list(c.KEYS)].equals(targets[list(c.KEYS)].reset_index(drop=True)):
        raise c.ContractError("Window target row/stay/date identity mismatch")
    n, k = len(targets), len(rows.values.columns)
    values = np.full((n, window, k), np.nan, dtype=np.float32)
    observed = np.zeros((n, window), dtype=bool)
    available = np.zeros((n, window, k), dtype=bool)
    gates = np.zeros((n, window, 3), dtype=bool)
    source_positions = np.full((n, window), -1, dtype=np.int64)
    source_index = pd.MultiIndex.from_frame(rows.keys[list(c.KEYS)])
    source_values = rows.values.to_numpy(np.float32, copy=False)
    source_gates = rows.gates.to_numpy(bool)
    admission = rows.admission[positions]
    for slot, lag in enumerate(range(window - 1, -1, -1)):
        day = targets[c.DATE].to_numpy(np.float64) - lag
        lookup = pd.MultiIndex.from_arrays([targets[c.H], day], names=c.KEYS)
        found = source_index.get_indexer(lookup)
        ok = (found >= 0) & (day >= admission)
        observed[:, slot] = ok
        source_positions[ok, slot] = found[ok]
        values[ok, slot] = source_values[found[ok]]
        available[ok, slot] = rows.available[found[ok]]
        gates[ok, slot] = source_gates[found[ok]]
    return WindowBatch(targets.reset_index(drop=True).copy(), tuple(rows.values.columns),
                       values, observed, available, gates, source_positions,
                       rows.bank, rows.dependencies)


def iter_windows(rows, targets=None, *, window, batch_size=1024):
    """Yield bounded target batches; caller must not accumulate every arm/window."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise c.ContractError("batch_size must be a positive integer")
    targets = rows.keys if targets is None else targets
    # Validate globally before batching, including duplicates crossing a boundary.
    c.validate_keys(targets, frame=rows.frame)
    for start in range(0, len(targets), batch_size):
        yield calendar_windows(rows, targets.iloc[start:start + batch_size], window=window)


def _check_batch(batch):
    c.assert_predictors(batch.columns, bank=batch.bank, dependencies=batch.dependencies)
    v = batch.values
    if v.ndim != 3 or v.shape[1] not in c.WINDOW_LENGTHS or v.shape[2] != len(batch.columns):
        raise c.ContractError("Invalid window tensor shape")
    if batch.row_observed.shape != v.shape[:2] or batch.available.shape != v.shape:
        raise c.ContractError("Invalid availability tensor shape")
    if np.isinf(v).any() or np.isfinite(v[~batch.available]).any():
        raise c.ContractError("Window contains an infinite or structurally unavailable value")
    if np.isfinite(v[~batch.row_observed]).any():
        raise c.ContractError("Padding contains observed values")


def tree_matrix(batch, representation):
    """R1 (lag0 first) or R3; w1 both return exactly the original k columns.

    No preprocessing is fitted. Native/gated NaNs remain NaN. `concepts` records
    variable groups for interpretation; structural counts are identified apart.
    """
    _check_batch(batch)
    if representation not in ("R1", "R3"):
        raise c.ContractError("Only R1/R3 are tree representations")
    n, w, k = batch.values.shape
    dimensions = k if w == 1 else (k * w + w + 1 if representation == "R1" else 7 * k + 1)
    if dimensions > c.SPEC["windows"]["actual_engineered_column_cap"]:
        raise c.ContractError("Engineered tree input exceeds 800 columns")
    if w == 1:
        return TreeMatrix(batch.keys.copy(), batch.values[:, 0].copy(), batch.columns,
                          batch.columns, ())
    count = batch.row_observed.sum(axis=1).astype(np.float32)
    if representation == "R1":
        values = np.column_stack((batch.values[:, ::-1].reshape(n, w * k),
                                  batch.row_observed[:, ::-1].astype(np.float32), count))
        names = tuple(f"{name}__lag{lag}" for lag in range(w) for name in batch.columns)
        structure = tuple(f"row_observed__lag{lag}" for lag in range(w)) + ("window_observed_day_count",)
        concepts = batch.columns * w + (None,) * (w + 1)
        return TreeMatrix(batch.keys.copy(), values, names + structure, concepts, structure)
    v = batch.values
    finite = np.isfinite(v)
    nobs = finite.sum(axis=1)
    total = np.where(finite, v, 0).sum(axis=1, dtype=np.float64)
    mean = np.divide(total, nobs, out=np.full((n, k), np.nan), where=nobs > 0)
    minimum = np.where(finite, v, np.inf).min(axis=1)
    maximum = np.where(finite, v, -np.inf).max(axis=1)
    minimum[nobs == 0] = maximum[nobs == 0] = np.nan
    first = np.argmax(finite, axis=1)
    last = w - 1 - np.argmax(finite[:, ::-1], axis=1)
    first_value = np.take_along_axis(v, first[:, None, :], axis=1)[:, 0]
    last_value = np.take_along_axis(v, last[:, None, :], axis=1)[:, 0]
    delta = last_value - first_value
    delta[nobs < 2] = np.nan
    paired = finite[:, :-1] & finite[:, 1:]
    difference = np.abs(v[:, 1:].astype(np.float64) - v[:, :-1])
    changed = (paired & (difference > c.SPEC["windows"]["R3"]["change_tolerance"])).sum(axis=1).astype(np.float32)
    changed[~paired.any(axis=1)] = np.nan
    summaries = np.stack((v[:, -1], mean, minimum, maximum, delta, changed, nobs), axis=2)
    values = np.column_stack((summaries.reshape(n, k * 7), count)).astype(np.float32)
    statistics = c.SPEC["windows"]["R3"]["per_feature"]
    names = tuple(f"{name}__{stat}" for name in batch.columns for stat in statistics)
    structure = tuple(f"{name}__n_observed" for name in batch.columns) + ("window_observed_day_count",)
    concepts = tuple(name for name in batch.columns for _ in statistics) + (None,)
    return TreeMatrix(batch.keys.copy(), values, names + ("window_observed_day_count",), concepts, structure)


def r4_tensor(batch, *, transform=None):
    """Return float32 [n,w,2*k+1]: values, missing flags, row-observed flag.

    A transform, if supplied, is an ALREADY FIT rowwise sequence preprocessor:
    signed-log1p / training-only imputation/scale / clip[-6,6]. This module never
    fits statistics. Without it, native missing value channels stay NaN; that
    tensor is an inspection/packing intermediate, not network-ready input.
    Padding values are always zero, missing=1, row_observed=0. All k channels
    are dynamic, and transform receives only unique observed source rows.
    """
    _check_batch(batch)
    n, w, k = batch.values.shape
    if 2 * k + 1 > c.SPEC["windows"]["actual_engineered_column_cap"]:
        raise c.ContractError("R4 channel count exceeds 800")
    missing = np.isnan(batch.values)
    values = batch.values.copy()
    if transform is not None:
        observed = batch.row_observed
        ids = batch.source_positions[observed]
        _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
        unique_values = values[observed][first]
        converted = np.asarray(transform(unique_values.copy()), dtype=np.float32)
        if converted.shape != unique_values.shape or not np.isfinite(converted).all():
            raise c.ContractError("Frozen sequence transform must return finite values with the same shape")
        values[observed] = converted[inverse]
    values[~batch.row_observed] = 0
    return np.concatenate((values, missing.astype(np.float32),
                           batch.row_observed[..., None].astype(np.float32)), axis=2)
