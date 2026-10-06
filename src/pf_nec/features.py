"""Frozen D5 masking and row adapters; source-model bank excluded."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import contract as c


@dataclass(frozen=True)
class FeatureRows:
    """One masked source row per key. Gates/anchors are metadata, never X columns."""
    keys: pd.DataFrame
    values: pd.DataFrame
    available: np.ndarray
    gates: pd.DataFrame
    admission: np.ndarray
    bank: str
    frame: str
    dependencies: dict


def availability_gates(keys, anchors, *, frame="PI72-CLEAN"):
    """Strict < cutoff, including exact midnight and ICU+2h boundary cases."""
    aligned = c.align_anchors(keys, anchors, frame=frame)
    t = keys[c.DATE].to_numpy(np.float64)
    surg, icu = aligned.surg.to_numpy(np.float64), aligned.icu_serial.to_numpy(np.float64)
    post = t >= surg
    plausible = np.isfinite(icu) & (icu - surg >= 0) & (icu - surg <= 2)
    surgery = post & ((t > surg) | (plausible & (icu < t + 1)))
    two_hour = post & plausible & (icu + 2 / 24 < t + 1)
    gates = pd.DataFrame({"post_surg": post, "surgery_available": surgery,
                          "two_hour_available": two_hour})
    return gates, aligned


def build_features(keys, values, anchors, *, bank, frame="PI72-CLEAN", raw=None,
                   provenance=None, dependencies=None):
    """Align exact row IDs, mask historical days, and return float32 FeatureRows.

    `values` is predictor-only with its index named source_row_id (PI) or
    __harness_row_index (A). `keys` and `anchors` use the strict contract schemas.
    Derived columns must declare their dependency DAG and inherit root gates.
    Source vis/is and the two guarded event columns must be built as base columns
    first; aliases cannot bypass reconstruction/provenance by supplying values.
    """
    row = c.validate_keys(keys, frame=frame)
    roots = c.predictor_roots(values.columns, bank=bank, dependencies=dependencies)
    if frame == "A-formal" and bank != c.D5_BANK:
        raise c.ContractError("Task A requires its own D5 rows; no source PI inner join")
    if values.index.name != row or not values.index.is_unique or values.index.hasnans:
        raise c.ContractError("Feature table needs a unique, named, nonmissing row index")
    if not keys[row].isin(values.index).all():
        raise c.ContractError("Missing feature rows; target rows may not be dropped")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise c.ContractError("Predictors must be numeric (original unknown codes retained)")
    x = values.loc[keys[row]].reset_index(drop=True).astype(np.float32)
    if np.isinf(x.to_numpy()).any():
        raise c.ContractError("Infinite predictors are not permitted")
    gates, aligned = availability_gates(keys, anchors, frame=frame)
    if (keys[c.DATE].to_numpy() < aligned.adm.to_numpy()).any():
        raise c.ContractError("Observed source row precedes admission")
    available = np.ones(x.shape, dtype=bool)
    if bank == c.D5_BANK:
        spec = c.SPEC["feature_sets"]["D5SAFE"]
        preop = set(spec["preop_full_mask"])
        surgery = set(spec["preop_existing_surgery_mask"])
        doses = set(c.SPEC["availability"]["two_hour_columns_D5"])
    else:
        raise c.ContractError("Source-model bank is not exported")
    for j, name in enumerate(x):
        if roots[name] & preop:
            available[:, j] &= gates.post_surg.to_numpy()
        if roots[name] & surgery:
            available[:, j] &= gates.surgery_available.to_numpy()
        if roots[name] & doses:
            available[:, j] &= gates.two_hour_available.to_numpy()
    # Preserve all cache NaNs; an availability gate never fabricates a value.
    masked = x.to_numpy(np.float32, copy=True)
    masked[~available] = np.nan
    x = pd.DataFrame(masked, columns=x.columns)
    return FeatureRows(keys.reset_index(drop=True).copy(), x, available, gates,
                       aligned.adm.to_numpy(np.float64), bank, frame,
                       {} if dependencies is None else dict(dependencies))


def derive_features(rows, values, dependencies):
    """Gate externally derived numeric columns against already-masked roots.

    Compute from rows.values only; never hand raw surgery values to an encoder.
    Unknown/category indicators remain NaN while their source gate is closed.
    """
    if not values.index.equals(rows.values.index) or len(values) != len(rows.values):
        raise c.ContractError("Derived rows changed order/index")
    deps = {**rows.dependencies, **dependencies}
    c.predictor_roots(values.columns, bank=rows.bank, dependencies=deps)
    allowed = np.ones(values.shape, bool)
    for j, name in enumerate(values):
        parents = dependencies.get(name)
        if not parents or any(parent not in rows.values for parent in parents):
            raise c.ContractError("Derivations must depend directly on existing masked inputs")
        allowed[:, j] = rows.available[:, rows.values.columns.get_indexer(parents)].all(axis=1)
    x = values.astype(np.float32).copy()
    if np.isinf(x.to_numpy()).any():
        raise c.ContractError("Infinite derived feature")
    masked = x.to_numpy(np.float32, copy=True)
    masked[~allowed] = np.nan
    x = pd.DataFrame(masked, columns=x.columns)
    return FeatureRows(rows.keys.copy(), x, allowed, rows.gates.copy(), rows.admission.copy(),
                       rows.bank, rows.frame, deps)


def select_features(rows, columns):
    """Select a declared fixed/fold-local ordered list without rerunning gates."""
    columns = tuple(columns)
    c.assert_predictors(columns, bank=rows.bank, dependencies=rows.dependencies)
    if not set(columns) <= set(rows.values):
        raise c.ContractError("Selected feature absent from this bank")
    indices = rows.values.columns.get_indexer(columns)
    return FeatureRows(rows.keys.copy(), rows.values.loc[:, list(columns)].copy(),
                       rows.available[:, indices].copy(), rows.gates.copy(), rows.admission.copy(),
                       rows.bank, rows.frame, rows.dependencies.copy())


def take_rows(rows, identities):
    """Exact-order fit/apply subset; missing or repeated row IDs fail explicitly."""
    identity = c.validate_keys(rows.keys, frame=rows.frame)
    requested = pd.Index(identities)
    if requested.has_duplicates or requested.hasnans:
        raise c.ContractError("Duplicate or missing requested row ID")
    positions = pd.Index(rows.keys[identity]).get_indexer(requested)
    if (positions < 0).any():
        raise c.ContractError("Requested row ID absent from this retained frame")
    return FeatureRows(rows.keys.iloc[positions].reset_index(drop=True),
                       rows.values.iloc[positions].reset_index(drop=True), rows.available[positions].copy(),
                       rows.gates.iloc[positions].reset_index(drop=True), rows.admission[positions].copy(),
                       rows.bank, rows.frame, rows.dependencies.copy())


def from_task_a(frame, anchors, columns=None):
    """Adapter for a controller-owned nested.common.load_frames(A,v2.6) result.

    Explicitly projects the label-free boundary without changing keys/risk set.
    `anchors` must already be projected to H/adm/surg/icu_serial.
    """
    columns = c.D5SAFE if columns is None else tuple(columns)
    c.assert_predictors(columns, bank=c.D5_BANK)
    keys = frame[[c.H, c.DATE, c.HARNESS_ROW]].copy()
    values = frame[[c.HARNESS_ROW, *columns]].set_index(c.HARNESS_ROW)
    return build_features(keys, values, anchors, bank=c.D5_BANK, frame="A-formal")
