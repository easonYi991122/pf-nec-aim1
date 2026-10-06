"""Frozen Task A formal frame loading and population digest."""
import hashlib
import json
import numpy as np
import pandas as pd
from ..harness import harness_v3 as hv


H = hv.H


ROW = "__harness_row_index"


META = [H, "set", "site", "y3", "y30", "post_surg", "pod_raw", "doa", "lead", "group",
        "nec", "surg", "actual_date"]


LABELS = {"A": "y3", "B30": "yB30", "Bd": "y30"}


def load_frames(task, data, columns=()):
    if task not in LABELS:
        raise ValueError(f"Unknown task: {task}; choose A, B30 or Bd")
    hv.use_data(data)
    df = hv.dev_rows(cols=list(dict.fromkeys(META + list(columns))))
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("dev_rows() must retain the harness's RangeIndex row order")
    df = df.assign(**{ROW: np.arange(len(df), dtype=np.int64)})
    formal = df.loc[df["set"].eq("formal")].copy().reset_index(drop=True)
    if formal.empty or formal[H].isna().any() or not formal.post_surg.isin([0, 1]).all():
        raise ValueError("Empty formal population, missing stay, or invalid phase")
    for r in range(hv.N_REPEATS):
        fc = f"fold_r{r}"
        if not formal[fc].isin(range(hv.N_FOLDS)).all() or formal.groupby(H)[fc].nunique().max() != 1:
            raise ValueError(f"Invalid frozen stay folds: {fc}")
    # Labels, risk-set membership, order and all frozen folds are checked again at evaluation.
    key_cols = [ROW] + META + [f"fold_r{r}" for r in range(hv.N_REPEATS)] + ["fold_site", "late_era"]
    population_hash = hashlib.sha256(pd.util.hash_pandas_object(formal[key_cols], index=False).values.tobytes()).hexdigest()
    if task == "B30":
        frames = {name: hv.landmark_frame(formal, name) for name in hv.LANDMARKS}
        if any(f.empty or f[H].duplicated().any() for f in frames.values()):
            raise ValueError("Each landmark must have a nonempty, one-row-per-stay frame")
    else:
        frames = {task: formal}
    for frame in frames.values():
        if not frame[LABELS[task]].isin([0, 1]).all():
            raise ValueError("Task labels must be binary and nonmissing")
    return frames, population_hash


def json_safe(value):
    """Strict JSON: undefined harness statistics are null, never NaN literals."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
