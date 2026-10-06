"""Frozen interval support states with prefix invariance checks."""
import copy
import hashlib
from pathlib import Path
from datetime import datetime
from . import core as common
import numpy as np
import pandas as pd
H = "hospitalizationidnew"


SOURCES = {
    "vent": ("MechVent", "ventstartSHIFT", "ventendSHIFT"),
    "ecmo": ("ECMO", "MechCircSuppInitDtTmSHIFT", "MechCircSuppDiscDtTmSHIFT"),
    "sternum": ("Sternum", "SternumDtSHIFT", "SternumClosedDtSHIFT"),
}


STATES = ("in_use_eod", "confirmed_restarts", "current_run_days", "days_since_weaning")


PAIRS = (("vent", "ecmo"), ("vent", "sternum"), ("ecmo", "sternum"))


COLUMNS = [f"suppS_{support}_{state}" for support in SOURCES for state in STATES] + [
    f"suppS_{a}_{b}_eod" for a, b in PAIRS]


EMPTY = np.empty((0, 4), dtype=float)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stay_ids(values):
    if values.isna().any():
        raise ValueError("Missing hospitalization key")
    ids = values.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    if ids.isin(("", ".", "nan", "None", "<NA>")).any():
        raise ValueError("Missing hospitalization key")
    return ids


def _date(value):
    """Return Excel-epoch serial and date-only flag; missing is not an event."""
    value = str(value).strip()
    for fmt, date_only in (("%d%b%y:%H:%M:%S", False), ("%d-%b-%y", True)):
        try:
            parsed = datetime.strptime(value, fmt)
            return (parsed - datetime(1899, 12, 30)).total_seconds() / 86400, date_only
        except ValueError:
            pass
    return np.nan, False


def load_intervals(raw_dir):
    """Read only keys/start/end; return label-free intervals and aggregate audit.

    Each array has start, union right boundary, recorded end day, date-only-end.
    Unusable ends become infinity; no raw frame is mutated.
    """
    intervals, audit = {}, {}
    for support, (table, start_field, end_field) in SOURCES.items():
        path = Path(raw_dir) / f"{table}.csv"
        frame = pd.read_csv(path, usecols=["hospitalizationidNEW", start_field, end_field],
                            dtype=str, keep_default_na=False, encoding="utf-8-sig")
        keys = _stay_ids(frame.hospitalizationidNEW)
        starts = np.array([_date(v)[0] for v in frame[start_field]], dtype=float)
        parsed_ends = [_date(v) for v in frame[end_field]]
        ends = np.array([v[0] for v in parsed_ends], dtype=float)
        date_only = np.array([v[1] for v in parsed_ends], dtype=bool)
        boundary = ends + date_only
        reversed_end = np.isfinite(starts) & np.isfinite(ends) & (boundary < starts)
        usable = np.isfinite(ends) & ~reversed_end
        packed = np.column_stack((starts, np.where(usable, boundary, np.inf),
                                  np.where(usable, np.floor(ends), np.nan), date_only))
        valid = pd.DataFrame({H: keys, "position": np.arange(len(frame))}).loc[np.isfinite(starts)]
        intervals[support] = {}
        for stay, rows in valid.groupby(H, sort=False):
            values = packed[rows.position.to_numpy()]
            intervals[support][stay] = values[np.argsort(values[:, 0], kind="stable")]
        audit[table] = {"rows": len(frame), "stays": keys.nunique(),
                        "missing_or_invalid_start": int((~np.isfinite(starts)).sum()),
                        "missing_or_invalid_end": int((~np.isfinite(ends)).sum()),
                        "reversed_end": int(reversed_end.sum()), "date_only_end": int(date_only.sum()),
                        "sha256": file_hash(path)}
    return intervals, common.json_safe(audit)


def _support_state(records, day):
    cutoff = day + 1
    visible = records[records[:, 0] < cutoff].copy()
    if not len(visible):
        return (0, 0, 0, np.nan)
    observed_end = np.where(visible[:, 3].astype(bool), visible[:, 2] <= day,
                            visible[:, 1] < cutoff)
    visible[~observed_end, 1] = np.inf
    visible[~observed_end, 2] = np.nan
    merged = []
    for start, end, end_day, _ in visible[np.argsort(visible[:, 0], kind="stable")]:
        if not merged or start > merged[-1][1]:
            merged.append([start, end, end_day])
        elif end > merged[-1][1]:
            merged[-1][1:] = [end, end_day]
        elif end == merged[-1][1] and np.isfinite(end):
            merged[-1][2] = max(merged[-1][2], end_day)
    restarts = sum(np.floor(b[0]) - np.ceil(a[1]) >= 1 for a, b in zip(merged, merged[1:]))
    active = int(np.isinf(merged[-1][1]))
    run_days = day - np.floor(merged[-1][0]) + 1 if active else 0
    completed = [end_day for _, end, end_day in merged if np.isfinite(end)]
    since = day - completed[-1] if completed else np.nan
    return (active, restarts, run_days, since)


def _keys(frame):
    ids = _stay_ids(frame[H]).to_numpy()
    days = frame.actual_date.to_numpy(dtype=float)
    if not np.isfinite(days).all() or not np.equal(days, np.floor(days)).all():
        raise ValueError("Scoring dates must be finite integer calendar days")
    return ids, days


def build_features(frame, intervals):
    """Build exactly 15 float32 columns, preserving row order and pandas index."""
    ids, days = _keys(frame)
    if set(intervals) != set(SOURCES):
        raise ValueError("All three support sources are required")
    values = np.empty((len(frame), 15), dtype=np.float32)
    memo = {}
    for i, (stay, day) in enumerate(zip(ids, days)):
        key = (stay, day)
        if key not in memo:
            states = [_support_state(intervals[s].get(stay, EMPTY), day) for s in SOURCES]
            on = {s: state[0] for s, state in zip(SOURCES, states)}
            memo[key] = [v for state in states for v in state] + [on[a] * on[b] for a, b in PAIRS]
        values[i] = memo[key]
    return pd.DataFrame(values, columns=COLUMNS, index=frame.index)


def future_perturbation_audit(frame, intervals):
    """Delete/alter future intervals and erase/shift unknown ends for every key.

    Also used on real POD3 keys; only aggregate counts leave memory.
    """
    ids, days = _keys(frame)
    count = future_starts = future_ends = 0
    for stay, day in set(zip(ids, days)):
        for support in SOURCES:
            records = intervals[support].get(stay, EMPTY)
            expected = _support_state(records, day)
            future = records[:, 0] >= day + 1
            unknown = np.where(records[:, 3].astype(bool), records[:, 2] > day,
                               records[:, 1] >= day + 1) | ~np.isfinite(records[:, 1])
            prefix = records[~future].copy()
            erase = unknown[~future]
            prefix[erase, 1] = np.inf
            prefix[erase, 2] = np.nan
            shifted = records.copy()
            shifted[future, 0] += 100
            shifted[unknown, 1] = day + 200
            shifted[unknown, 2] = day + 200
            shifted[unknown, 3] = 0
            np.testing.assert_array_equal(expected, _support_state(prefix, day))
            np.testing.assert_array_equal(expected, _support_state(shifted, day))
            count += 2
            future_starts += int(future.sum())
            future_ends += int((unknown & ~future).sum())
    return {"comparisons": count, "future_start_records_across_scoring_rows": future_starts,
            "unobserved_end_records_across_scoring_rows": future_ends, "passed": True}
