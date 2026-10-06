"""Evaluation harness v3 (post-lock, 2026-09-24): the one-time locked evaluation (cc1.0) is done, so the former locked
stays join the development pool. All metric code is imported unchanged from harness_v2 (r6); only the split table differs.

Splits (splits_v3.parquet, by hospitalization, all 11,907 stays): 5 repeats x 5-fold StratifiedGroupKFold stratified on
"has any positive row" (seed = 100 + repeat, so v3 folds differ from v2), plus 5 site groups (GroupKFold on site).
`late_era` marks index surgery >= 2022-10-01: from 2022Q4 the extract is right-truncated (stays still in hospital at the
Feb-2024 extraction are missing; long stays and NEC under-represented), used only for sensitivity analyses.
There is no untouched hold-out any more: every v3 number is a cross-validated development estimate.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .harness_v2 import (B, CACHE, H, LANDMARKS, N_FOLDS, N_REPEATS, PRIMARY_B, PRIMARY_B_LABEL,  # noqa: F401
                        evaluate_A, evaluate_B, evaluate_Bdaily, evaluate_transfer, landmark_frame,
                        threshold_from_oof)
from . import harness_v2 as _v2

# Data version of the row cache (r2, 2026-09-24): "v2.5" = v2_rows.parquet (default, all v25P runs);
# "v2.6" = v26_rows.parquet (E5 semantic fixes; same population, labels, row keys and order). Splits are shared.
ROWS_FILES = {"v2.5": "v2_rows.parquet", "v2.6": "v26_rows.parquet"}
DATA = "v2.5"


def use_data(version):
    global DATA
    assert version in ROWS_FILES, version
    DATA = version


def rows(cols=None):
    return pd.read_parquet(CACHE / ROWS_FILES[DATA], columns=cols)


LATE_ERA_SERIAL = (pd.Timestamp("2022-10-01") - pd.Timestamp("1899-12-30")).days
SPLIT_COLS = [H, "surg", "site", "late_era"] + [f"fold_r{k}" for k in range(N_REPEATS)] + ["fold_site"]


def splits():
    path = CACHE / "splits_v3.parquet"
    if not path.exists():
        raise FileNotFoundError("splits_v3.parquet missing: the controller must run make_splits() explicitly")
    return pd.read_parquet(path, columns=SPLIT_COLS)


def make_splits():
    """Controller-only: create the v3 split file."""
    path = CACHE / "splits_v3.parquet"
    if path.exists():
        raise FileExistsError(path)
    r = rows([H, "surg", "site", "y3"])
    st = r.groupby(H).agg(surg=("surg", "first"), site=("site", "first"), pos=("y3", "max")).reset_index()
    st["late_era"] = (st.surg >= LATE_ERA_SERIAL).astype(int)
    for k in range(N_REPEATS):
        f = np.full(len(st), -1)
        for j, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=100 + k).split(st, st.pos, st[H])):
            f[te] = j
        st[f"fold_r{k}"] = f
    f = np.full(len(st), -1)
    for j, (_, te) in enumerate(GroupKFold(N_FOLDS).split(st, st.pos, st.site)):
        f[te] = j
    st["fold_site"] = f
    st[SPLIT_COLS].to_parquet(path)
    return st[SPLIT_COLS]


def dev_rows(cols=None):
    sp = splits()
    r = rows(cols)
    return r.merge(sp[[H, "late_era"] + [f"fold_r{k}" for k in range(N_REPEATS)] + ["fold_site"]], on=H).reset_index(drop=True)


def file_hash():
    import hashlib
    from pathlib import Path
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(_v2.__file__).read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    import json
    sp = make_splits() if not (CACHE / "splits_v3.parquet").exists() else splits()
    d = dev_rows([H, "set", "y3"])
    print(json.dumps({"stays": int(len(sp)), "late_era_stays": int(sp.late_era.sum()),
                      "rows_by_set": d.set.value_counts().to_dict(),
                      "pos_rows_by_set": d.groupby("set").y3.sum().astype(int).to_dict(),
                      "harness_v3_sha256": file_hash()}, indent=1))
