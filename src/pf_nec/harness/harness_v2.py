"""Frozen evaluation harness v2 (DEC-002): Task A (72-h rolling warning) and Task B (landmark → NEC by POD31)
on the clean-format v2 rows. DO NOT EDIT during the loop; changes need a harness-revision record and a new hash.

Splits (by hospitalization, all sets): locked = index surgery ≥ 2022-07-01 (sealed until the final evaluation);
development = 5 repeats × 5-fold StratifiedGroupKFold (seed = repeat), stratified on "has any positive row";
plus 5 site groups (GroupKFold on site) for new-site sensitivity.

Task A frame = formal rows (evaluation); aux rows (c_aux, d_window, late_aux) may be added to TRAINING folds only,
and are scored for transfer evaluation. Task B frames are built from formal rows at landmarks.

Revision r2 (2026-09-24, before any valid v2.2 result): data v2.2 (uniform as-of rebuild); Task B gets two labels —
yB = first NEC after the landmark up to POD31 (clean window) and yB30 = first NEC within 30 days after the landmark
(any POD; follow-up from the raw NEC record) — the user decides which is primary; d-transfer uses d_window + late_aux.
Revision r3 (2026-09-24, before any valid v2.3 result): data v2.3; transfer metrics renamed and restricted —
preop_pooled_auc = c_aux preoperative rows + formal preoperative rows (both groups' positives); late_auc = late_aux
rows only (POD>31); formal_plus_dwindow_auc = sensitivity adding the d stays' rows up to POD31; metrics per repeat.
Revision r4 (2026-09-24, DEC-005, before any v2.5 result): data v2.5; Task B primary label = yB30 (30 days after the
landmark; yB up to POD31 kept for POD3 comparison); Task B-daily = y30 on every formal day (evaluate_Bdaily);
locked-set helpers with a FIXED alert threshold taken from development OOF (evaluate_A_locked, threshold_from_oof).
Revision r5 (2026-09-24, DEC-006, before the locked set is opened): (1) bootstrap percentiles drop non-finite draws
(a resample without both classes) and report the number of valid draws (no r4 development CI had an invalid draw, so r4
numbers are unchanged); (2) dev_rows()/locked_rows() filter stays at the parquet read, so locked rows never enter memory
on the development path and locked label columns are read only when requested; (3) evaluate_A_locked reports the
unpenalized calibration slope and the calibration intercept with logit(p) as a fixed offset (plus the logit mean
difference, renamed), and the number of valid bootstrap draws.
Revision r6 (2026-09-24, DEC-006 round 2, before the locked set is opened): splits() reads only outcome-free columns
(the stay-level "pos" stratifier is excluded) and never regenerates the split file at run time (make_splits() is a
separate controller step); the offset calibration intercept uses a bracketed root finder (NaN only for a single-class
set); evaluate_B / evaluate_Bdaily also report the number of valid bootstrap draws. Development metrics are unchanged.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..config import CACHE_ROOT as CACHE
H = "hospitalizationidnew"
LOCK_SERIAL = (pd.Timestamp("2022-07-01") - pd.Timestamp("1899-12-30")).days
N_REPEATS, N_FOLDS, B = 5, 5, 400
LANDMARKS = {"DOA0": ("doa", 0), "DOA2": ("doa", 2), "POD0": ("pod", 0), "POD1": ("pod", 1), "POD3": ("pod", 3), "POD7": ("pod", 7)}
PRIMARY_B = ("DOA0", "POD3")
PRIMARY_B_LABEL = "yB30"


def rows(cols=None, stays=None):
    """stays: optional stay ids; filtered at the parquet read (DEC-006) so other stays never enter memory."""
    filters = None if stays is None else [(H, "in", [float(x) for x in stays])]
    return pd.read_parquet(CACHE / "v2_rows.parquet", columns=cols, filters=filters)


SPLIT_COLS = [H, "surg", "site", "locked"] + [f"fold_r{k}" for k in range(N_REPEATS)] + ["fold_site"]


def splits():
    """Frozen split table without outcome columns (DEC-006 r2). Missing file = error; see make_splits()."""
    path = CACHE / "splits_v2.parquet"
    if not path.exists():
        raise FileNotFoundError("splits_v2.parquet missing: the controller must run make_splits() explicitly")
    return pd.read_parquet(path, columns=SPLIT_COLS)


def make_splits():
    """Controller-only: create the split file (stratified on each stay's outcome; never called at run time)."""
    path = CACHE / "splits_v2.parquet"
    if path.exists():
        raise FileExistsError(path)
    r = rows([H, "surg", "site", "y3", "set"])
    st = r.groupby(H).agg(surg=("surg", "first"), site=("site", "first"), pos=("y3", "max")).reset_index()
    st["locked"] = (st.surg >= LOCK_SERIAL).astype(int)
    dev = st[st.locked == 0].reset_index(drop=True)
    for k in range(N_REPEATS):
        f = np.full(len(dev), -1)
        for j, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=k).split(dev, dev.pos, dev[H])):
            f[te] = j
        dev[f"fold_r{k}"] = f
    f = np.full(len(dev), -1)
    for j, (_, te) in enumerate(GroupKFold(N_FOLDS).split(dev, dev.pos, dev.site)):
        f[te] = j
    dev["fold_site"] = f
    out = st.merge(dev[[H] + [f"fold_r{k}" for k in range(N_REPEATS)] + ["fold_site"]], on=H, how="left")
    out.to_parquet(path)
    return out


def dev_rows(cols=None):
    sp = splits()
    r = rows(cols, stays=sp.loc[sp.locked == 0, H])
    r = r.merge(sp[[H, "locked"] + [f"fold_r{k}" for k in range(N_REPEATS)] + ["fold_site"]], on=H)
    return r[r.locked == 0].drop(columns="locked").reset_index(drop=True)


def locked_rows(cols=None):
    if os.environ.get("LOCKED_OK") != "1":
        raise RuntimeError("Locked set sealed until the final evaluation (LOCKED_OK=1 only once).")
    sp = splits()
    r = rows(cols, stays=sp.loc[sp.locked == 1, H])
    assert r[H].isin(sp.loc[sp.locked == 1, H]).all()
    return r.reset_index(drop=True)


def landmark_frame(df, name):
    """One row per eligible formal stay at the landmark (end of day). Label: first NEC after the landmark, ≤ POD31."""
    kind, k = LANDMARKS[name]
    f = df[df.set == "formal"]
    if kind == "doa":
        f = f[f.doa == k]
    else:
        f = f[(f.post_surg == 1) & (f.pod_raw == k)]
    f = f.copy()
    f["yB"] = ((f.nec > f.actual_date) & (f.nec - f.surg <= 31)).astype(int)
    f["yB30"] = ((f.nec > f.actual_date) & (f.nec - f.actual_date <= 30)).astype(int)
    return f.reset_index(drop=True)


# ------------------------------------------------------------------ metrics
def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def _calib_slope(y, p):
    from ..legacy.calib_mle import calib_slope_mle
    return calib_slope_mle(y, p)


def _ci(x):
    """95% percentile interval over finite bootstrap draws only (DEC-006); NaN if none is finite."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2


def _calib_intercept(y, lp):
    """alpha such that mean(expit(alpha + lp)) = mean(y): calibration-in-the-large with logit(p) as a fixed offset.
    Bracketed root finding (monotone in alpha); NaN when the set has a single class (not estimable)."""
    from scipy.optimize import brentq
    from scipy.special import expit
    ybar = float(np.mean(y))
    if ybar <= 0 or ybar >= 1:
        return float("nan")
    f = lambda a: float(np.mean(expit(a + lp))) - ybar  # noqa: E731
    lo, hi = -50.0, 50.0
    while f(lo) > 0:
        lo *= 2
    while f(hi) < 0:
        hi *= 2
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500))


def _boot(df, P, y, ref, b, seed):
    rng = np.random.default_rng(seed)
    hs = df[H].values
    uniq, inv = np.unique(hs, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(order)]
    boots, deltas = [], []
    for _ in range(b):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        a = np.nanmean([_auc(y[ix], P[ix, r]) for r in range(P.shape[1])])
        boots.append(a)
        if ref is not None:
            deltas.append(a - np.nanmean([_auc(y[ix], ref[ix, r]) for r in range(ref.shape[1])]))
    return boots, deltas


def evaluate_A(df, P, ref=None, b=B, seed=2026):
    """df = formal dev rows (y3, post_surg, pod_raw, doa, group, lead); P = OOF (n, repeats). auc_lead3 keeps only the
    positive rows exactly 3 days before NEC (one per case, like the clean table's outcome_3d) plus all negative rows."""
    y = df.y3.values
    per = []
    phases = {"preop": df.post_surg.values == 0, "pod0_3": (df.post_surg.values == 1) & (df.pod_raw.values <= 3),
              "pod4plus": (df.post_surg.values == 1) & (df.pod_raw.values >= 4)}
    for r in range(P.shape[1]):
        p = P[:, r]
        thr = np.quantile(p[y == 0], 0.95)  # 5% of negative days alarm
        m = {"auc": _auc(y, p), "ap": average_precision_score(y, p), "calib_slope": _calib_slope(y, p),
             "auc_lead3": _auc(y[(y == 0) | (df.lead.values == 3)], p[(y == 0) | (df.lead.values == 3)])}
        for k, msk in phases.items():
            m[f"auc_{k}"] = _auc(y[msk], p[msk])
        pos = df[y == 1].assign(p=p[y == 1])
        det = pos.groupby(H).agg(g=("group", "first"), hit=("p", lambda s: (s >= thr).any()))
        m["detect_at_fpr5_all"] = det.hit.mean()
        for gname in ("a", "b"):
            m[f"detect_at_fpr5_{gname}"] = det[det.g == gname].hit.mean()
        per.append(m)
    per = pd.DataFrame(per)
    boots, deltas = _boot(df, P, y, ref, b, seed)
    out = {"n_rows": int(len(y)), "n_pos_rows": int(y.sum()), "n_cases": int(df.loc[y == 1, H].nunique()),
           "point": per.mean().to_dict(), "repeat_sd": per.std().to_dict(),
           "auc_ci": _ci(boots)}
    if ref is not None:
        out["delta_auc"] = float(per.auc.mean() - np.nanmean([_auc(y, ref[:, r]) for r in range(ref.shape[1])]))
        out["delta_ci"] = _ci(deltas)
    return out


def evaluate_B(frames, preds, ref=None, b=B, seed=2026, label="yB"):
    """frames: {landmark: df with yB/yB30}; preds: {landmark: (n, repeats)}; ref: same structure or None."""
    out = {}
    for name, df in frames.items():
        y = df[label].values
        P = preds[name]
        aucs = [_auc(y, P[:, r]) for r in range(P.shape[1])]
        R = None if ref is None else ref[name]
        boots, deltas = _boot(df, P, y, R, b, seed)
        o = {"n": int(len(y)), "n_pos": int(y.sum()), "auc": float(np.nanmean(aucs)), "auc_sd": float(np.nanstd(aucs)),
             "ap": float(np.mean([average_precision_score(y, P[:, r]) for r in range(P.shape[1])])),
             "auc_ci": _ci(boots), "boot_valid": int(np.isfinite(boots).sum())}
        if R is not None:
            o["delta_auc"] = float(o["auc"] - np.nanmean([_auc(y, R[:, r]) for r in range(R.shape[1])]))
            o["delta_ci"] = _ci(deltas)
        out[name] = o
    return out


def evaluate_transfer(aux_df, P_aux, formal_df, P_formal):
    """Per repeat, then averaged. P_aux/P_formal: (n, repeats)."""
    out = []
    pre_f = formal_df.post_surg.values == 0
    c = aux_df.set.values == "c_aux"
    late = aux_df.set.values == "late_aux"
    dw = aux_df.set.values == "d_window"
    for r in range(P_formal.shape[1]):
        m = {}
        if c.any():
            m["preop_pooled_auc"] = _auc(np.r_[aux_df.y3.values[c], formal_df.y3.values[pre_f]], np.r_[P_aux[c, r], P_formal[pre_f, r]])
        if late.any():
            m["late_auc"] = _auc(aux_df.y3.values[late], P_aux[late, r])
        if dw.any():
            m["formal_plus_dwindow_auc"] = _auc(np.r_[formal_df.y3.values, aux_df.y3.values[dw]], np.r_[P_formal[:, r], P_aux[dw, r]])
        out.append(m)
    res = pd.DataFrame(out).mean().to_dict()
    res.update(preop_pooled_pos_rows=int(aux_df.y3.values[c].sum() + formal_df.y3.values[pre_f].sum()),
               late_pos_rows=int(aux_df.y3.values[late].sum()))
    return res


def evaluate_Bdaily(df, P, ref=None, b=B, seed=2026):
    """Task B-daily: y30 on every formal day row; pooled AUC per repeat (averaged), phase AUCs, and the AUC of the daily
    model restricted to each landmark's rows (the same rows landmark_frame would select)."""
    y = df.y30.values
    per = []
    phases = {"preop": df.post_surg.values == 0, "pod0_3": (df.post_surg.values == 1) & (df.pod_raw.values <= 3),
              "pod4plus": (df.post_surg.values == 1) & (df.pod_raw.values >= 4)}
    lm_masks = {}
    for name, (kind, k) in LANDMARKS.items():
        lm_masks[name] = (df.doa.values == k) if kind == "doa" else ((df.post_surg.values == 1) & (df.pod_raw.values == k))
    for r in range(P.shape[1]):
        p = P[:, r]
        m = {"auc": _auc(y, p), "ap": average_precision_score(y, p)}
        for k, msk in phases.items():
            m[f"auc_{k}"] = _auc(y[msk], p[msk])
        for k, msk in lm_masks.items():
            m[f"auc_at_{k}"] = _auc(y[msk], p[msk])
        per.append(m)
    per = pd.DataFrame(per)
    boots, deltas = _boot(df, P, y, ref, b, seed)
    out = {"n_rows": int(len(y)), "n_pos_rows": int(y.sum()), "point": per.mean().to_dict(), "repeat_sd": per.std().to_dict(),
           "auc_ci": _ci(boots), "boot_valid": int(np.isfinite(boots).sum())}
    if ref is not None:
        out["delta_auc"] = float(per.auc.mean() - np.nanmean([_auc(y, ref[:, r]) for r in range(ref.shape[1])]))
        out["delta_ci"] = _ci(deltas)
    return out


def threshold_from_oof(df, P, alert_rate=0.05):
    """Fixed alert threshold from DEVELOPMENT out-of-fold scores: the (1 - alert_rate) quantile of negative-day scores,
    per repeat, averaged over repeats. Frozen before the locked set is opened."""
    y = df.y3.values
    return float(np.mean([np.quantile(P[y == 0, r], 1 - alert_rate) for r in range(P.shape[1])]))


def evaluate_A_locked(df, p, thr, ref=None, b=1000, seed=2027):
    """Locked-set Task A with a FIXED threshold (never re-estimated on the locked labels). p: final-model scores (n,)."""
    y = df.y3.values
    P = p[:, None]
    R = None if ref is None else ref[:, None]
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc))
    from sklearn.linear_model import LogisticRegression
    slope = _calib_slope(y, p)
    citl = _calib_intercept(y, lp)
    lmd = float(np.log(y.mean() / (1 - y.mean())) - np.log(pc.mean() / (1 - pc.mean())))
    phases = {"preop": df.post_surg.values == 0, "pod0_3": (df.post_surg.values == 1) & (df.pod_raw.values <= 3),
              "pod4plus": (df.post_surg.values == 1) & (df.pod_raw.values >= 4)}
    alert = p >= thr
    pos = df[y == 1].assign(hit=alert[y == 1])
    det = pos.groupby(H).agg(g=("group", "first"), hit=("hit", "max"))
    boots, deltas = _boot(df, P, y, R, b, seed)
    out = {"n_rows": int(len(y)), "n_pos_rows": int(y.sum()), "n_cases": int(df.loc[y == 1, H].nunique()),
           "auc": _auc(y, p), "auc_ci": _ci(boots),
           "ap": float(average_precision_score(y, p)), "calib_slope": slope, "calib_intercept": citl,
           "logit_mean_difference": lmd, "boot_valid": int(np.isfinite(boots).sum()),
           "threshold": thr, "negative_day_alert_rate": float(alert[y == 0].mean()), "detection_all": float(det.hit.mean()),
           "detection_a": float(det[det.g == "a"].hit.mean()), "detection_b": float(det[det.g == "b"].hit.mean())}
    for k, msk in phases.items():
        out[f"auc_{k}"] = _auc(y[msk], p[msk])
    if ref is not None:
        out["delta_auc"] = float(out["auc"] - _auc(y, ref))
        out["delta_ci"] = _ci(deltas)
    return out


def file_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    import json
    sp = splits()
    d = dev_rows([H, "set", "y3", "group"])
    s = {"stays": int(len(sp)), "locked_stays": int(sp.locked.sum()), "dev_rows_by_set": d.set.value_counts().to_dict(),
         "dev_pos_rows_by_set": d.groupby("set").y3.sum().astype(int).to_dict(),
         "dev_formal_cases": int(d[(d.set == "formal") & (d.y3 == 1)][H].nunique()), "sha256": file_hash()}
    df = dev_rows()
    s["taskB"] = {k: {"n": int(len(f)), "pos_POD31": int(f.yB.sum()), "pos_30d": int(f.yB30.sum())} for k, f in ((k, landmark_frame(df, k)) for k in LANDMARKS)}
    print(json.dumps(s, indent=1))
