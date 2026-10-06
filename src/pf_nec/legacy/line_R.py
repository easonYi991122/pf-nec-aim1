"""Frozen raw-code mapping and pool-local shrinkage functions."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config
H = "hospitalizationidnew"
RAW = config.DATA_ROOT / "Raw CSV Files"


STRENGTH = 100


FOLDS = 3


FIELDS = ("fund", "primary_proc", "primary_surgdiag", "procedures", "preop_risk")


COLUMNS = ("raw_fund_rate", "raw_primary_proc_rate", "raw_primary_surgdiag_rate",
           "raw_procedures_mean", "raw_procedures_max", "raw_procedures_n",
           "raw_preop_risk_mean", "raw_preop_risk_max", "raw_preop_risk_n")


RAW_COLUMNS = {
    "IndexSurgHosp": ["hospitalizationidNEW", "operativeidNEW", "FundDiagnosis", "ProcPrimary",
                      "cardsurgdtSHIFT", "icupacuadmitdttmSHIFT", "hospadmitdtSHIFT"],
    "Surgdiag": ["hospitalizationidNEW", "operativeidNEW", "surgdiag", "SurgDiagPrim"],
    "Procedures": ["hospitalizationidNEW", "operativeidNEW", "ProcName"],
    "PreopRiskFactor": ["hospitalizationidNEW", "preopfactor"],
}


def _code(value):
    if pd.isna(value) or str(value).strip() in ("", ".", "NA", "NaN", "nan", "None"):
        return ""
    return re.sub(r"\.0+$", "", str(value).strip())


def _serial(values):
    """Same two raw date formats / Excel epoch as build_v26.raw_dt."""
    values = values.astype(str)
    dates = pd.to_datetime(values, format="%d-%b-%y", errors="coerce").fillna(
        pd.to_datetime(values, format="%d%b%y:%H:%M:%S", errors="coerce"))
    return (dates - pd.Timestamp("1899-12-30")).dt.total_seconds() / 86400


def read_tables(raw_dir=RAW):
    return {name: pd.read_csv(Path(raw_dir) / f"{name}.csv", usecols=columns,
                             encoding="cp1252", dtype=str, keep_default_na=False)
            for name, columns in RAW_COLUMNS.items()}


def build_codebook(tables):
    """Label-free, one-stay code sets; subordinate surgery tables use the index ID.

    Primary Surgdiag means exactly one distinct nonmissing code flagged 1.
    Duplicate records of that same code do not create a second diagnosis.
    PreopRiskFactor has no operation ID/date, so it uses the v2.6 surgery gate.
    """
    raw = {name: tables[name][columns].copy() for name, columns in RAW_COLUMNS.items()}
    for table in raw.values():
        table["_stay"] = table.hospitalizationidNEW.map(_code)
        if table._stay.eq("").any():
            raise ValueError("Raw code tables require nonmissing stay IDs")
    master = raw["IndexSurgHosp"].set_index("_stay")
    if not master.index.is_unique:
        raise ValueError("IndexSurgHosp must have one row per stay")
    index_op = master.operativeidNEW.map(_code)
    book = pd.DataFrame(index=master.index)
    book["adm"] = np.floor(_serial(master.hospadmitdtSHIFT))
    book["surg"] = np.floor(_serial(master.cardsurgdtSHIFT))
    book["icu"] = _serial(master.icupacuadmitdttmSHIFT)
    for field, source in (("fund", "FundDiagnosis"), ("primary_proc", "ProcPrimary")):
        book[field] = master[source].map(lambda v: (_code(v),) if _code(v) else ())

    for name, field, source in (("Surgdiag", "primary_surgdiag", "surgdiag"),
                                ("Procedures", "procedures", "ProcName"),
                                ("PreopRiskFactor", "preop_risk", "preopfactor")):
        table = raw[name]
        if name != "PreopRiskFactor":
            op = table.operativeidNEW.map(_code)
            table = table.loc[op.ne("") & op.eq(table._stay.map(index_op))].copy()
        if name == "Surgdiag":
            table = table.loc[pd.to_numeric(table.SurgDiagPrim, errors="coerce").eq(1)].copy()
        table["_code"] = table[source].map(_code)
        table = table.loc[table._code.ne("")]
        if name == "PreopRiskFactor":
            table = table.loc[~table._code.isin(["320", "330"])]
        sets = table.groupby("_stay", sort=False)._code.agg(lambda s: tuple(sorted(set(s))))
        if name == "Surgdiag":
            sets = sets.map(lambda codes: codes if len(codes) == 1 else ())
        book[field] = [sets.get(stay, ()) for stay in book.index]
    return book


def _asof(frame, book):
    """FundDiagnosis is admission-known (DEC-005); all other fields use v2.6 gates."""
    ids = frame[H].map(_code)
    if ids.eq("").any() or not ids.isin(book.index).all():
        raise ValueError("Prediction stays are missing from IndexSurgHosp")
    data = book.loc[ids].reset_index(drop=True)
    dates = frame.actual_date.to_numpy(dtype=float)
    if not np.isfinite(dates).all():
        raise ValueError("Prediction dates must be finite")
    if "surg" in frame and not np.array_equal(frame.surg.to_numpy(), data.surg.to_numpy(), equal_nan=True):
        raise ValueError("Raw and harness index-surgery dates differ")
    admitted = dates >= data.adm.to_numpy()
    pod = dates - data.surg.to_numpy()
    plausible_icu = (data.icu - data.surg).between(0, 2).to_numpy()
    done_by_eod = np.where(plausible_icu, dates + 1 >= data.icu.to_numpy(), pod >= 1)
    surgical = admitted & (pod >= 0) & ((pod != 0) | done_by_eod)
    # Empty tuples never enter counts; the separate gate restores NaNs on output.
    for field in FIELDS:
        visible = admitted if field == "fund" else surgical
        data[field] = [codes if ok else () for codes, ok in zip(data[field], visible)]
    data["_stay"] = ids.to_numpy()
    data["_admitted"], data["_surgical"] = admitted, surgical
    return data


def _fit_rates(data):
    """One label per stay and one count per stay/code, including rare codes."""
    stays = data[["_stay", "_label"]].drop_duplicates("_stay")
    prior = float(stays._label.mean())
    rates = {}
    for field in FIELDS:
        codes = data[["_stay", "_label", field]].explode(field).dropna(subset=[field])
        codes = codes.drop_duplicates(["_stay", field])
        counts = codes.groupby(field)._label.agg(["sum", "count"])
        rates[field] = ((counts["sum"] + STRENGTH * prior) / (counts["count"] + STRENGTH)).to_dict()
    return prior, rates


def _encode(data, prior, rates):
    result = np.empty((len(data), len(COLUMNS)), dtype=float)
    col = 0
    for field in FIELDS:
        encoded = [[rates[field].get(code, prior) for code in codes] for codes in data[field]]
        if field in FIELDS[:3]:
            result[:, col] = [values[0] if values else prior for values in encoded]
            col += 1
        else:
            result[:, col] = [sum(values) / len(values) if values else prior for values in encoded]
            result[:, col + 1] = [max(values) if values else prior for values in encoded]
            result[:, col + 2] = [len(values) for values in encoded]
            col += 3
    result[~data._admitted.to_numpy(), 0] = np.nan
    result[~data._surgical.to_numpy(), 1:] = np.nan
    return result
