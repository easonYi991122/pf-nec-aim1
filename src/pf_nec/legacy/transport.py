"""Frozen complete-calendar feature transport, label-free prefix construction."""
import ast
import copy
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from . import core, build_v26 as build, rebuild_v26 as rebuild
H = core.H


@lru_cache(maxsize=1)
def feature_assembler():
    """Extract the unaltered feature/gating/output suffix, failing on API drift."""
    path = Path(build.__file__)
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assemble_rows")
    boundary = [i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "EXCLUDE" for t in n.targets)]
    if len(boundary) != 1 or [a.arg for a in function.args.args] != ["R", "st", "populated"]:
        raise core.ContractError("Feature assembler extraction boundary changed")
    suffix = copy.deepcopy(function.body[boundary[0]:])
    prefix = ast.parse('''
rows = R.merge(st, on=H, how="inner", validate="many_to_one")
rows["doa"] = rows.actual_date - rows.adm
rows["post_surg"] = (rows.actual_date >= rows.surg).astype(int)
rows["pod_raw"] = rows.actual_date - rows.surg
rows["set"] = "pirep"
rows["y3"] = np.nan
rows["lead"] = np.nan
rows["y30"] = np.nan
''').body
    function = copy.deepcopy(function)
    function.name = "assemble_complete_prefix"
    function.body = prefix + suffix
    namespace = dict(build.__dict__)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name]


def calendar_rows(stays):
    """Full admission-through-discharge metadata, including non-formal strata."""
    if stays[H].duplicated().any() or not np.isfinite(stays[[H, "adm", "surg", "dis"]].to_numpy(float)).all():
        raise core.ContractError("Invalid full-calendar stay anchors")
    anchors = stays[["adm", "surg", "dis"]].to_numpy(float)
    if not np.equal(anchors, np.floor(anchors)).all() or stays.dis.lt(stays.adm).any():
        raise core.ContractError("Invalid/noninteger full-calendar bounds")
    spans = (stays.dis - stays.adm + 1).astype(int).to_numpy()
    full = stays.iloc[np.repeat(np.arange(len(stays)), spans)].reset_index(drop=True)
    full["actual_date"] = np.concatenate([np.arange(a, z + 1) for a, z in zip(stays.adm, stays.dis)])
    full["pod_raw"] = full.actual_date - full.surg
    full["post_surg"] = full.pod_raw.ge(0).astype(np.int8)
    return full


def rebuild_prefix(builder, stays, stay_ids):
    """Reconstruct whole trajectories before selecting dates; preserve float64 IDs."""
    matrices = []
    for stay in stay_ids:
        raw_key = rebuild.hid(stay)
        if raw_key not in builder.master.index:
            raise core.ContractError("Requested stay is missing from raw index table")
        _, values, supplement = builder.build(raw_key)
        matrix = pd.DataFrame(values.astype(np.float32), columns=builder.cols)
        # Features are float32, but keys are never rounded through float32.
        matrix[H] = float(stay)
        extras = {c.removeprefix("v26__") if c.startswith("v26__") else "supp_" + c:
                  np.asarray(v, np.float32) for c, v in supplement.items()
                  if c not in ("hid", "day", "actual_date")}
        matrices.append(pd.concat([matrix, pd.DataFrame(extras)], axis=1))
    if not matrices:
        raise ValueError("No requested stays")
    rebuilt = pd.concat(matrices, ignore_index=True)
    populated = [c for c, rule in builder.rules.items() if rule["status"] != "unresolved_not_populated"]
    frame, spec = feature_assembler()(rebuilt, stays.loc[stays[H].isin(stay_ids)], populated)
    columns = core.d5_columns()
    if spec["all_features"] != columns:
        raise core.ContractError("Reconstructed feature schema/order differs from registered D5")
    if frame.duplicated([H, "actual_date"]).any() or np.isinf(frame[columns].to_numpy()).any():
        raise core.ContractError("Duplicate calendar key or infinite feature")
    return frame[[H, "actual_date"] + columns]


def select_requested(complete, requested):
    keys = [H, "actual_date"]
    if complete.duplicated(keys).any() or requested.duplicated(keys).any():
        raise core.ContractError("Duplicate requested/calendar key")
    result = requested[[core.SOURCE_ROW] + keys].merge(complete, on=keys, how="left",
                                                     validate="one_to_one", sort=False, indicator=True)
    if not result._merge.eq("both").all():
        raise core.ContractError("Incomplete D5 transport; no row deletion or replacement allowed")
    return result.drop(columns="_merge")


def transport_batches(builder, stays, requested, batch_stays=128):
    """Yield bounded-memory private feature blocks; caller owns artifact writing."""
    ids = np.sort(requested[H].unique())
    if not np.isin(ids, stays[H]).all():
        raise core.ContractError("Requested stay lacks calendar metadata")
    for start in range(0, len(ids), batch_stays):
        subset = ids[start:start + batch_stays]
        full = rebuild_prefix(builder, stays, subset)
        yield select_requested(full, requested.loc[requested[H].isin(subset)])


def observed_prefix(raw, cutoffs):
    """Erase unavailable dated observations, retaining observed interval starts.

    Admission/surgery/discharge anchors, static code dictionaries and the
    registered ICU availability gates are held fixed. Multi-event records are
    masked field by field, so an earlier event on the same record is preserved.
    """
    point_dates = {table: fields[0] for table, fields in rebuild.EVENTS.items()}
    point_dates.update(AllOperations="cardsurgdtSHIFT", Catheterizations="cardcathdtSHIFT",
                       RiskSurgVIS="RSVISdttmSHIFT")
    changed, audit = {}, {"removed_future_records": 0, "erased_future_dates": 0}
    for name, source in raw.items():
        frame = source.loc[source.hid.isin(cutoffs.index)].copy()
        limits = frame.hid.map(cutoffs) + 1
        start = point_dates.get(name)
        if name in rebuild.INTERVALS:
            start = rebuild.INTERVALS[name][0]
        if start:
            future = frame[start + "__serial"].ge(limits)
            audit["removed_future_records"] += int(future.sum())
            frame = frame.loc[~future].copy()
            limits = limits.loc[~future]
        if name in rebuild.INTERVALS:
            dates = [rebuild.INTERVALS[name][1]]
        elif name in ("Therapies", "Complications"):
            dates = [c[:-8] for c in frame if c.endswith("__serial")]
        else:
            dates = []
        for date in dates:
            future = frame[date + "__serial"].ge(limits)
            audit["erased_future_dates"] += int(future.sum())
            frame.loc[future, date] = ""
            frame.loc[future, date + "__serial"] = np.nan
        changed[name] = frame
    return changed, audit
