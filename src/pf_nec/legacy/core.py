"""Read-only private-cache interface; no PI model/feature-list access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import wp0
from .. import config
H, ROW, LABEL = wp0.H, wp0.ROW, wp0.LABEL
SOURCE_ROW = "source_row_id"
RAW = config.DATA_ROOT / "Raw CSV Files"
PRIVATE = config.CACHE_ROOT / "core"
OLD_CACHE = config.CACHE_ROOT / "v26_rows.parquet"


class ContractError(ValueError):
    """A frozen acceptance condition failed; dependent fitting must stop."""


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def frame_hash(frame):
    h = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    h.update(json.dumps([(c, str(frame[c].dtype)) for c in frame]).encode())
    return h.hexdigest()


def reconcile_source(clean, calendar):
    matched, strata, audit = wp0.reconcile_rows(clean, calendar)
    # Reconcile the untouched source first. DEC-011 excludes missing labels;
    # such rows need no feature row, including placeholders outside the stay.
    # Every observed label must match, with no calendar-anchor disagreement.
    if matched[LABEL].notna().sum() != clean[LABEL].notna().sum() or any(audit["join_disagreements"].values()):
        error = ContractError("Full-calendar reconciliation has missing labelled rows or anchor disagreements")
        error.reconciliation_audit = audit
        raise error
    matched = matched.sort_values(ROW).reset_index(drop=True)
    matched[SOURCE_ROW] = matched[ROW].astype(np.int64)
    matched = matched.merge(calendar[[H, "actual_date", "site"]], on=[H, "actual_date"],
                            how="left", validate="one_to_one", sort=False)
    matched["C_h"] = matched[H].map(strata).astype(np.int8)
    labelled = matched.loc[matched[LABEL].notna()].reset_index(drop=True)
    labelled[LABEL] = labelled[LABEL].astype(np.int8)
    return labelled, strata, audit


def load_source_rows(repeat=None, partition=None):
    if (repeat is None) != (partition is None) or partition not in (None, "train", "test"):
        raise ValueError("Supply both repeat and train/test partition, or neither")
    filters = None if repeat is None else [(H, "in", load_split(repeat)[partition].tolist())]
    return _validated_table("rows.parquet", filters=filters)


def feature_spec():
    return json.loads(Path(__file__).with_name("v26_features.json").read_text())


def d5_columns():
    return feature_spec()["all_features"]


def _validated_table(name, columns=None, filters=None):
    manifest = json.loads((PRIVATE / "manifest.json").read_text())
    if Path(name).name != name or name not in manifest["files"]:
        raise ContractError("Unregistered private cache table")
    path = PRIVATE / name
    if file_hash(path) != manifest["files"][name]["sha256"]:
        raise ContractError("Private cache table changed: " + name)
    return pd.read_parquet(path, columns=columns, filters=filters)


def load_split(repeat=1):
    if type(repeat) is not int or repeat not in range(1, 6):
        raise ContractError("PI repeat must be 1 through 5")
    frame = _validated_table(f"split_{repeat:02d}.parquet")
    return {part: frame.loc[frame.partition.eq(part), H].to_numpy() for part in ("train", "test")}


def load_features(block="D5", source_rows=None):
    from .line_S import COLUMNS
    if block not in ("D5", "S"):
        raise ContractError("Only deterministic D5 and support states are available")
    columns = d5_columns() if block == "D5" else list(COLUMNS)
    filters = None if source_rows is None else [(SOURCE_ROW, "in", list(source_rows))]
    table = _validated_table(f"{block}.parquet", [SOURCE_ROW] + columns, filters).set_index(SOURCE_ROW)
    if not table.index.is_unique:
        raise ContractError("Duplicate private feature row")
    return table if source_rows is None else table.loc[list(source_rows)]
