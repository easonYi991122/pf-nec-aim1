"""RX-I1: frozen schemas and label-free input boundaries (no fitting/scoring)."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from . import config
WS = config.CACHE_ROOT
SPEC_SHA256 = "a59449eef27d124fbcb8e1b64d96bc14351c02f28e43c25482b05bf7fdf76639"
H, DATE, SOURCE_ROW = "hospitalizationidnew", "actual_date", "source_row_id"
HARNESS_ROW = "__harness_row_index"
KEYS = (H, DATE)
ANCHORS = (H, "adm", "surg", "icu_serial")
FRAMES = ("PI72-CLEAN", "A-formal")
SOURCE_BANK, D5_BANK = "source_PI_guard", "D5SAFE"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


class ContractError(ValueError):
    """Fail closed; never repair a risk set, feature menu or identity implicitly."""


class ProvenanceError(ContractError):
    """An excluded source-model arm/program must remain incomplete."""


def load_spec():
    payload = (HERE / "spec_v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != "2922c52bce4c3d4a24e41360946cba1bd093a539df242453c3a707069c064286":
        raise ContractError("Frozen RX-D1 specification changed")
    return json.loads(payload)


SPEC = load_spec()
PI10 = tuple(SPEC["feature_sets"]["PI10"]["columns"])
D5SAFE = tuple(SPEC["feature_sets"]["D5SAFE"]["columns"])
ACTION16 = tuple(SPEC["feature_sets"]["ACTION16"]["columns"])
WINDOW_LENGTHS = tuple(SPEC["windows"]["lengths"])


def feature_columns(recipe, *, selected=None, frame="PI72-CLEAN"):
    """Ordered fixed list, or a caller's fold-local GAIN/SHADOW list; never rank."""
    if recipe == "PI10":
        raise ContractError("Source-model feature arms are excluded from this handoff")
    if frame not in FRAMES or recipe not in SPEC["feature_sets"]:
        raise ContractError("Unknown frame/feature recipe")
    if frame == "A-formal" and (recipe == "PI10" or recipe.startswith("SHADOW")):
        raise ContractError("Task A does not use source PI lists or SHADOW")
    entry = SPEC["feature_sets"][recipe]
    if "selector" in entry:
        if selected is None or len(selected) != entry["nominal_k"]:
            raise ContractError("A full fold-local selected list is required")
        columns = tuple(selected)
        assert_predictors(columns, bank=D5_BANK)
    else:
        if selected is not None:
            raise ContractError("A fixed recipe cannot be replaced by selected columns")
        columns = tuple(entry["columns"])
    if len(columns) != len(set(columns)):
        raise ContractError("Duplicate predictor")
    return columns


def _forbidden_name(name):
    low = name.lower()
    exact = {s.lower() for s in SPEC["deny_features"]["exact"]}
    exact |= {"label", "labels", "outcome", "case", "case_label", "case_status",
              "discharge", "death", "mortality", "total_stay_length", "last_date"}
    # Reject common decorated aliases, not just the final model column name.
    if any(re.search(r"(?:^|[^a-z0-9])" + re.escape(s) + r"(?:$|[^a-z0-9])", low)
           for s in exact):
        return True
    return any(p in low for p in SPEC["deny_features"]["prefix_case_insensitive"])


def predictor_roots(columns, *, bank, dependencies=None):
    """Validate the entire explicit dependency DAG and return each input's roots.

    Only registered bank columns can be leaves. Derived/encoded/lagged aliases
    need dependencies, including when used for SHAP, embeddings or weights.
    Availability metadata is deliberately not an allowable model-input root.
    """
    if bank != D5_BANK:
        raise ContractError("Unknown input bank")
    leaves = set(D5SAFE)
    deps = {} if dependencies is None else dependencies
    columns = tuple(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ContractError("Empty or duplicate predictor list")
    memo, visiting = {}, set()

    def walk(name):
        # Registered names such as recorded_active_day are not metadata `day`.
        decorated = name
        if isinstance(name, str):
            for leaf in leaves:
                if name == leaf or name.startswith(leaf + "__"):
                    decorated = name[len(leaf):]
                    break
        if not isinstance(name, str) or _forbidden_name(decorated):
            raise ContractError(f"Forbidden feature dependency: {name}")
        if name in visiting:
            raise ContractError("Feature dependency cycle")
        if name in memo:
            return memo[name]
        if name in deps:
            parents = deps[name]
            if isinstance(parents, str) or not parents:
                raise ContractError("Dependencies must be a nonempty sequence")
            visiting.add(name)
            roots = frozenset().union(*(walk(parent) for parent in parents))
            visiting.remove(name)
        elif name in leaves:
            roots = frozenset((name,))
        else:
            raise ContractError(f"Unregistered feature/dependency: {name}")
        memo[name] = roots
        return roots

    # Extra graph entries cannot hide an outcome dependency off the selected path.
    for name in deps:
        walk(name)
    return {name: walk(name) for name in columns}


def assert_predictors(columns, *, bank, dependencies=None):
    predictor_roots(columns, bank=bank, dependencies=dependencies)


def validate_stay_ids(ids):
    """No rounding/coercion: preserve float64, integer or exact string identities."""
    if ids.isna().any() or ids.dtype == np.dtype("float32"):
        raise ContractError("Missing or float32 hospitalization identity")
    if pd.api.types.is_numeric_dtype(ids.dtype):
        if not np.isfinite(ids.to_numpy()).all():
            raise ContractError("Nonfinite hospitalization identity")
    elif not all(isinstance(v, str) and v.strip() for v in ids):
        raise ContractError("Mixed/non-string object hospitalization identities")


def validate_keys(keys, *, frame="PI72-CLEAN"):
    """Strict metadata projection: only stay/date and the frame's row identity."""
    if frame not in FRAMES:
        raise ContractError("Unknown frame")
    row = SOURCE_ROW if frame == "PI72-CLEAN" else HARNESS_ROW
    required = {H, DATE, row}
    if set(keys.columns) != required or not keys.columns.is_unique:
        raise ContractError("Keys must contain only stay/date/frame row identity")
    validate_stay_ids(keys[H])
    dates = keys[DATE].to_numpy(np.float64)
    if not np.isfinite(dates).all() or not np.equal(dates, np.floor(dates)).all():
        raise ContractError("Calendar keys must be finite integer serial days")
    if keys.duplicated(list(KEYS)).any() or keys[row].isna().any() or keys[row].duplicated().any():
        raise ContractError("Duplicate calendar/row identity or missing row identity")
    if not pd.api.types.is_integer_dtype(keys[row].dtype):
        raise ContractError("Row identities must be exact integer positions")
    return row


def align_anchors(keys, anchors, *, frame="PI72-CLEAN"):
    """Align only admission/surgery/ICU gates; never read discharge or outcomes."""
    validate_keys(keys, frame=frame)
    if set(anchors.columns) != set(ANCHORS) or not anchors.columns.is_unique:
        raise ContractError("Anchors must contain only stay/adm/surg/icu_serial")
    validate_stay_ids(anchors[H])
    if anchors[H].duplicated().any():
        raise ContractError("Duplicate stay anchors")
    if not keys[H].isin(anchors[H]).all():
        raise ContractError("Missing stay anchors; no inner join permitted")
    out = anchors.set_index(H).loc[keys[H]].reset_index(drop=True)
    for c in ("adm", "surg"):
        v = out[c].to_numpy(np.float64)
        if not np.isfinite(v).all() or not np.equal(v, np.floor(v)).all():
            raise ContractError("Admission/surgery must be finite calendar days")
    if (out.surg < out.adm).any():
        raise ContractError("Surgery precedes admission")
    return out


def canonical_arm(recipe, window, representation, learner, *, frame="PI72-CLEAN"):
    """Tree fit ID; w1 aliases reuse the exact PI-T8 or Task-A single-day ID.

    This does not authorize a new menu entry. Sequence profile-slot IDs are
    separately frozen in SPEC['arms'] and are owned by the sequence orchestrator.
    """
    if window not in WINDOW_LENGTHS or representation not in ("R1", "R3"):
        raise ContractError("Unregistered window/representation")
    if recipe not in SPEC["feature_sets"] or frame not in FRAMES:
        raise ContractError("Unknown recipe/frame")
    if frame == "A-formal":
        if window == 1:
            for arm in SPEC["Task_A_arms"]:
                if arm["features"] == recipe and arm["learner"] == learner:
                    return arm["id"]
        elif recipe == "GAIN50" and learner == "LGB":
            return f"A-T4-GAIN50-w{window}-{representation}-LGB"
        raise ContractError("Tree arm is outside the frozen Task A menu")
    arm = f"T8-{recipe}-{learner}" if window == 1 else f"T4-{recipe}-w{window}-{representation}-{learner}"
    if arm not in {entry["id"] for entry in SPEC["arms"]}:
        raise ContractError("Tree arm is outside the frozen PI menu")
    return arm


def reset_threads():
    """Call again after importing core; cap loaded numerical runtimes at two."""
    for name in THREAD_ENV:
        os.environ[name] = "2"
    import torch  # before any downstream XGBoost/LightGBM import
    import pyarrow as pa
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() > 2:
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError as exc:
            raise ContractError("Torch interop pool already started above two threads") from exc
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    threadpool_limits(limits=2)




def import_core():
    reset_threads()
    from .legacy import core
    return core
