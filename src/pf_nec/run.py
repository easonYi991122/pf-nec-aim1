import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gc
from functools import lru_cache
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import uuid


import numpy as np
import pandas as pd
from . import contract as c, evaluate as ev, features as f, selectors as sel, trees, config


class IncompleteJob(RuntimeError):
    """Missing dependency or resource limit; never shrink the frozen design."""


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # pid in the temp name: concurrent shards share run-level files (controller fix 2026-10-04)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def versions():
    from importlib.metadata import version
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "torch", "scikit-learn", "lightgbm", "pyarrow"):
        result[name] = version(name)
    return result


def freeze_versions(directory):
    path = Path(directory) / "versions.json"
    current = versions()
    if path.exists() and json.loads(path.read_text()) != current:
        raise IncompleteJob("Runtime package versions changed; no silent upgrade/resume")
    write_json(path, current)


class ResourceGuard:
    def __init__(self, cache, *, wall_seconds=8 * 3600, rss_limit=8 * 2**30,
                 cache_limit=2 * 2**30, budget_directory=None):
        self.cache = Path(cache)
        self.start = time.monotonic()
        self.wall_seconds, self.rss_limit, self.cache_limit = wall_seconds, rss_limit, cache_limit
        self.peak = 0
        self.checkpoint_lock = threading.Lock()
        self.budget_directory = None if budget_directory is None else Path(budget_directory)
        self.cpu_start = cpu_seconds()
        self.prior_cpu = 0.
        if self.budget_directory is not None:
            for path in self.budget_directory.rglob("receipt.json"):
                receipt = json.loads(path.read_text())
                if receipt.get("charge_cpu"):
                    self.prior_cpu += receipt.get("cpu_seconds", 0.)
            budget = self.budget_directory / "resource_budget.json"
            if budget.exists():
                self.prior_cpu = max(self.prior_cpu, json.loads(budget.read_text())["cpu_seconds"])

    def checkpoint(self):
        if self.budget_directory is not None:
            with self.checkpoint_lock:
                write_json(self.budget_directory / "resource_budget.json",
                           {"cpu_seconds": self.prior_cpu + cpu_seconds() - self.cpu_start,
                            "peak_rss_bytes": self.peak, "updated": utc()})

    def __call__(self, *, allocation=0):
        # stdlib peak RSS: bytes on Mac, KiB on Linux; no package installation.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        rss = current_rss(peak)
        self.peak = max(self.peak, peak, rss)
        if self.peak > self.rss_limit or rss + allocation > self.rss_limit:
            raise IncompleteJob("RSS/allocation exceeds configured cap; checkpoint and stop")
        if time.monotonic() - self.start > self.wall_seconds:
            raise IncompleteJob("Job wall time cap reached")
        size = directory_bytes(self.cache)
        if size > self.cache_limit:
            raise IncompleteJob("Persistent private cache exceeds configured cap")
        if self.prior_cpu + cpu_seconds() - self.cpu_start > c.SPEC["compute"]["CPU_hours_cap"] * 3600:
            raise IncompleteJob("120 CPU-hour budget reached; no menu/repeat reduction")
        if self.budget_directory is not None:
            retained = directory_bytes(self.budget_directory)
            if retained + size > c.SPEC["compute"]["retained_total_GiB"] * 2**30:
                raise IncompleteJob("New retained outputs exceed 6 GiB")


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


@lru_cache(maxsize=1)
def _mach_rss_reader():
    import ctypes

    class TaskInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("virtual_size", "resident_size", "resident_size_max")]
        _fields_ += [(name, ctypes.c_int32) for name in ("user_seconds", "user_usec", "system_seconds",
                                                       "system_usec", "policy", "suspend_count")]

    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.mach_task_self.restype = ctypes.c_uint32
    library.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]

    def read():
        info, count = TaskInfo(), ctypes.c_uint32(ctypes.sizeof(TaskInfo) // 4)
        if library.task_info(library.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count)) != 0:
            raise OSError("mach task_info failed")
        return int(info.resident_size)

    return read


def current_rss(fallback):
    try:
        if sys.platform == "darwin":
            return _mach_rss_reader()()
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return fallback  # conservative stdlib peak fallback, never disable cap


def directory_bytes(directory):
    # os.walk tolerates directories vanishing mid-scan (another shard's
    # tempdir cleanup, controller fix 2026-10-04); rglob raised there.
    size = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                size += os.stat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass  # atomic .partial -> final rename during a resource poll
    return size


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []
        # A crash leaves an attempted fit visible and chargeable.
        for record in self.records:
            if record["status"] == "running":
                record.update(status="incomplete", end=utc(), error="interrupted previous process")
        self.save()

    def save(self):
        write_json(self.path, self.records)

    def recover(self, attempt_id):
        matches = [record for record in self.records if record["attempt_id"] == attempt_id]
        if len(matches) != 1:
            raise IncompleteJob("Checkpoint has no unique fit ledger attempt")
        record = matches[0]
        if record["status"] != "completed":
            record.update(status="completed", end=utc(), recovered_from_complete_checkpoint=True)
            record.pop("error", None)
            self.save()

    @contextmanager
    def fit(self, key, *, arm, kind):
        if any(r["fit_key"] == key and r["status"] == "completed" for r in self.records):
            raise c.ContractError("Successful canonical fit must be resumed, not fitted twice")
        record = {"attempt_id": uuid.uuid4().hex, "fit_key": key, "arm": arm,
                  "kind": kind, "status": "running", "start": utc()}
        self.records.append(record)
        self.save()
        try:
            yield record
        except BaseException as exc:
            record.update(status="incomplete" if isinstance(exc, (IncompleteJob, KeyboardInterrupt)) else "failed",
                          end=utc(), error=f"{type(exc).__name__}: {exc}")
            self.save()
            raise
        else:
            record.update(status="completed", end=utc())
            self.save()


class Jobs:
    """One lean receipt per job; checkpoints carry content bindings, no hashes registry."""
    def __init__(self, directory, *, guard=None, gate=None, charge_cpu=True, ledger=None):
        self.directory = Path(directory)
        self.guard, self.gate = guard, gate
        self.charge_cpu = charge_cpu
        self.ledger = ledger

    def execute(self, name, binding, work, *, big=True):
        directory = self.directory / name
        receipt_path, result_path = directory / "receipt.json", directory / "result.json"
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous["binding"] != binding:
                raise c.ContractError("Checkpoint binding changed; use a new run directory")
            if previous["status"] == "completed":
                if not result_path.exists():
                    raise IncompleteJob("Completed receipt has no checkpoint result")
                result = json.loads(result_path.read_text())
                validate_checkpoint_outputs(result)
                return result
            # A native fit may finish and publish its complete checkpoint just
            # before a process is interrupted while committing the receipt.
            if result_path.exists() and self.ledger is not None:
                result = json.loads(result_path.read_text())
                if result.get("fit_attempt_id") and all(Path(path).exists() for path in result.get("output_paths", [])):
                    if self.guard:
                        self.guard()
                    self.ledger.recover(result["fit_attempt_id"])
                    previous.update(status="completed", exit_code=0, end=utc(), recovered_from_checkpoint=True,
                                    rows=result.get("rows"), columns=result.get("input_columns", result.get("columns")),
                                    output_paths=result.get("output_paths", []))
                    previous.pop("error", None)
                    write_json(receipt_path, previous)
                    return result
        started, cpu_start = time.monotonic(), cpu_seconds()
        receipt = {"job": name, "command": [sys.executable, *sys.argv], "binding": binding,
                   "status": "running", "exit_code": None, "start": utc(), "end": None,
                   "rows": None, "columns": None, "output_paths": [], "charge_cpu": self.charge_cpu}
        write_json(receipt_path, receipt)
        stop = threading.Event()

        def monitor():
            while not stop.wait(.25):
                try:
                    self.guard()
                except IncompleteJob as exc:
                    receipt.update(status="incomplete", exit_code=3, end=utc(), error=str(exc),
                                   peak_rss_bytes=self.guard.peak, cpu_seconds=cpu_seconds() - cpu_start,
                                   wall_seconds=time.monotonic() - started)
                    write_json(receipt_path, receipt)
                    self.guard.checkpoint()
                    # Native learner calls need not return to Python to stop.
                    os._exit(3)

        watcher = None
        try:
            if big and self.gate:
                self.gate(directory)
            if self.guard:
                self.guard()
                watcher = threading.Thread(target=monitor, daemon=True)
                watcher.start()
            result = work(directory)
            if self.guard:
                self.guard()
            write_json(result_path, result)
            receipt.update(status="completed", exit_code=0, rows=result.get("rows"),
                           columns=result.get("input_columns", result.get("columns")), output_paths=result.get("output_paths", []))
        except BaseException as exc:
            receipt.update(status="incomplete" if isinstance(exc, (IncompleteJob, KeyboardInterrupt)) else "failed",
                           exit_code=3 if isinstance(exc, IncompleteJob) else 1,
                           error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            if watcher:
                watcher.join(timeout=1)
            receipt["end"] = utc()
            receipt["wall_seconds"] = time.monotonic() - started
            receipt["cpu_seconds"] = cpu_seconds() - cpu_start
            if self.guard:
                receipt["peak_rss_bytes"] = self.guard.peak
                self.guard.checkpoint()
            write_json(receipt_path, receipt)
        return result


def validate_checkpoint_outputs(value):
    if isinstance(value, dict):
        for path in value.get("output_paths", []):
            if not Path(path).exists():
                raise IncompleteJob("Checkpoint output missing; no silent refit")
        for name in ("model_path", "prediction_path"):
            if name in value and not Path(value[name]).is_file():
                raise IncompleteJob(f"Checkpoint {name} missing; no silent refit")
        for path in value.get("checkpoint_manifests", []):
            if not Path(path).is_file():
                raise IncompleteJob("Checkpoint manifest missing")
            validate_checkpoint_outputs(json.loads(Path(path).read_text()))
        for name, child in value.items():
            if name != "checkpoint_manifests":
                validate_checkpoint_outputs(child)
    elif isinstance(value, list):
        for child in value:
            validate_checkpoint_outputs(child)


@dataclass(frozen=True)
class Arm:
    recipe: str
    learner: str = "LGB"
    window: int = 1
    representation: str = "R1"
    frame: str = "PI72-CLEAN"

    @property
    def id(self):
        return c.canonical_arm(self.recipe, self.window, self.representation, self.learner, frame=self.frame)

    @property
    def k(self):
        return c.SPEC["feature_sets"][self.recipe]["nominal_k"]

    @property
    def columns(self):
        return trees.input_columns(self.k, self.window, self.representation)


def arm_from_id(identity, frame="PI72-CLEAN"):
    entries = c.SPEC["arms"] if frame == "PI72-CLEAN" else c.SPEC["Task_A_arms"]
    for entry in entries:
        if entry["id"] == identity:
            return Arm(entry["features"], entry["learner"], entry.get("window", 1),
                       "R1" if entry.get("representation") in (None, "single_day") else entry["representation"], frame)
    if frame == "A-formal" and identity.startswith("A-T4-GAIN50-w"):
        pieces = identity.split("-")
        return Arm("GAIN50", "LGB", int(pieces[3][1:]), pieces[4], frame)
    raise c.ContractError("Unknown frozen tree arm")


def unique_arms(arms):
    return list({arm.id: arm for arm in arms}.values())


def choose(scores, arms, *, program="T4-SELECT-TREE"):
    arms = unique_arms(arms)
    if set(scores) != {arm.id for arm in arms} or not all(np.isfinite(v) for v in scores.values()):
        raise IncompleteJob("Selection requires complete finite frozen menu; no candidate dropping")
    best = max(scores.values())
    tied = [arm for arm in arms if best - scores[arm.id] <= .002]
    if program.startswith("T8-"):
        return min(tied, key=lambda a: (a.k, 2 if a.recipe.startswith("SHADOW") else
                                        1 if a.recipe.startswith("GAIN") else 0, a.id.encode()))
    if program.startswith("A-"):
        return min(tied, key=lambda a: (a.k, a.window, a.columns, a.id.encode()))
    return min(tied, key=lambda a: (a.k, a.window, a.columns, a.learner != "LGB",
                                   a.representation != "R3", a.id.encode()))


def context_targets(targets, context, partition):
    """Validate one context; full five-fold OOF is checked only after assembly."""
    if context.frame == "PI72-CLEAN":
        return ev.validate_targets(targets, context.frame)
    required = ev.KEY + ev.META + ["fold"]
    if not set(required) <= set(targets) or targets[required].isna().any().any() or targets.empty:
        raise c.ContractError("Incomplete Task A context metadata")
    c.validate_stay_ids(targets.stay)
    if targets.row_key.dtype == np.float32 or targets.duplicated(ev.KEY).any():
        raise c.ContractError("Invalid Task A context row identity")
    if not targets.y.isin([0, 1]).all() or not targets.case.isin([0, 1]).all():
        raise c.ContractError("Task A context label/stratum invalid")
    if ((targets.y == 1) & (targets.case == 0)).any() or targets.groupby("stay").case.nunique().gt(1).any():
        raise c.ContractError("Task A source case stratum changed")
    if not np.isfinite(targets.pod).all() or not targets.pod.eq(np.floor(targets.pod)).all():
        raise c.ContractError("Task A context POD must be integer calendar days")
    if targets.duplicated(["repeat", "stay", "pod"]).any() or context.repeat not in range(5):
        raise c.ContractError("Task A context calendar keys/repeat invalid")
    if not targets.repeat.eq(context.repeat).all() or targets.groupby("stay").fold.nunique().gt(1).any():
        raise c.ContractError("Task A context repeat/stay folds changed")
    if not targets.fold.isin(range(5)).all() or not (targets.fold.eq(context.fold) if partition == "test"
                                                    else targets.fold.ne(context.fold)).all():
        raise c.ContractError("Task A context outer partition changed")
    return targets.reset_index(drop=True).copy()


class TreeEngine:
    """An outer *training* context. Test inputs are requested only in apply().

    loader(base_recipe, partition) -> FeatureRows. metadata(partition) returns
    the separate evaluation schema. Neither is called with test in inner().
    """
    def __init__(self, context, loader, metadata, directory, *, guard=None, gate=None):
        self.context, self.loader, self.metadata = context, loader, metadata
        self.directory, self.guard = Path(directory), guard
        self.ledger = Ledger(self.directory / "fit_ledger.json")
        self.jobs = Jobs(self.directory / "jobs", guard=guard, gate=gate, ledger=self.ledger)
        self.rankings = sel.RankingCache(self.directory / "rankings")
        self.train_meta = context_targets(metadata("train"), context, "train")
        self.labels = self.train_meta.set_index("row_key").y

    def base(self, recipe, partition="train"):
        base = "D5SAFE" if recipe.startswith(("GAIN", "SHADOW")) or recipe == "ACTION16" else recipe
        rows = self.loader(base, partition)
        if rows.frame != self.context.frame:
            raise c.ContractError("Feature frame differs from outer context; no cross-frame join")
        meta = self.train_meta if partition == "train" else self.metadata("test")
        identity = c.validate_keys(rows.keys, frame=rows.frame)
        if set(rows.keys[identity]) != set(meta.row_key) or len(rows.keys) != len(meta):
            raise c.ContractError("Feature/target key coverage mismatch; never inner join")
        rows = f.take_rows(rows, meta.row_key)
        if not np.array_equal(rows.keys[c.H], meta.stay) or not np.array_equal(rows.gates.post_surg, meta.pod.ge(0)):
            raise c.ContractError("Feature/target stay or phase mismatch")
        return rows

    def selected(self, rows, recipe, labels, context, arm):
        columns = self.rankings.select(recipe, rows, labels, context,
                                       ledger=self.ledger, arm=arm, guard=self.guard)
        return f.select_features(rows, columns)

    def inner(self, arm):
        rows = self.base(arm.recipe)
        y = sel.aligned_labels(rows, self.labels)
        binding = sel.pool_binding(rows, y)
        folds = list(sel.inner_folds(rows.keys, y, self.context))
        results = []
        identity = c.validate_keys(rows.keys, frame=rows.frame)
        for inner, (fit_indices, val_indices) in enumerate(folds):
            context = replace(self.context, inner=inner, pool="inner-fit")

            def work(directory, fit_indices=fit_indices, val_indices=val_indices, context=context):
                child = f.take_rows(rows, rows.keys[identity].iloc[fit_indices])
                labels = self.labels.reindex(child.keys[identity])
                child = self.selected(child, arm.recipe, labels, context, arm.id)
                validation = f.select_features(f.take_rows(rows, rows.keys[identity].iloc[val_indices]), child.values.columns)
                train = trees.matrix(child, window=arm.window, representation=arm.representation, guard=self.guard)
                valid = trees.matrix(validation, window=arm.window, representation=arm.representation, guard=self.guard)
                with self.ledger.fit(f"{context.key}/{binding}/{arm.id}", arm=arm.id, kind="inner") as record:
                    model = trees.fit_tree(train, y[fit_indices], learner=arm.learner, context=context, arm=arm.id,
                                           valid=valid, valid_y=y[val_indices],
                                           valid_post=self.train_meta.pod.iloc[val_indices].ge(0), guard=self.guard)
                    score = trees.primary_auc(y[val_indices], model.predict(valid.values),
                                              self.train_meta.pod.iloc[val_indices].ge(0), context.frame)
                    record.update(rows=len(fit_indices), columns=len(train.columns), rounds=model.rounds,
                                  selected=list(child.values.columns))
                    result = {"score": score, "rounds": model.rounds, "selected": list(child.values.columns),
                              "rows": len(fit_indices), "validation_rows": len(val_indices),
                              "columns": len(train.columns), "output_paths": [], "fit_attempt_id": record["attempt_id"]}
                    # Publish checkpoint before committing the successful fit.
                    write_json(directory / "result.json", result)
                return result

            results.append(self.jobs.execute(f"inner/{arm.id}/i{inner}", binding, work))
        del rows
        gc.collect()
        return {"arm": arm.id, "score": float(np.mean([r["score"] for r in results])),
                "rounds": trees.median_rounds([r["rounds"] for r in results]), "inner": results,
                "binding": binding}

    def fit(self, arm, inner_result=None):
        # Both aliases map to the same inner jobs and final model job.
        result = self.inner(arm) if inner_result is None else inner_result
        if result["arm"] != arm.id:
            raise c.ContractError("Stopping result belongs to another arm")
        rows = self.base(arm.recipe)
        y = sel.aligned_labels(rows, self.labels)
        binding = sel.pool_binding(rows, y)
        if binding != result["binding"]:
            raise c.ContractError("Outer training pool changed after inner selection")
        context = replace(self.context, inner=-1, pool="outer")

        def work(directory):
            child = self.selected(rows, arm.recipe, self.labels, context, arm.id)
            train = trees.matrix(child, window=arm.window, representation=arm.representation, guard=self.guard)
            with self.ledger.fit(f"{context.key}/{binding}/{arm.id}", arm=arm.id, kind="outer") as record:
                model = trees.fit_tree(train, y, learner=arm.learner, context=context, arm=arm.id,
                                       rounds=result["rounds"], guard=self.guard)
                path = directory / ("model.txt" if arm.learner == "LGB" else "model.json")
                model.save(path)
                record.update(rows=len(y), columns=len(train.columns), rounds=model.rounds,
                              selected=list(child.values.columns))
                info = {"arm": arm.id, "learner": arm.learner, "rounds": model.rounds,
                        "parameters": model.parameters, "selected": list(child.values.columns),
                        "columns": list(train.columns), "rows": len(y), "input_columns": len(train.columns),
                        "model_path": str(path.resolve()), "output_paths": [str(path.resolve())],
                        "binding": binding, "inner_score": result["score"], "fit_attempt_id": record["attempt_id"]}
                write_json(directory / "result.json", info)
            return info

        info = self.jobs.execute(f"outer/{arm.id}", binding, work)
        del rows
        gc.collect()
        return info

    def apply(self, arm, info=None, *, name=None):
        info = self.fit(arm) if info is None else info
        if info["arm"] != arm.id:
            raise c.ContractError("Apply model identity mismatch")
        rows = f.select_features(self.base(arm.recipe, "test"), info["selected"])
        targets = context_targets(self.metadata("test"), self.context, "test")
        # Apply cache may change with X/y/metadata; it cannot change a fit job.
        binding = sel.pool_binding(rows, sel.aligned_labels(rows, targets.set_index("row_key").y))
        binding += ":" + info["binding"] + ":" + str(info["rounds"])

        def work(directory):
            model = trees.load_fit(info["model_path"], info)
            probability = np.empty(len(rows.keys), dtype=np.float64)
            offset = 0
            from . import windows
            for batch in windows.iter_windows(rows, window=arm.window):
                if self.guard:
                    self.guard()
                matrix = windows.tree_matrix(batch, arm.representation)
                if tuple(matrix.columns) != tuple(info["columns"]):
                    raise c.ContractError("Apply layout differs from fitted model")
                probability[offset:offset + len(batch.keys)] = model.predict(matrix.values)
                offset += len(batch.keys)
            if offset != len(rows.keys):
                raise c.ContractError("Incomplete streamed apply coverage")
            prediction = targets.assign(probability=probability)
            if self.context.frame == "PI72-CLEAN":
                ev.align_predictions(targets, {arm.id: prediction}, frame=self.context.frame)
            elif not prediction.drop(columns="probability").equals(targets):
                raise c.ContractError("Task A apply metadata/coverage changed")
            path = directory / "predictions.parquet"
            prediction.to_parquet(path, index=False)
            return {"arm": arm.id, "model_path": info["model_path"], "selected": info["selected"],
                    "rows": len(targets), "columns": len(info["columns"]), "complete_key_coverage": True,
                    "prediction_path": str(path.resolve()), "output_paths": [str(path.resolve())]}

        return self.jobs.execute(f"apply/{name or arm.id}", binding, work)

    def menu(self, arms, *, program):
        arms = unique_arms(arms)
        inner = {arm.id: self.inner(arm) for arm in arms}
        scores = {identity: result["score"] for identity, result in inner.items()}
        chosen = choose(scores, arms, program=program)
        write_json(self.directory / "choices" / f"{program}.json",
                   {"program": program, "selected": chosen.id, "scores": scores,
                    "selection_scope": "outer-training-only", "outer_context": self.context.key})
        return chosen, inner


def disk_gate(directory):
    """Local free-space gate only; no dispatcher, remote probe or network."""
    import shutil
    config.writable(directory)
    free = shutil.disk_usage(config.CACHE_ROOT).free
    if free < 2 * 2**30:
        raise IncompleteJob("At least 2 GiB free private-cache disk space is required")


def pi_provider(context, *, provenance=None):
    core = c.import_core()
    config.initialize()
    def metadata(partition):
        original = core.load_source_rows(context.repeat, partition)
        return pd.DataFrame({"repeat": context.repeat, "row_key": original[c.SOURCE_ROW],
            "stay": original[c.H], "y": original[core.LABEL], "pod": original.day,
            "case": original.C_h}).reset_index(drop=True)
    def loader(recipe, partition):
        if recipe != "D5SAFE":
            raise c.ContractError("The export supports only the independent D5 source bank")
        original = core.load_source_rows(context.repeat, partition)
        keys = original[[c.H, c.DATE, c.SOURCE_ROW]].copy()
        anchors = core._validated_table("stays.parquet", columns=list(c.ANCHORS),
                                        filters=[(c.H, "in", keys[c.H].unique().tolist())])
        values = core._validated_table("D5.parquet", columns=[c.SOURCE_ROW, *c.D5SAFE],
                                      filters=[(c.SOURCE_ROW, "in", keys[c.SOURCE_ROW].tolist())])
        return f.build_features(keys, values.set_index(c.SOURCE_ROW), anchors, bank=c.D5_BANK)
    return loader, metadata


def task_a_provider(context):
    core = c.import_core()
    from .legacy.common import load_frames
    from .formal_ids import _formal_ids
    from .legacy.rebuild_v26 import serial
    frames, population_hash = load_frames(task="A", data="v2.6", columns=list(c.D5SAFE))
    frame = _formal_ids(frames["A"], core)
    if (len(frame), frame[c.H].nunique(), int(frame.y3.sum()), frame.loc[frame.y3.eq(1), c.H].nunique()) != (301468, 11674, 1066, 359):
        raise c.ContractError("Task A formal population differs from frozen contract")
    raw_columns = ["hospitalizationidNEW", "hospadmitdtSHIFT", "cardsurgdtSHIFT", "icupacuadmitdttmSHIFT"]
    anchors = pd.read_csv(config.DATA_ROOT / "Raw CSV Files/IndexSurgHosp.csv", usecols=raw_columns,
                         dtype=str, encoding="cp1252", keep_default_na=False)
    anchors = anchors.rename(columns=dict(zip(raw_columns, c.ANCHORS)))
    anchors[c.H] = pd.to_numeric(anchors[c.H], errors="raise").astype(np.float64)
    anchors = anchors.loc[anchors[c.H].isin(frame[c.H].unique())].copy()
    for name in ("adm", "surg", "icu_serial"):
        anchors[name] = anchors[name].map(serial)
        if name != "icu_serial":
            anchors[name] = np.floor(anchors[name])
    if anchors[c.H].duplicated().any() or not frame[c.H].isin(anchors[c.H]).all():
        raise c.ContractError("Task A index anchor keys missing/duplicated")
    if not frame.surg.eq(frame[c.H].map(anchors.set_index(c.H).surg)).all():
        raise c.ContractError("Task A raw and formal surgery days differ")
    test = frame[f"fold_r{context.repeat}"].eq(context.fold)
    case = frame.groupby(c.H).y3.max()
    def pool(partition):
        return frame.loc[test if partition == "test" else ~test].reset_index(drop=True)
    def metadata(partition):
        rows = pool(partition)
        return pd.DataFrame({"repeat": context.repeat, "row_key": rows[c.HARNESS_ROW],
            "stay": rows[c.H], "y": rows.y3, "pod": rows.pod_raw,
            "case": rows[c.H].map(case), "fold": rows[f"fold_r{context.repeat}"]})
    def loader(recipe, partition):
        if recipe != "D5SAFE":
            raise c.ContractError("Task A uses only its own D5 source")
        return f.from_task_a(pool(partition), anchors)
    return loader, metadata, population_hash
