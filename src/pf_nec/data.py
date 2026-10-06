"""Rebuild private frames from authorized clean/raw inputs, without old caches.

The clean-category mappings are empirical label-free code crosswalks. They are
regenerated locally and never distributed with this repository. Clinical NEC
labels do not enter mapping inference or the feature assembly suffix.
"""
from collections import defaultdict
import gc
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, contract as c
from .legacy import core, wp0, build_v26 as build, rebuild_v26 as rebuild, transport, line_R, line_S


def mapping_inputs(clean, raw):
    """Reproduce the old empirical OR/group mapping algorithm, no dictionary copy."""
    first = clean.groupby(c.H, sort=False).first()
    master = raw["IndexSurgHosp"].copy()
    master.index = pd.to_numeric(master.hospitalizationidNEW, errors="raise")
    if not master.index.is_unique or not first.index.isin(master.index).all():
        raise ValueError("Clean/raw mapping identity mismatch")
    definitions = {}
    for prefix, field in (("funddiagnosis_", "FundDiagnosis"), ("procprimary_", "ProcPrimary")):
        for col in (n for n in clean if n.startswith(prefix)):
            codes = sorted(set(master.loc[first.index[first[col].eq(1)], field]))
            definitions[col] = {"members": [{"code": value} for value in codes]}
    risks = defaultdict(set)
    for row in raw["PreopRiskFactor"][["hospitalizationidNEW", "preopfactor"]].itertuples(index=False):
        risks[float(row[0])].add(row[1])
    by_code = defaultdict(set)
    for i, hid in enumerate(first.index):
        for code in risks[hid]:
            if code.isdigit():
                by_code[code].add(i)
    groups = {}
    for col in (n for n in clean if n.startswith("PreopRiskFactor_")):
        positives = set(np.flatnonzero(first[col].eq(1)))
        candidates = sorted((code for code, rows in by_code.items() if rows and rows <= positives), key=int)
        target = set().union(*(by_code[code] for code in candidates))
        if target != positives:
            raise ValueError("Empirical risk-code crosswalk no longer reproduces clean flags")
        covers = []
        for size in range(len(candidates) + 1):
            for subset in itertools.combinations(candidates, size):
                if set().union(*(by_code[code] for code in subset)) == target:
                    covers.append(list(subset))
            if covers:
                break
        if len(covers) != 1:
            raise ValueError("Risk-code mapping is ambiguous; clinical/source adjudication required")
        groups[col] = {"minimal_compatible_code_sets": covers}
    return definitions, {"risk_minimum_compatible_groups": groups}


class TableWriter:
    def __init__(self, path):
        self.path = config.writable(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_suffix(self.path.suffix + ".partial")
        self.writer = None
        self.rows = 0

    def write(self, frame):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temp, table.schema, compression="zstd")
        if not table.schema.equals(self.writer.schema, check_metadata=False):
            raise ValueError("Private frame schema changed between batches")
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self, success):
        if self.writer is not None:
            self.writer.close()
        if success:
            self.temp.replace(self.path)


def build_frames(*, batch_stays=128, guard=None):
    config.require_data()
    config.initialize()
    c.reset_threads()
    if batch_stays < 1 or batch_stays > 256:
        raise ValueError("batch_stays must be 1 through 256")
    if (core.PRIVATE / "manifest.json").exists():
        raise FileExistsError("Private frames already built; use a fresh PF_CACHE_ROOT for rebuild")
    started = time.monotonic()
    clean_path = config.DATA_ROOT / "NEC Cleaned Data.csv"
    raw_dir = config.DATA_ROOT / "Raw CSV Files"
    inputs = {p.name: core.file_hash(p) for p in [clean_path] + sorted(raw_dir.glob("*.csv"))}
    if len(inputs) != 21:
        raise ValueError("Expected one clean file and twenty raw tables")
    header = pd.read_csv(clean_path, nrows=0).columns.tolist()
    fields = [n for n in header if n.startswith(("funddiagnosis_", "procprimary_", "PreopRiskFactor_"))]
    clean = pd.read_csv(clean_path, usecols=[c.H, *fields], low_memory=False)
    raw = rebuild.read_raw(raw_dir)
    definitions, checks = mapping_inputs(clean, raw)
    del clean
    builder = rebuild.Rebuilder(raw=raw, cols=header, defs=definitions, addchecks=checks)
    mapping_path = config.CACHE_ROOT / "reconstruction_mappings.json"
    mapping_path.write_text(json.dumps({"definitions": definitions, "checks": checks}, sort_keys=True) + "\n")
    source = pd.read_csv(clean_path, usecols=[c.H, "cardsurgdtshift_index", "day", core.LABEL, "necbelldtshift"], low_memory=False)
    idx = pd.read_csv(raw_dir / "IndexSurgHosp.csv", encoding="cp1252", low_memory=False, na_values=["."])
    nec = pd.read_csv(raw_dir / "NEC.csv", encoding="cp1252", low_memory=False, na_values=["."])
    risk = pd.read_csv(raw_dir / "PreopRiskFactor.csv", encoding="cp1252", low_memory=False, na_values=["."])
    stays_all = build.build_stays(idx, nec, risk, set(source[c.H]))
    stays = stays_all.loc[stays_all[c.H].isin(source[c.H])].reset_index(drop=True)
    calendar = transport.calendar_rows(stays)
    rows, strata, audit = core.reconcile_source(source, calendar)
    source[c.SOURCE_ROW] = np.arange(len(source), dtype=np.int64)
    source[c.DATE] = source.cardsurgdtshift_index + source.day
    source_metadata = source.merge(stays, on=c.H, how="left", validate="many_to_one", sort=False)
    if (len(rows), len(strata), int(strata.sum()), int(rows[core.LABEL].sum())) != (318992, 11931, 356, 356):
        raise c.ContractError("PI72-CLEAN population changed")
    del calendar, idx, nec, risk
    gc.collect()
    core.PRIVATE.mkdir(parents=True, exist_ok=True)
    formal_writer = TableWriter(config.CACHE_ROOT / "v26_rows.parquet")
    pi_writer = TableWriter(core.PRIVATE / "D5.parquet")
    support_writer = TableWriter(core.PRIVATE / "S.parquet")
    intervals, _ = line_S.load_intervals(raw_dir)
    all_ids = np.sort(stays_all[c.H].to_numpy())
    success = False
    try:
        for start in range(0, len(all_ids), batch_stays):
            if guard:
                guard()
            ids = all_ids[start:start + batch_stays]
            matrices = []
            for hid in ids:
                _, values, supplement = builder.build(rebuild.hid(hid))
                matrix = pd.DataFrame(values.astype(np.float32), columns=header)
                extra = {name.removeprefix("v26__") if name.startswith("v26__") else "supp_" + name:
                         np.asarray(value, np.float32) for name, value in supplement.items()
                         if name not in ("hid", "day", "actual_date")}
                matrix = pd.concat([matrix, pd.DataFrame(extra)], axis=1)
                matrix["_exact_hid"] = hid
                matrices.append(matrix)
            rebuilt = pd.concat(matrices, ignore_index=True)
            populated = [name for name, rule in builder.rules.items() if rule["status"] != "unresolved_not_populated"]
            batch_anchors = stays_all.loc[stays_all[c.H].isin(ids)]
            # Preserve the frozen formal backing-cache identity representation.
            formal, schema = build.assemble_rows(rebuilt.drop(columns="_exact_hid"), batch_anchors, populated)
            if schema["all_features"] != core.d5_columns():
                raise c.ContractError("Reconstructed D5 columns/order changed")
            formal_writer.write(formal)
            # Complete-calendar PI transport preserves the original float64 key.
            exact = rebuilt.pop("_exact_hid")
            rebuilt[c.H] = exact
            complete, schema = transport.feature_assembler()(rebuilt, batch_anchors, populated)
            requested = rows.loc[rows[c.H].isin(ids)]
            if len(requested):
                transported = transport.select_requested(complete[[c.H, c.DATE, *core.d5_columns()]], requested)
                pi_writer.write(transported)
                support = line_S.build_features(requested[[c.H, c.DATE]], intervals)
                support.insert(0, c.DATE, requested[c.DATE].to_numpy())
                support.insert(0, c.H, requested[c.H].to_numpy())
                support.insert(0, c.SOURCE_ROW, requested[c.SOURCE_ROW].to_numpy())
                support_writer.write(support)
            del matrices, rebuilt, complete, transported, formal
            gc.collect()
            print(json.dumps({"stage": "feature-build", "stays_done": min(start + batch_stays, len(all_ids)),
                              "stays_total": len(all_ids)}), flush=True)
        success = True
    finally:
        for writer in (formal_writer, pi_writer, support_writer):
            writer.close(success)
    old = pd.read_parquet(config.CACHE_ROOT / "v26_rows.parquet", columns=[c.H, c.DATE])
    old_keys = pd.MultiIndex.from_frame(old)
    source_metadata["matched_old_cache"] = pd.MultiIndex.from_frame(source_metadata[[c.H, c.DATE]]).isin(old_keys)
    rows["matched_old_cache"] = source_metadata.set_index(c.SOURCE_ROW).loc[rows[c.SOURCE_ROW], "matched_old_cache"].to_numpy()
    source_metadata.to_parquet(core.PRIVATE / "source_metadata.parquet", index=False)
    rows.to_parquet(core.PRIVATE / "rows.parquet", index=False)
    stays.to_parquet(core.PRIVATE / "stays.parquet", index=False)
    book = line_R.build_codebook(line_R.read_tables(raw_dir))
    requested_stays = pd.Index(np.sort(rows[c.H].unique())).map(line_R._code)
    if not requested_stays.is_unique or not requested_stays.isin(book.index).all():
        raise c.ContractError("Raw G codebook lost a source stay")
    book = book.loc[requested_stays].copy()
    book.index.name = "raw_stay_key"
    book.reset_index().to_parquet(core.PRIVATE / "raw_codes.parquet", index=False)
    for repeat in range(1, 6):
        split = wp0.make_split(strata, 20260925 + 1000 * repeat, matched_stays=None)
        frame = pd.DataFrame({c.H: np.concatenate([split["train"], split["test"]]),
                             "partition": ["train"] * len(split["train"]) + ["test"] * len(split["test"])})
        frame["C_h"] = frame[c.H].map(strata).astype(np.int8)
        frame.to_parquet(core.PRIVATE / f"split_{repeat:02d}.parquet", index=False)
    from .harness import harness_v3 as hv
    hv.use_data("v2.6")
    hv.make_splits()
    records = {}
    names = ("rows", "stays", "source_metadata", "D5", "S", "raw_codes") + tuple(f"split_{r:02d}" for r in range(1, 6))
    for name in names:
        path = core.PRIVATE / f"{name}.parquet"
        records[path.name] = {"sha256": core.file_hash(path)}
    manifest = {"schema": 1, "source_spec_sha256": c.SPEC_SHA256, "files": records,
                "input_sha256": inputs, "population_frame_hash": core.frame_hash(rows),
                "formal_backing_rows": formal_writer.rows, "pi_rows": pi_writer.rows,
                "peak_rss_bytes": None if guard is None else guard.peak,
                "wall_seconds": time.monotonic() - started}
    for p in [clean_path] + sorted(raw_dir.glob("*.csv")):
        if core.file_hash(p) != inputs[p.name]:
            raise c.ContractError("Protected input changed during feature build")
    (core.PRIVATE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
