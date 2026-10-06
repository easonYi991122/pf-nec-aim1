"""RX-I11: one SHAP stream, bounded repeat storage, exact legacy reductions.

The original interpret.shap_importance/paired_pod_changes remain the oracle.
Shared layout/index preparation and unused-statistic projection preserve the
reported stay reductions and packet boundaries. Private temporary arrays use an
explicit caller-owned directory and are removed even on failure. No fitting,
data discovery, output registration or changes to the binding digest occur.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import resource
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pf_nec import contract as c
from . import interpret as it

CELL_LIMIT = int(1.5 * 1024**3)
RSS_LIMIT = 8 * 1024**3
DISK_LIMIT = int(c.SPEC["compute"]["temp_per_job_GiB"] * 1024**3)


class _SpillLimit(Exception):
    """Retry a smaller deterministic group; never change a numeric reduction."""


class _BoundedFile:
    """Stop a ZIP write before it can cross the private temporary-space cap."""
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise _SpillLimit()
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _rss_check(limit, allocation=0):
    from pf_nec.run import current_rss
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if max(peak, current_rss(peak) + allocation) > limit:
        raise it.EvaluationError("T3 fast RSS/allocation exceeds the per-process limit")


class _Layout:
    """Cache only validated column indices; do not replace any floating sum."""
    def __init__(self, layout, features, width):
        self.frame = it.validate_layout(layout, width)
        self.indices = []
        for feature in features:
            ix = np.flatnonzero(self.frame.feature.eq(feature).to_numpy())
            part = self.frame.iloc[ix]
            obs = ix[part.kind.isin(["observation", "missingness", "structural"]).to_numpy()]
            value = ix[part.kind.eq("value").to_numpy()]
            lags = [(int(lag), ix[part.lag.eq(lag).to_numpy()]) for lag in sorted(part.lag.dropna().unique())]
            self.indices.append((feature, ix, obs, value, lags))
        if not set(self.frame.feature) <= set(features):
            raise it.EvaluationError("Feature universe is duplicated or omits a source")

    def group(self, packet, features):
        phi = np.asarray(packet["phi"], float)
        if phi.ndim != 2 or phi.shape[1] != len(self.frame) or not np.isfinite(phi).all():
            raise it.EvaluationError("SHAP must be a finite row-by-column matrix")
        gross, signed, observation = (np.zeros((len(phi), len(features))) for _ in range(3))
        nonmissing = np.zeros_like(gross)
        values = None
        if "values" in packet:
            values = np.asarray(packet["values"], float)
            if values.shape != phi.shape or np.isinf(values).any():
                raise it.EvaluationError("Feature values must match SHAP columns; NaN allowed, infinity not")
            if isinstance(packet["values"], pd.DataFrame) and list(packet["values"].columns) != list(self.frame.column):
                raise it.EvaluationError("Feature value column order differs from lineage")
        included, lag_profile = [], {}
        for j, (feature, ix, obs, value, lags) in enumerate(self.indices):
            included.append(bool(len(ix)))
            gross[:, j] = np.abs(phi[:, ix]).sum(axis=1)
            signed[:, j] = phi[:, ix].sum(axis=1)
            observation[:, j] = np.abs(phi[:, obs]).sum(axis=1)
            for lag, slots in lags:
                lag_profile[feature, lag] = np.abs(phi[:, slots]).sum(axis=1)
            if values is not None:
                nonmissing[:, j] = np.isfinite(values[:, value]).any(axis=1)
        grouped = dict(abs_sum=gross, net_abs=np.abs(signed), signed=signed,
                       observation_abs=observation, included=np.array(included, bool), lag_profile=lag_profile)
        it.check_group_sums(phi, grouped)
        stats = np.stack([grouped[k] for k in ("abs_sum", "net_abs", "signed", "observation_abs")] + [nonmissing], axis=1)
        return grouped, stats


class _Store:
    """Keep one repeat resident; losslessly spill/reload at repeat boundaries.

    Reappearing repeats are supported. Positions preserve each dictionary's
    first-seen stay order; neither spilling nor loading performs arithmetic.
    """
    def __init__(self, directory, plan, names, paired, disk_limit):
        self.directory, self.plan = Path(directory), plan
        self.index = pd.Index(plan.stays.stay)
        self.names = list(names)
        self.pair_names = [f"POD{lo}-{hi}" for lo, hi in it.POD_BINS] if paired else []
        self.cells = {name: {} for name in self.names + self.pair_names}
        self.pair_order = {name: [] for name in self.pair_names}
        self.active = None
        self.paths = {name: {} for name in self.cells}
        self.disk_limit, self.disk_bytes = disk_limit, 0

    def switch(self, repeat):
        if repeat == self.active:
            return
        self.flush()
        self.active = repeat
        for name, cells in self.cells.items():
            if repeat not in self.paths[name]:
                continue
            with np.load(self.paths[name][repeat], allow_pickle=False) as saved:
                positions, counts, sums, folds = (saved[k] for k in ("positions", "count", "sums", "folds"))
            for stay, count, total, fold in zip(self.index.to_numpy()[positions], counts, sums, folds):
                key = (repeat, int(fold), stay) if name in self.pair_order else (repeat, stay)
                cells[key] = int(count), total

    def flush(self):
        if self.active is None:
            return
        for slot, (name, cells) in enumerate(self.cells.items()):
            if not cells:
                continue
            keys, values = zip(*cells.items())
            path = self.directory / f"s{slot}-r{self.active}.npz"
            # Same stack and dtype as interpret._compile_cells; disk round trips
            # preserve IEEE bits and the original insertion order.
            old_bytes = path.stat().st_size if path.exists() else 0
            remaining = self.disk_limit - (self.disk_bytes - old_bytes)
            with path.open("wb") as stream:
                # Explicit ZIP context also closes on a capped write (NumPy's
                # savez helper does not close its archive on write exceptions).
                with ZipFile(_BoundedFile(stream, remaining), "w", compression=ZIP_DEFLATED) as archive:
                    arrays = dict(positions=self.index.get_indexer([key[-1] for key in keys]),
                                  count=np.array([v[0] for v in values], float),
                                  sums=np.stack([v[1] for v in values]),
                                  folds=np.array([key[1] if name in self.pair_order else -1 for key in keys], int))
                    for name_in_zip, array in arrays.items():
                        with archive.open(name_in_zip + ".npy", "w", force_zip64=True) as member:
                            np.lib.format.write_array(member, array, allow_pickle=False)
                    del arrays, array
            self.disk_bytes += path.stat().st_size - old_bytes
            self.paths[name][self.active] = str(path)
            cells.clear()
            del keys, values
        self.active = None


def _compiled(paths, repeats):
    if any(r not in paths for r in repeats):
        return None
    blocks = []
    for repeat in repeats:
        with np.load(paths[repeat], allow_pickle=False) as saved:
            blocks.append(tuple(saved[k] for k in ("positions", "count", "sums")))
    return blocks


def _stratum_job(job):
    """Legacy finalizer with invariant equal-stay divisions cached once."""
    with threadpool_limits(limits=2):
        return _finish_stratum(job)


def _worker_init():
    c.reset_threads()  # includes torch-before-learner and Arrow/native caps


def _bootstrap_cells(compiled):
    if compiled is None:
        return None, None
    draws, sensitivity = [], []
    for positions, count, sums in compiled:
        # Bootstrap CIs never use statistic 4 (nonmissing counts); equal-stay
        # CIs use only statistic 0. Preserve the stay reduction axis/order.
        draws.append((positions, count, sums[:, :4, :]))
        # A scalar einsum output uses a different summation kernel. Keep the
        # original five-stat shape for one-feature programs, including -0.0.
        stay_sums = sums if sums.shape[-1] == 1 else sums[:, :1, :]
        sensitivity.append((positions, np.ones(len(count)), stay_sums / count[:, None, None]))
    return draws, sensitivity


def _finish_stratum(job):
    frame, view, features, repeats, plan = (job[k] for k in ("frame", "view", "features", "repeats", "plan"))
    selected, lag_sums, lag_sources = (job[k] for k in ("selected", "lag_sums", "lag_sources"))
    counts, attempts = job["counts"], job["attempts"]
    extra = job["cell_bytes"] if len(features) == 1 else job["cell_bytes"] // 5
    _rss_check(job["rss_limit"], job["cell_bytes"] + extra)
    compiled = _compiled(job["paths"], repeats)
    draw_compiled, stay_compiled = _bootstrap_cells(compiled)
    point = it._cell_means(compiled)
    stay_point = it._cell_means(stay_compiled)
    support_ok = all(counts[str(r)]["unique_stays"] >= 20 for r in repeats)
    sample = np.full((attempts, 4, len(features)), np.nan)
    stay_sample = np.full((attempts, len(features)), np.nan)
    rank_sample = np.full((attempts, len(features)), np.nan)
    if support_ok and point is not None:
        rng = np.random.default_rng(it.seed(frame, "bootstrap"))
        for b in range(attempts):
            if b % 64 == 0:
                _rss_check(job["rss_limit"])
            multiplicity = plan.draw(rng)
            means = it._cell_means(draw_compiled, multiplicity)
            sensitivity = it._cell_means(stay_compiled, multiplicity)
            if means is not None:
                sample[b] = means.mean(axis=0)[:4]
                rank_sample[b] = it._ranks(sample[b, 0], features)
                stay_sample[b] = sensitivity.mean(axis=0)[0]
    mean = np.full((5, len(features)), np.nan) if point is None else point.mean(axis=0)
    ranks = np.full((len(repeats), len(features)), np.nan) if point is None or not support_ok else np.stack([it._ranks(p[0], features) for p in point])
    selection = np.stack(list(selected.values())) if selected else np.zeros((0, len(features)), bool)
    feature_results = []
    for j, feature in enumerate(features):
        entry = {"feature": feature, "mean_abs": it._nullable(mean[0, j]), "net_mean_abs": it._nullable(mean[1, j]),
                 "signed_mean": it._nullable(mean[2, j]), "observation_missingness_mean_abs": it._nullable(mean[3, j]),
                 "normalized_share": it._nullable(mean[0, j] / mean[0].sum()) if mean[0].sum() > 0 else None,
                 "selection_frequency": float(selection[:, j].mean()) if len(selection) else None,
                 "included_in_any_model": bool(selection[:, j].any()),
                 "nonmissing_row_fraction": it._nullable(mean[4, j]) if job["observed_values"] else None,
                 "equal_stay_mean_abs": it._nullable(stay_point.mean(axis=0)[0, j]) if stay_point is not None else None,
                 "per_repeat": [{"repeat": int(r), "mean_abs": it._nullable(point[i, 0, j]) if point is not None else None,
                                 "rank": it._nullable(ranks[i, j])} for i, r in enumerate(repeats)],
                 "rank_interval": it._percentile(rank_sample[:, j]),
                 "repeat_rank_range": [float(ranks[:, j].min()), float(ranks[:, j].max())] if np.isfinite(ranks[:, j]).all() else None,
                 "top10_frequency": None, "top20_frequency": None,
                 "ci": {field: it._percentile(sample[:, k, j]) for k, field in enumerate(("mean_abs", "net_mean_abs", "signed_mean", "observation_missingness_mean_abs"))},
                 "equal_stay_ci": it._percentile(stay_sample[:, j])}
        if support_ok and np.isfinite(ranks[:, j]).all():
            included_by_repeat = np.array([any(v[j] for (rep, _), v in selected.items() if rep == r) for r in repeats])
            entry["top10_frequency"] = float(((ranks[:, j] <= 10) & included_by_repeat).mean())
            entry["top20_frequency"] = float(((ranks[:, j] <= 20) & included_by_repeat).mean())
        feature_results.append(entry)
    profile = []
    for feature, lag in sorted(lag_sources):
        per_repeat = [lag_sums[r, feature, lag] / counts[str(r)]["rows"] if counts[str(r)]["rows"] else np.nan for r in repeats]
        profile.append(dict(feature=feature, lag=lag, mean_abs=it._nullable(np.mean(per_repeat))))
    return dict(counts_by_repeat=counts, union_unique_stays=job["union_unique_stays"],
                explanation_coverage=1. if job["rows"] else None, rank_ci_support_ok=support_ok,
                features=feature_results, lag_profile=profile,
                case_only_auc=None if view == "lead" else "not_computed_here")


def _pair_bin(store, name):
    values = {}
    for repeat, path in store.paths[name].items():
        with np.load(path, allow_pickle=False) as saved:
            positions, counts, sums, folds = (saved[k] for k in ("positions", "count", "sums", "folds"))
        for stay, count, total, fold in zip(store.index.to_numpy()[positions], counts, sums, folds):
            values[repeat, int(fold), stay] = count, total
    # Preserve the legacy set construction as well as dictionary insertion
    # order. Do this in the parent: Python's string hash salt differs on spawn.
    return {key: values[key] for key in store.pair_order[name]}


def _finish_paired(store, repeats, plan, frame, features, rss_limit):
    result = []
    for earlier, later in zip(store.pair_names[:-1], store.pair_names[1:]):
        _rss_check(rss_limit)
        before, after = _pair_bin(store, earlier), _pair_bin(store, later)
        common = set(before) & set(after)
        cells = {}
        for repeat, fold, stay in common:
            n0, s0 = before[repeat, fold, stay]
            n1, s1 = after[repeat, fold, stay]
            if (repeat, stay) in cells:
                raise it.EvaluationError("Same stay attributed by multiple models in a repeat")
            cells[repeat, stay] = (1, (s1 / n1 - s0 / n0)[None, :])
        counts = {str(r): sum(rep == r for rep, _ in cells) for r in repeats}
        compiled = it._compile_cells(cells, repeats, plan)
        del before, after, cells
        point = it._cell_means(compiled)
        samples = np.full((it.BOOTSTRAPS, len(features)), np.nan)
        if all(n >= 20 for n in counts.values()) and point is not None:
            rng = np.random.default_rng(it.seed(frame, "bootstrap"))
            for b in range(it.BOOTSTRAPS):
                if b % 64 == 0:
                    _rss_check(rss_limit)
                means = it._cell_means(compiled, plan.draw(rng))
                if means is not None:
                    samples[b] = means.mean(axis=0)[0]
        result.append(dict(earlier=earlier, later=later, paired_stays_by_repeat=counts,
                           features=[dict(feature=f, later_minus_earlier=it._nullable(point.mean(axis=0)[0, j]) if point is not None else None,
                                          uncertainty=it._percentile(samples[:, j])) for j, f in enumerate(features)]))
    return dict(frame=frame, comparisons=result, conditioning="same fitted model; observed in both adjacent POD bins",
                weighting="equal repeats; equal paired stays", eligibility_filter=False)


def _groups(jobs, pair_bytes, limit):
    """Deterministic grouped-pass fallback if one repeat cannot hold all views."""
    groups, current, used = [], [], 0
    for key, cost in [(job["key"], job["repeat_bytes"]) for job in jobs] + ([(None, pair_bytes)] if pair_bytes is not None else []):
        if cost > limit:
            raise it.EvaluationError("One T3 accumulator exceeds the cell memory limit; never subsample rows")
        if current and used + cost > limit:
            groups.append(current)
            current, used = [], 0
        current.append(key)
        used += cost
    if current:
        groups.append(current)
    return groups


def _consume(rows, packets, features, jobs, store, guard, rss_limit):
    lookup, seen, models = rows.set_index(["repeat", "row_key"]), set(), {}
    observed_values, max_error, layout = True, 0., None
    views = list(dict.fromkeys(job["view"] for job in jobs))
    if store.pair_names and "pod" not in views:
        views.append("pod")
    for packet in packets:
        if guard:
            guard()
        _rss_check(rss_limit)
        part = it._packet_rows(packet, lookup, seen, models)
        phi = np.asarray(packet["phi"])
        if len(phi) != len(part):
            raise it.EvaluationError("SHAP rows and target keys differ")
        max_error = max(max_error, it.check_shap_additivity(phi, packet["base"], packet["margin"])["max_absolute_error"])
        candidate = pd.DataFrame(packet["layout"]).reset_index(drop=True)
        if layout is None or not layout.frame.equals(candidate):
            layout = _Layout(candidate, features, phi.shape[1])
        grouped, stats = layout.group(packet, features)
        observed_values &= "values" in packet
        repeat = int(part.repeat.iloc[0])
        fold = int(part.fold.iloc[0]) if "fold" in part else -1
        store.switch(repeat)
        masks = {view: it.time_masks(part, view) for view in views}
        indices = {}
        for view, strata in masks.items():
            for name, mask in strata.items():
                positions = np.flatnonzero(mask)
                indices[view, name] = [(stay, positions[ix]) for stay, ix in part.loc[mask].groupby("stay", sort=False).indices.items()]
        for job in jobs:
            name, view, key = job["name"], job["view"], job["key"]
            mask = masks[view][name]
            if not mask.any():
                continue
            previous = job["selected"].get((repeat, fold))
            if previous is not None and not np.array_equal(previous, grouped["included"]):
                raise it.EvaluationError("Selected feature set changed within one fitted model")
            job["selected"][repeat, fold] = grouped["included"].copy()
            cells = store.cells[key]
            for stay, positions in indices[view, name]:
                cell = (repeat, stay)
                old_count, old_sum = cells.get(cell, (0, np.zeros((5, len(features)))))
                cells[cell] = old_count + len(positions), old_sum + stats[positions].sum(axis=0)
            for (feature, lag), values in grouped["lag_profile"].items():
                job["lag_sources"].add((feature, lag))
                job["lag_sums"][repeat, feature, lag] += float(values[mask].sum())
        for name in store.pair_names:
            cells = store.cells[name]
            for stay, positions in indices["pod", name]:
                key = (repeat, fold, stay)
                if key not in cells:
                    store.pair_order[name].append(key)
                count, sums = cells.get(key, (0, np.zeros(len(features))))
                cells[key] = count + len(positions), sums + grouped["abs_sum"][positions].sum(axis=0)
    if len(seen) != len(rows):
        raise it.EvaluationError("Incomplete held-out SHAP coverage; no silent row omission")
    store.flush()
    for job in jobs:
        job.update(observed_values=observed_values, paths=store.paths[job["key"]], max_error=max_error)


@threadpool_limits.wrap(limits=2)
def shap_views(targets, packets_factory, *, frame, feature_universe, views,
               scratch_dir, paired=False, workers=0, guard=None,
               cell_limit=CELL_LIMIT, rss_limit=RSS_LIMIT, disk_limit=DISK_LIMIT):
    """Return (legacy-ordered SHAP views, paired POD or None).

    workers=0 (default) finalizes serially; >=2 explicitly enables spawn
    workers for independent stratum bootstraps, each at two numeric threads.
    SHAP is produced once per packet per memory group, always in caller order.
    Losslessly compressed temporary arrays are capped at the frozen 2 GiB.
    The resident repeat and each final stratum obey the 1.5 GiB cell guard.
    Larger combined repeats or spills use deterministic grouped passes without
    sampling. A failed spill is discarded before replaying smaller groups.
    """
    if not callable(packets_factory):
        raise it.EvaluationError("Provide a reiterable packet factory for bounded grouped passes")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise it.EvaluationError("workers must be a nonnegative integer (0 disables processes)")
    if not views or len(set(views)) != len(views):
        raise it.EvaluationError("Declare unique interpretation views")
    if not 0 < cell_limit <= CELL_LIMIT or rss_limit <= 0 or not 0 < disk_limit <= DISK_LIMIT:
        raise it.EvaluationError("Invalid T3 cell/RSS memory limit")
    c.reset_threads()
    rows = it.validate_targets(targets, frame)
    features = list(feature_universe)
    if not features or len(set(features)) != len(features):
        raise it.EvaluationError("Declare one unique feature universe across fits")
    repeats = sorted(rows.repeat.unique())
    # Only draw() is used, so avoid serializing row-to-stay mappings to workers.
    plan = it.StayBootstrap(rows[["stay", "case"]].drop_duplicates("stay"))
    jobs = []
    for view in views:
        for name, mask in it.time_masks(rows, view).items():
            part = rows.loc[mask]
            counts = {str(r): it.stage_counts(part.loc[part.repeat.eq(r)])["total"] for r in repeats}
            sizes = [counts[str(r)]["unique_stays"] * 5 * len(features) * 8 for r in repeats]
            if sum(sizes) > CELL_LIMIT:
                raise it.EvaluationError("SHAP stay aggregates exceed 1.5 GiB; stream one stratum/model at a time, never subsample rows")
            jobs.append(dict(key=(view, name), view=view, name=name, frame=frame, features=features,
                             repeats=repeats, plan=plan, counts=counts, union_unique_stays=int(part.stay.nunique()),
                             rows=len(part), repeat_bytes=max(sizes), cell_bytes=sum(sizes),
                             selected={}, lag_sums=defaultdict(float), lag_sources=set(),
                             attempts=it.BOOTSTRAPS, rss_limit=rss_limit))
    pair_bytes = None
    if paired:
        sizes = defaultdict(int)
        for name, mask in it.time_masks(rows, "pod").items():
            if name != "preop":
                for r, part in rows.loc[mask].groupby("repeat", sort=False):
                    sizes[r] += part.stay.nunique() * len(features) * 8
        pair_bytes = max(sizes.values(), default=0)
    groups = _groups(jobs, pair_bytes, cell_limit)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output, paired_result = {view: {} for view in views}, None
    with TemporaryDirectory(prefix=".t3-fast-", dir=scratch_dir) as temporary:
        number = 0
        while groups:
            group = groups.pop(0)
            directory = Path(temporary) / str(number)
            directory.mkdir()
            number += 1
            chosen = [job for job in jobs if job["key"] in group]
            for job in chosen:
                job.update(selected={}, lag_sums=defaultdict(float), lag_sources=set())
            store = _Store(directory, plan, [job["key"] for job in chosen], None in group, disk_limit)
            try:
                _consume(rows, packets_factory(), features, chosen, store, guard, rss_limit)
            except _SpillLimit:
                # Only this attempt's private temporary files are removed.
                # Recreate every accumulator: a partial attempt cannot enter
                # the accepted reduction or double-count any packet.
                shutil.rmtree(directory)
                if len(group) == 1:
                    raise it.EvaluationError("One T3 stratum exceeds the temporary disk limit; never subsample rows") from None
                middle = len(group) // 2
                groups[:0] = [group[:middle], group[middle:]]
                continue
            if workers >= 2 and chosen:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_worker_init) as pool:
                    results = list(pool.map(_stratum_job, chosen, chunksize=1))
            else:
                results = [_stratum_job(job) for job in chosen]
            for job, result in zip(chosen, results):
                output[job["view"]][job["name"]] = result
            if store.pair_names:
                paired_result = _finish_paired(store, repeats, plan, frame, features, rss_limit)
            for paths in store.paths.values():
                for path in paths.values():
                    Path(path).unlink()
            if guard:
                guard()
    reports = {}
    for view in views:
        # _shap_views previously popped strata and appended it last. Preserve
        # that exact JSON key order as well as the per-stratum lag-source sets.
        last = next(job for job in reversed(jobs) if job["view"] == view)
        reports[view] = dict(frame=frame, view=view, shap_scale="raw_margin", additivity_max_error=last["max_error"],
                             weighting="equal repeats; equal rows within repeat", sensitivity="equal stays within repeat",
                             uncertainty="conditional stay bootstrap; descriptive pointwise intervals",
                             unselected="zero program attribution; not a biological null", attempted_draws=it.BOOTSTRAPS,
                             strata=output[view])
    return reports, paired_result
