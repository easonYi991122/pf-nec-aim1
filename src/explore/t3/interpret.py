"""Held-out, descriptive RX-D1 interpretation; never a model-selection input.

Inputs have already passed the feature/risk-set contract. Outcome dates enter
only these evaluation tables, never the predictor matrix or donor matching.
All public summaries are aggregate; native_shap_chunks yields private arrays
for immediate aggregation. Nothing here loads real data, fits, or writes files.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pf_nec.evaluate import (BOOTSTRAPS, MIN_VALID, SPEC, BinaryMetric, EvaluationError,
                       StayBootstrap, _nullable, seed, stage_counts, validate_targets)

POD_BINS = ((0, 2), (3, 7), (8, 14), (15, 31))
LEAD_BINS = ((1, 1), (2, 2), (3, 3), (4, 7), (8, 14), (15, 31), (32, np.inf))
SHAP_TOLERANCE = 1e-4
KINDS = ("value", "observation", "missingness", "structural")


def tree_layout(matrix, *, window, representation, dependencies=None):
    """Adapt windows.TreeMatrix's explicit concepts/observation metadata.

    The only parsed names are the frozen R1 __lagN suffixes. Global row/window
    indicators get a separate structural group included in the SHAP identity.
    """
    if window not in SPEC["windows"]["lengths"] or representation not in ("R1", "R3"):
        raise EvaluationError("Unregistered tree window/representation")
    if len(matrix.columns) != len(matrix.concepts):
        raise EvaluationError("TreeMatrix concept metadata is incomplete")
    dependencies = {} if dependencies is None else dependencies
    layout = []
    for column, concept in zip(matrix.columns, matrix.concepts):
        if window == 1:
            lag = 0
        elif representation == "R1" and "__lag" in column:
            lag = int(column.rsplit("__lag", 1)[1])
            if not 0 <= lag < window:
                raise EvaluationError("TreeMatrix lag outside declared window")
        else:
            lag = None
        kind = "structural" if concept is None else "observation" if column in matrix.observation_columns else "value"
        layout.append(dict(column=column, feature=concept if concept is not None else "__window_structure__",
                           lag=lag, kind=kind, dependencies=list(dependencies.get(concept, []))))
    return validate_layout(layout, matrix.values.shape[1])


def window_patterns(row_observed, gates):
    """Encode WindowBatch's oldest-to-current structural metadata for donors."""
    observed, gates = np.asarray(row_observed), np.asarray(gates)
    if observed.ndim != 2 or gates.shape != (*observed.shape, 3) or not np.isin(observed, [0, 1]).all() or not np.isin(gates, [0, 1]).all():
        raise EvaluationError("Expected row-observed [n,w] and post/surgery/two-hour gates [n,w,3]")
    def encode(bits):
        return ["".join("1" if v else "0" for v in row) for row in bits]
    return pd.DataFrame({"row_observed_pattern": encode(observed), "surgery_pattern": encode(gates[:, :, 1]),
                         "two_hour_pattern": encode(gates[:, :, 2])})


def _cap_torch_threads():
    import torch  # libomp before any callback importing XGBoost/LightGBM.
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() > 2:
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError as exc:
            raise EvaluationError("Torch interop already started above two threads; reset in run.py before work") from exc


def validate_layout(layout, width=None):
    """Explicit column -> source/lag/kind lineage, including missingness columns.

    R3's seven summaries share one source feature and have lag=None. Mark
    n_observed and missingness separately. Global window indicators must have
    their own structural source group, so every column is accounted for once.
    ``dependencies`` may list additional source names for denylist checking;
    the upstream contract must still verify the complete dependency graph.
    """
    layout = pd.DataFrame(layout).copy().reset_index(drop=True)
    required = {"column", "feature", "lag", "kind"}
    if not required <= set(layout) or layout.empty or layout.column.duplicated().any():
        raise EvaluationError("Explicit, unique SHAP column lineage is required")
    if width is not None and len(layout) != width:
        raise EvaluationError("Attribution width and column lineage differ")
    if layout[["column", "feature", "kind"]].isna().any().any() or not layout.kind.isin(KINDS).all():
        raise EvaluationError("Invalid SHAP source/kind")
    if not all(isinstance(x, str) and x for x in list(layout.column) + list(layout.feature)):
        raise EvaluationError("Column/source names must be nonempty strings")
    lag = layout.lag.dropna().to_numpy(float)
    if not np.isfinite(lag).all() or (lag < 0).any() or (lag != np.floor(lag)).any():
        raise EvaluationError("Lag must be a nonnegative calendar offset or null")
    denied = {s.lower() for s in SPEC["deny_features"]["exact"]} | {"y", "case", "event_day", "row_key", "stay", "repeat", "fold"}
    prefixes = tuple(s.lower() for s in SPEC["deny_features"]["prefix_case_insensitive"])
    for item in layout.to_dict("records"):
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, (list, tuple)):
            raise EvaluationError("Dependencies must be an explicit list")
        names = [item["column"], item["feature"], *dependencies]
        if any(str(name).lower() in denied or str(name).lower().startswith(prefixes) for name in names):
            raise EvaluationError("Forbidden outcome/identity dependency in attribution inputs")
    return layout


def check_shap_additivity(phi, base, margin):
    """Assert raw-margin additivity with the frozen absolute 1e-4 tolerance."""
    phi, margin, base = np.asarray(phi, float), np.asarray(margin, float), np.asarray(base, float)
    if phi.ndim != 2 or margin.shape != (len(phi),) or base.shape not in ((), (len(phi),)):
        raise EvaluationError("Invalid SHAP/base/margin shapes (binary margin only)")
    if not np.isfinite(phi).all() or not np.isfinite(base).all() or not np.isfinite(margin).all():
        raise EvaluationError("Nonfinite SHAP/base/margin")
    error = float(np.max(np.abs(phi.sum(axis=1) + base - margin), initial=0.))
    if error > SHAP_TOLERANCE:
        raise EvaluationError(f"SHAP margin additivity failed: {error:g}")
    return {"max_absolute_error": error, "tolerance": SHAP_TOLERANCE}


def check_group_sums(phi, grouped):
    """Check absolute/signed partitions and net-vs-gross cancellation identities."""
    phi = np.asarray(phi, float)
    gross, net, signed = (np.asarray(grouped[k], float) for k in ("abs_sum", "net_abs", "signed"))
    if gross.shape != net.shape or gross.shape != signed.shape or gross.ndim != 2 or len(gross) != len(phi):
        raise EvaluationError("Grouped SHAP shape mismatch")
    errors = [np.max(np.abs(gross.sum(axis=1) - np.abs(phi).sum(axis=1)), initial=0.),
              np.max(np.abs(signed.sum(axis=1) - phi.sum(axis=1)), initial=0.),
              np.max(np.abs(net - np.abs(signed)), initial=0.)]
    if not np.isfinite(gross).all() or not np.isfinite(net).all() or not np.isfinite(signed).all():
        raise EvaluationError("Nonfinite grouped SHAP")
    if max(errors) > SHAP_TOLERANCE or (gross < -1e-12).any() or (net > gross + SHAP_TOLERANCE).any():
        raise EvaluationError("SHAP group-sum/cancellation identity failed")
    return {"max_absolute_error": float(max(errors))}


def group_shap(phi, layout, *, feature_universe=None):
    """Return row-level private feature sums and per-lag absolute profiles.

    Main attribution is sum(abs(phi)), not abs(sum(phi)). Unselected sources
    in a declared program universe have zero attribution and included=False.
    """
    phi = np.asarray(phi, float)
    if phi.ndim != 2 or not np.isfinite(phi).all():
        raise EvaluationError("SHAP must be a finite row-by-column matrix")
    layout = validate_layout(layout, phi.shape[1])
    features = sorted(set(layout.feature), key=lambda s: s.encode("utf-8")) if feature_universe is None else list(feature_universe)
    if len(set(features)) != len(features) or not set(layout.feature) <= set(features):
        raise EvaluationError("Feature universe is duplicated or omits a source")
    gross, signed, observation = (np.zeros((len(phi), len(features))) for _ in range(3))
    included, lag_profile = [], {}
    for j, feature in enumerate(features):
        ix = np.flatnonzero(layout.feature.eq(feature).to_numpy())
        included.append(bool(len(ix)))
        gross[:, j] = np.abs(phi[:, ix]).sum(axis=1)
        signed[:, j] = phi[:, ix].sum(axis=1)
        obs = ix[layout.iloc[ix].kind.isin(["observation", "missingness", "structural"]).to_numpy()]
        observation[:, j] = np.abs(phi[:, obs]).sum(axis=1)
        for lag in sorted(layout.iloc[ix].lag.dropna().unique()):
            slots = ix[layout.iloc[ix].lag.eq(lag).to_numpy()]
            lag_profile[feature, int(lag)] = np.abs(phi[:, slots]).sum(axis=1)
    result = dict(features=features, abs_sum=gross, net_abs=np.abs(signed), signed=signed,
                  observation_abs=observation, included=np.array(included, bool), lag_profile=lag_profile)
    result["identity"] = check_group_sums(phi, result)
    return result


def native_shap_chunks(model, X, *, learner, chunk_rows=1024):
    """Exact native path-dependent tree contributions from an existing model.

    Yield {start, stop, phi, base, margin}; every chunk is checked on the raw
    margin scale. Binary LGB/XGB only; neural models use grouped permutation.
    """
    if learner not in ("LGB", "XGB") or not 1 <= chunk_rows <= 1024:
        raise EvaluationError("Native SHAP requires LGB/XGB and chunks <=1024")
    _cap_torch_threads()
    if learner == "XGB":
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        booster.set_param({"nthread": 2})
    else:
        booster = model.booster_ if hasattr(model, "booster_") else model
    with threadpool_limits(limits=2):
        for start in range(0, len(X), chunk_rows):
            stop = min(start + chunk_rows, len(X))
            block = X.iloc[start:stop] if isinstance(X, pd.DataFrame) else X[start:stop]
            if learner == "XGB":
                matrix = xgb.DMatrix(block, nthread=2)
                contribution = booster.predict(matrix, pred_contribs=True, approx_contribs=False)
                margin = booster.predict(matrix, output_margin=True)
            else:
                contribution = booster.predict(block, pred_contrib=True, num_threads=2)
                margin = booster.predict(block, raw_score=True, num_threads=2)
            contribution = np.asarray(contribution)
            if contribution.shape != (len(block), X.shape[1] + 1):
                raise EvaluationError("Expected binary native contributions plus one base column")
            phi, base = contribution[:, :-1], contribution[:, -1]
            check_shap_additivity(phi, base, margin)
            yield dict(start=start, stop=stop, phi=phi, base=base, margin=np.asarray(margin))


def time_masks(rows, view):
    """Frozen descriptive strata; lead strata are case-only, never case-only AUC."""
    if view == "overall":
        return {"post": rows.pod.ge(0).to_numpy(), "all": np.ones(len(rows), bool)}
    if view == "pod":
        return {"preop": rows.pod.lt(0).to_numpy(), **{f"POD{lo}-{hi}": rows.pod.between(lo, hi).to_numpy() for lo, hi in POD_BINS}}
    if view == "lead":
        if not {"day", "event_day"} <= set(rows):
            raise EvaluationError("Case lead interpretation needs validated day/event_day metadata")
        lead = rows.event_day - rows.day
        if rows.loc[rows.case.eq(1), "event_day"].isna().any():
            raise EvaluationError("Missing event anchor in cases")
        if not np.isfinite(rows.day).all() or not rows.day.eq(np.floor(rows.day)).all():
            raise EvaluationError("Case lead bins require integer calendar days")
        events = rows.loc[rows.case.eq(1), "event_day"]
        if not np.isfinite(events).all() or not events.eq(np.floor(events)).all() or rows.loc[rows.case.eq(1)].groupby("stay").event_day.nunique().gt(1).any():
            raise EvaluationError("Case event anchors must be fixed integer dates")
        return {(">31" if np.isinf(hi) else str(lo) if lo == hi else f"{lo}-{hi}"):
                (rows.case.eq(1) & lead.between(lo, hi)).to_numpy() for lo, hi in LEAD_BINS}
    raise EvaluationError("view must be overall, pod or lead")


def _packet_rows(packet, targets, seen, models):
    rows = packet["rows"].reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(rows[["repeat", "row_key"]])
    if keys.has_duplicates or any(k in seen for k in keys):
        raise EvaluationError("Duplicate held-out explanation rows")
    if not keys.isin(targets.index).all():
        raise EvaluationError("Explanation contains rows outside target universe")
    reference = targets.reindex(keys).reset_index()
    meta = ["stay", "y", "pod", "case"] + (["fold"] if "fold" in targets else [])
    if not rows[meta].eq(reference[meta]).all().all():
        raise EvaluationError("Explanation metadata differs from targets")
    # Evaluation-only anchors are always taken from independent targets.
    for name in ("day", "event_day"):
        if name in reference:
            rows[name] = reference[name]
    seen.update(keys)
    if rows.repeat.nunique() != 1 or ("fold" in rows and rows.fold.nunique() != 1):
        raise EvaluationError("A packet must belong to one held-out fitted model")
    model_id = packet.get("model_id")
    context = (int(rows.repeat.iloc[0]), int(rows.fold.iloc[0]) if "fold" in rows else -1)
    if not isinstance(model_id, str) or not model_id:
        raise EvaluationError("Bind each explanation packet to its preserved model_id")
    if context in models and models[context] != model_id:
        raise EvaluationError("Different fitted models mixed in one repeat/fold context")
    models[context] = model_id
    return rows


def _ranks(values, names):
    """Deterministic descending rank; UTF-8 source names break exact ties."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        return np.full(len(values), np.nan)
    order = sorted(range(len(values)), key=lambda j: (-values[j], names[j].encode("utf-8")))
    rank = np.empty(len(values), float)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def _percentile(samples, *, attempts=BOOTSTRAPS, minimum=MIN_VALID):
    finite = np.isfinite(samples)
    n = int(finite.sum())
    return {"ci95": np.quantile(np.asarray(samples)[finite], [.025, .975], method="linear").tolist() if n >= minimum else None,
            "valid_draws": n, "undefined_draws": attempts - n}


def _compile_cells(cells, repeats, plan):
    compiled = []
    index = pd.Index(plan.stays.stay)
    for repeat in repeats:
        entries = [(stay, v) for (rep, stay), v in cells.items() if rep == repeat]
        if not entries:
            return None
        stays, values = zip(*entries)
        count = np.array([v[0] for v in values], float)
        sums = np.stack([v[1] for v in values])
        compiled.append((index.get_indexer(stays), count, sums))
        # Release each repeat's dictionary arrays as soon as its dense block
        # exists, avoiding a second full copy of all stay-level summaries.
        for stay in stays:
            del cells[repeat, stay]
        del entries, values
    return compiled


def _cell_means(compiled, multiplicity=None, *, equal_stays=False):
    if compiled is None:
        return None
    means = []
    for positions, count, sums in compiled:
        weights = np.ones(len(positions)) if multiplicity is None else multiplicity[positions]
        if equal_stays:
            sums = sums / count[:, None, None]
            count = np.ones(len(count))
        denominator = np.dot(weights, count)
        if denominator <= 0:
            return None
        means.append(np.einsum("i,ijk->jk", weights, sums) / denominator)
    return np.stack(means)


def shap_views(targets, packets_factory, *, frame, feature_universe, views,
               scratch_dir, paired=False, workers=0, guard=None, rss_limit=8 * 1024**3):
    """Single-stream T3 driver; serial by default, exact legacy JSON semantics.

    Temporary stay arrays are confined to scratch_dir. Each spawned bootstrap
    worker is capped at two numeric threads and the caller's RSS limit.
    """
    from .t3_fast import shap_views as aggregate
    return aggregate(targets, packets_factory, frame=frame, feature_universe=feature_universe,
                     views=views, scratch_dir=scratch_dir, paired=paired, workers=workers,
                     guard=guard, rss_limit=rss_limit)


@threadpool_limits.wrap(limits=2)
def shap_importance(targets, packets, *, frame, feature_universe, view="overall", strata=None):
    """T3a/c streaming aggregate SHAP with repeat ranks and stay bootstrap.

    Packets contain rows/phi/base/margin/layout/model_id, optional values (same ordered
    columns as layout). Packets may be native-SHAP chunks; each target must
    occur exactly once. Only per-stay sums survive a chunk. Call separately
    for overall/POD/lead, streaming one model program and (on Task A) one
    stratum at a time via strata=[name]. Targets still cover the full universe.
    """
    rows = validate_targets(targets, frame)
    target_index = rows.set_index(["repeat", "row_key"])
    features = list(feature_universe)
    if not features or len(set(features)) != len(features):
        raise EvaluationError("Declare one unique feature universe across fits")
    repeats, plan = sorted(rows.repeat.unique()), StayBootstrap(rows)
    masks = time_masks(rows, view)
    if strata is not None:
        if not strata or len(set(strata)) != len(strata) or not set(strata) <= set(masks):
            raise EvaluationError("Unknown or duplicate requested interpretation strata")
        masks = {name: masks[name] for name in strata}
    n_cells = sum(len(rows.loc[mask, ["repeat", "stay"]].drop_duplicates()) for mask in masks.values())
    if n_cells * 5 * len(features) * 8 > 1.5 * 1024**3:
        raise EvaluationError("SHAP stay aggregates exceed 1.5 GiB; stream one stratum/model at a time, never subsample rows")
    cells = {name: {} for name in masks}
    selected, lag_sums, lag_sources = defaultdict(dict), defaultdict(lambda: defaultdict(float)), set()
    observed_values, seen, models, max_error = True, set(), {}, 0.
    for packet in packets:
        part = _packet_rows(packet, target_index, seen, models)
        phi = np.asarray(packet["phi"])
        if len(phi) != len(part):
            raise EvaluationError("SHAP rows and target keys differ")
        max_error = max(max_error, check_shap_additivity(phi, packet["base"], packet["margin"])["max_absolute_error"])
        layout = validate_layout(packet["layout"], phi.shape[1])
        grouped = group_shap(phi, layout, feature_universe=features)
        nonmissing = np.zeros_like(grouped["abs_sum"])
        if "values" in packet:
            values = np.asarray(packet["values"], float)
            if values.shape != phi.shape or np.isinf(values).any():
                raise EvaluationError("Feature values must match SHAP columns; NaN allowed, infinity not")
            if isinstance(packet["values"], pd.DataFrame) and list(packet["values"].columns) != list(layout.column):
                raise EvaluationError("Feature value column order differs from lineage")
            for j, feature in enumerate(features):
                ix = np.flatnonzero((layout.feature.eq(feature) & layout.kind.eq("value")).to_numpy())
                nonmissing[:, j] = np.isfinite(values[:, ix]).any(axis=1)
        else:
            observed_values = False
        stats = np.stack([grouped[k] for k in ("abs_sum", "net_abs", "signed", "observation_abs")] + [nonmissing], axis=1)
        repeat = int(part.repeat.iloc[0])
        model = (repeat, int(part.fold.iloc[0]) if "fold" in part else -1)
        for name, mask in time_masks(part, view).items():
            if name not in masks or not mask.any():
                continue
            previous = selected[name].get(model)
            if previous is not None and not np.array_equal(previous, grouped["included"]):
                raise EvaluationError("Selected feature set changed within one fitted model")
            selected[name][model] = grouped["included"].copy()
            for stay, ix in part.loc[mask].groupby("stay", sort=False).indices.items():
                positions = np.flatnonzero(mask)[ix]
                key = (repeat, stay)
                old_count, old_sum = cells[name].get(key, (0, np.zeros((5, len(features)))))
                cells[name][key] = (old_count + len(positions), old_sum + stats[positions].sum(axis=0))
            for (feature, lag), values in grouped["lag_profile"].items():
                lag_sources.add((feature, lag))
                lag_sums[name][repeat, feature, lag] += float(values[mask].sum())
    if len(seen) != len(rows):
        raise EvaluationError("Incomplete held-out SHAP coverage; no silent row omission")
    result = {}
    for name, mask in masks.items():
        part = rows.loc[mask]
        counts = {str(r): stage_counts(part.loc[part.repeat.eq(r)])["total"] for r in repeats}
        compiled = _compile_cells(cells[name], repeats, plan)
        point = _cell_means(compiled)
        stay_point = _cell_means(compiled, equal_stays=True)
        support_ok = all(counts[str(r)]["unique_stays"] >= 20 for r in repeats)
        sample = np.full((BOOTSTRAPS, 4, len(features)), np.nan)
        stay_sample = np.full((BOOTSTRAPS, len(features)), np.nan)
        rank_sample = np.full((BOOTSTRAPS, len(features)), np.nan)
        if support_ok and point is not None:
            rng = np.random.default_rng(seed(frame, "bootstrap"))
            for b in range(BOOTSTRAPS):
                multiplicity = plan.draw(rng)
                means = _cell_means(compiled, multiplicity)
                sensitivity = _cell_means(compiled, multiplicity, equal_stays=True)
                if means is not None:
                    sample[b] = means.mean(axis=0)[:4]
                    rank_sample[b] = _ranks(sample[b, 0], features)
                    stay_sample[b] = sensitivity.mean(axis=0)[0]
        mean = np.full((5, len(features)), np.nan) if point is None else point.mean(axis=0)
        ranks = np.full((len(repeats), len(features)), np.nan) if point is None or not support_ok else np.stack([_ranks(p[0], features) for p in point])
        selection = np.stack(list(selected[name].values())) if selected[name] else np.zeros((0, len(features)), bool)
        feature_results = []
        for j, feature in enumerate(features):
            entry = {"feature": feature, "mean_abs": _nullable(mean[0, j]), "net_mean_abs": _nullable(mean[1, j]),
                     "signed_mean": _nullable(mean[2, j]), "observation_missingness_mean_abs": _nullable(mean[3, j]),
                     "normalized_share": _nullable(mean[0, j] / mean[0].sum()) if mean[0].sum() > 0 else None,
                     "selection_frequency": float(selection[:, j].mean()) if len(selection) else None,
                     "included_in_any_model": bool(selection[:, j].any()),
                     "nonmissing_row_fraction": _nullable(mean[4, j]) if observed_values else None,
                     "equal_stay_mean_abs": _nullable(stay_point.mean(axis=0)[0, j]) if stay_point is not None else None,
                     "per_repeat": [{"repeat": int(r), "mean_abs": _nullable(point[i, 0, j]) if point is not None else None,
                                     "rank": _nullable(ranks[i, j])} for i, r in enumerate(repeats)],
                     "rank_interval": _percentile(rank_sample[:, j]),
                     "repeat_rank_range": [float(ranks[:, j].min()), float(ranks[:, j].max())] if np.isfinite(ranks[:, j]).all() else None,
                     "top10_frequency": None, "top20_frequency": None,
                     "ci": {field: _percentile(sample[:, k, j]) for k, field in enumerate(("mean_abs", "net_mean_abs", "signed_mean", "observation_missingness_mean_abs"))},
                     "equal_stay_ci": _percentile(stay_sample[:, j])}
            if support_ok and np.isfinite(ranks[:, j]).all():
                included_by_repeat = np.array([any(v[j] for (rep, _), v in selected[name].items() if rep == r) for r in repeats])
                entry["top10_frequency"] = float(((ranks[:, j] <= 10) & included_by_repeat).mean())
                entry["top20_frequency"] = float(((ranks[:, j] <= 20) & included_by_repeat).mean())
            feature_results.append(entry)
        profile = []
        for feature, lag in sorted(lag_sources):
            per_repeat = [lag_sums[name][r, feature, lag] / counts[str(r)]["rows"] if counts[str(r)]["rows"] else np.nan for r in repeats]
            profile.append(dict(feature=feature, lag=lag, mean_abs=_nullable(np.mean(per_repeat))))
        result[name] = dict(counts_by_repeat=counts, union_unique_stays=int(part.stay.nunique()),
                            explanation_coverage=1. if len(part) else None, rank_ci_support_ok=support_ok,
                            features=feature_results, lag_profile=profile,
                            case_only_auc=None if view == "lead" else "not_computed_here")
    return dict(frame=frame, view=view, strata=result, shap_scale="raw_margin", additivity_max_error=max_error,
                weighting="equal repeats; equal rows within repeat", sensitivity="equal stays within repeat",
                uncertainty="conditional stay bootstrap; descriptive pointwise intervals",
                unselected="zero program attribution; not a biological null", attempted_draws=BOOTSTRAPS)


def permutation_groups(layout, primary_feature_ranking):
    """Freeze top20 + present PI10/ACTION16, semantic closures, and lag blocks.

    Ranking is held-out SHAP for explanation budgeting only, never fitting.
    Entries are (source, mean_abs); ties use UTF-8 names. Group definitions use
    source concepts so they also work when selected columns differ across fits.
    """
    layout = validate_layout(layout)
    present = set(layout.feature)
    structural = {f for f, part in layout.groupby("feature") if part.kind.eq("structural").all()}
    ranking = sorted(primary_feature_ranking, key=lambda v: (-v[1], v[0].encode("utf-8")))
    if len({v[0] for v in ranking}) != len(ranking) or any(not np.isfinite(v[1]) or v[1] < 0 for v in ranking) or not {v[0] for v in ranking} <= present:
        raise EvaluationError("Invalid primary SHAP ranking")
    ranking = [v for v in ranking if v[0] not in structural]
    chosen = {f for f, _ in ranking[:20]} | set(SPEC["feature_sets"]["PI10"]["columns"]) | set(SPEC["feature_sets"]["ACTION16"]["columns"])
    groups = [{"name": f"feature:{f}", "features": [f]} for f in sorted(chosen & present)]
    for name, sources in SPEC["interpretation"]["permutation"]["semantic_group_columns"].items():
        if present.intersection(sources):
            groups.append({"name": f"semantic:{name}", "features": list(sources)})
    lags = sorted(layout.lag.dropna().unique())
    if len(lags) > 1:
        groups += [{"name": f"lag:{int(lag)}", "lag": int(lag)} for lag in lags]
    return {"groups": groups, "unmeasured_features": sorted(present - chosen), "structural_groups": sorted(structural),
            "unmeasured_value": None, "selection_used_for_model_fitting": False}


def permutation_columns(layout, group):
    """Resolve a source's entire declared dependency closure or one lag block."""
    layout = validate_layout(layout)
    if ("lag" in group) == ("features" in group):
        raise EvaluationError("A permutation group declares either sources or one lag")
    if "lag" in group:
        return np.flatnonzero(layout.lag.eq(group["lag"]).to_numpy())
    closure, selected = set(group["features"]), set()
    records = layout.to_dict("records")
    changed = True
    while changed:
        changed = False
        for j, item in enumerate(records):
            if j not in selected and (item["feature"] in closure or item["column"] in closure or closure.intersection(item.get("dependencies", []))):
                selected.add(j)
                closure.update([item["feature"], item["column"]])
                changed = True
    return np.array(sorted(selected), int)


def normalized_gain(gain, layout):
    """Training split-use description only; normalize within this fitted model."""
    gain = np.asarray(gain, float)
    layout = validate_layout(layout, len(gain))
    if gain.ndim != 1 or not np.isfinite(gain).all() or (gain < 0).any():
        raise EvaluationError("Gain must be a finite nonnegative column vector")
    total = gain.sum()
    return {"features": [{"feature": f, "normalized_gain": float(gain[layout.feature.eq(f)].sum() / total) if total else None}
                         for f in sorted(layout.feature.unique())],
            "interpretation": "within-model training split use; not held-out or causal importance"}


def matched_donors(rows, rng):
    """Distinct-stay cyclic donors; exact POD and historical structural gates.

    Patterns are binary strings (one or more bits per history slot), supplied
    by the window/gating builder. Native feature missingness is not a match
    key. y, case and future outcomes are neither required nor inspected.
    """
    keys = ["pod", "row_observed_pattern", "surgery_pattern", "two_hour_pattern"]
    if not set(["stay", *keys]) <= set(rows) or rows[["stay", *keys]].isna().any().any():
        raise EvaluationError("Permutation requires explicit structural matching metadata")
    for key in keys[1:]:
        if not all(isinstance(v, str) and v and set(v) <= {"0", "1"} for v in rows[key]):
            raise EvaluationError("Structural patterns must be nonempty binary strings")
        if rows[key].str.len().nunique() != 1:
            raise EvaluationError("Structural bit-pattern widths differ within one model")
    if rows.duplicated(["stay", "pod"]).any():
        raise EvaluationError("A stay has multiple target rows at one POD")
    donor = np.arange(len(rows))
    eligible = np.zeros(len(rows), bool)
    for _, ix in rows.groupby(keys, sort=True).indices.items():
        if len(ix) < 5:
            continue
        # Sort before RNG so donor assignments are invariant to input row order.
        ix = np.asarray(ix)[np.argsort(rows.iloc[ix].stay.to_numpy(), kind="stable")]
        shuffled = rng.permutation(ix)
        shifted = np.roll(shuffled, int(rng.integers(1, len(shuffled))))
        donor[shuffled], eligible[ix] = shifted, True
    if (rows.stay.to_numpy()[donor[eligible]] == rows.stay.to_numpy()[eligible]).any():
        raise EvaluationError("Self donor in an eligible permutation stratum")
    return donor, {"rows": len(rows), "permutable_rows": int(eligible.sum()),
                   "permutable_stays": int(rows.loc[eligible, "stay"].nunique()),
                   "coverage": float(eligible.mean()) if len(rows) else None}


@threadpool_limits.wrap(limits=2)
def grouped_permutation(targets, records_factory, groups, *, frame):
    """T3a grouped reliance, 20 perturbations, repeat ranks and stay CIs.

    Reiterable factory yields one complete held-out model context at a time:
    {rows, X, layout, predict, model_id}. predict returns probabilities without
    fitting; callers bind <=2 native threads. Groups are processed one at a
    time to bound memory. Task A pools fold predictions *before* each AUC.
    """
    _cap_torch_threads()
    rows = validate_targets(targets, frame)
    target_index = rows.set_index(["repeat", "row_key"])
    if not callable(records_factory):
        raise EvaluationError("Provide a factory so each group can stream the same fixed models")
    if len({g["name"] for g in groups}) != len(groups):
        raise EvaluationError("Duplicate permutation group names")
    plan, repeats = StayBootstrap(rows), sorted(rows.repeat.unique())
    lookup = pd.MultiIndex.from_frame(rows[["repeat", "row_key"]])
    results, draws_for_ranks = [], {"post": [], "all": []}
    for group in groups:
        if ("lag" in group) == ("features" in group):
            raise EvaluationError("A permutation group declares either sources or one lag")
        probabilities = np.empty((21, len(rows)))
        permutable, selected_rows, seen, models = np.zeros(len(rows), bool), np.zeros(len(rows), bool), set(), {}
        for record in records_factory():
            previous_models = len(models)
            part = _packet_rows(record, target_index, seen, models)
            if len(models) == previous_models:
                raise EvaluationError("Permutation needs one complete test set per model, not chunks")
            ix = lookup.get_indexer(pd.MultiIndex.from_frame(part[["repeat", "row_key"]]))
            X = np.asarray(record["X"])
            layout = validate_layout(record["layout"], X.shape[1])
            if X.ndim != 2 or len(X) != len(part) or np.isinf(X).any():
                raise EvaluationError("Invalid permutation predictor matrix")
            columns = permutation_columns(layout, group)
            selected_rows[ix] = bool(len(columns))
            prediction = np.asarray(record["predict"](X), float)
            BinaryMetric(part.y, prediction)  # explicit probability/shape validation
            probabilities[0, ix] = prediction
            for trial in range(20):
                rng = np.random.default_rng(seed(frame, "permutation", repeat=int(part.repeat.iloc[0]),
                                                fold=int(part.fold.iloc[0]) if "fold" in part else -1, trial=trial))
                donors, _ = matched_donors(part, rng)
                permutable[ix] |= (donors != np.arange(len(part))) & bool(len(columns))
                perturbed = X.copy()
                perturbed[:, columns] = X[donors][:, columns]
                value = np.asarray(record["predict"](perturbed), float)
                BinaryMetric(part.y, value)
                probabilities[trial + 1, ix] = value
        if len(seen) != len(rows):
            raise EvaluationError("Incomplete grouped permutation target coverage")
        views = {}
        for name, mask in time_masks(rows, "overall").items():
            prepared = []
            for r in repeats:
                ix = np.flatnonzero(rows.repeat.eq(r).to_numpy() & mask)
                n_rows = rows.iloc[ix].groupby("stay").stay.transform("size").to_numpy()
                prepared.append((ix, n_rows, [BinaryMetric(rows.y.to_numpy()[ix], p[ix]) for p in probabilities]))

            def differences(multiplicity=None, *, equal_stays=False):
                weight = None if multiplicity is None else plan.row_weights(multiplicity)
                deltas = []
                for ix, n_rows, metrics in prepared:
                    w = None if weight is None else weight[ix]
                    if equal_stays:
                        w = np.ones(len(ix)) / n_rows if w is None else w / n_rows
                    values = [m.score(w)["auc"] for m in metrics]
                    deltas.append(values[0] - np.mean(values[1:]))
                return np.array(deltas)

            point = differences()
            stay_point = differences(equal_stays=True)
            coverage = float(permutable[mask].mean()) if mask.any() else 0.
            draws = np.full(BOOTSTRAPS, np.nan)
            stay_draws = np.full(BOOTSTRAPS, np.nan)
            if coverage and np.isfinite(point).all():
                rng = np.random.default_rng(seed(frame, "bootstrap"))
                for b in range(BOOTSTRAPS):
                    multiplicity = plan.draw(rng)
                    draws[b] = differences(multiplicity).mean()
                    stay_draws[b] = differences(multiplicity, equal_stays=True).mean()
            views[name] = {"mean_auc_loss": _nullable(point.mean()) if coverage else None,
                           "per_repeat": [{"repeat": int(r), "auc_loss": _nullable(p) if coverage else None} for r, p in zip(repeats, point)],
                           "uncertainty": _percentile(draws), "coverage": coverage,
                           "equal_stay_mean_auc_loss": _nullable(stay_point.mean()) if coverage else None,
                           "equal_stay_uncertainty": _percentile(stay_draws),
                           "selected_row_fraction": float(selected_rows[mask].mean()) if mask.any() else None,
                           "counts": stage_counts(rows.loc[mask])["total"],
                           "status": "measured" if coverage else "unmeasured_no_exchangeable_rows"}
            draws_for_ranks[name].append(draws)
        results.append({"group": group, "views": views})
    names = [g["name"] for g in groups]
    for name in ("post", "all"):
        if not results:
            continue
        draws = np.stack(draws_for_ranks[name], axis=1)
        kinds = [g.get("kind", "lag" if "lag" in g else "semantic" if g["name"].startswith("semantic:") else "feature") for g in groups]
        ranks = np.full_like(draws, np.nan)
        for kind in dict.fromkeys(kinds):
            measured = [j for j, r in enumerate(results) if kinds[j] == kind and r["views"][name]["mean_auc_loss"] is not None]
            for b in range(BOOTSTRAPS):
                ranks[b, measured] = _ranks(draws[b, measured], [names[j] for j in measured])
            for repeat_index in range(len(repeats)):
                per_repeat = [results[j]["views"][name]["per_repeat"][repeat_index]["auc_loss"] for j in measured]
                rr = _ranks(np.array([np.nan if v is None else v for v in per_repeat]), [names[j] for j in measured])
                for j, rank in zip(measured, rr):
                    results[j]["views"][name]["per_repeat"][repeat_index]["rank"] = _nullable(rank)
        for j, result in enumerate(results):
            result["views"][name]["rank_interval"] = _percentile(ranks[:, j])
            result["views"][name]["rank_family"] = kinds[j]
    return dict(frame=frame, repetitions=20, groups=results, bootstrap_unit="hospitalization",
                interpretation="fixed-model reliance; possible off-manifold blocks; not causal",
                iteration_sd_is_sampling_uncertainty=False, conditional_on_fitted_models=True)


def validate_event_anchors(rows, eligible_events, *, frame):
    """PI anchors come from positive source rows, checked against eligible raw NEC.

    eligible_events supplies stay/event_day (all candidate events, not an
    earliest-any-NEC table). PI candidates are restricted to surgery POD1..31.
    Task A rows supply its own formal first-NEC event_day, checked if raw event
    metadata is provided. Disagreement stops event interpretation only.
    """
    if frame not in SPEC["frames"] or not {"stay", "case", "y", "day", "pod", "event_day"} <= set(rows):
        raise EvaluationError("Missing event alignment metadata")
    surgery = (rows.day - rows.pod).groupby(rows.stay)
    if surgery.nunique().gt(1).any():
        raise EvaluationError("Inconsistent surgery day within stay")
    cases = pd.Index(rows.loc[rows.case.eq(1), "stay"].unique())
    if frame == "PI72-CLEAN":
        positive = rows.loc[rows.y.eq(1)]
        if positive.stay.duplicated().any() or set(positive.stay) != set(cases) or positive.event_day.isna().any():
            raise EvaluationError("PI cases need exactly one positive source row with necbelldtshift")
        anchors = positive.set_index("stay").event_day.reindex(cases)
        if eligible_events is None:
            raise EvaluationError("PI event plots require independently audited eligible raw NEC dates")
    else:
        values = rows.loc[rows.case.eq(1)]
        if values.event_day.isna().any() or values.groupby("stay").event_day.nunique().gt(1).any():
            raise EvaluationError("Task A requires its formal first NEC for every case")
        anchors = values.drop_duplicates("stay").set_index("stay").event_day.reindex(cases)
    if not np.isfinite(anchors.to_numpy(float)).all() or not anchors.eq(np.floor(anchors)).all():
        raise EvaluationError("Case event dates must be finite calendar-day integers")
    if frame == "PI72-CLEAN" and not (positive.event_day - positive.day).isin([1, 2, 3]).all():
        raise EvaluationError("PI positive-row event lead must be one to three calendar days")
    if eligible_events is not None:
        if not {"stay", "event_day"} <= set(eligible_events) or eligible_events[["stay", "event_day"]].isna().any().any():
            raise EvaluationError("Invalid independent NEC event metadata")
        events = eligible_events.loc[eligible_events.stay.isin(cases)].copy()
        if frame == "PI72-CLEAN":
            event_pod = events.event_day - events.stay.map(surgery.first())
            events = events.loc[event_pod.between(1, 31)]
        first = events.groupby("stay").event_day.min().reindex(cases)
        if first.isna().any() or not first.eq(anchors).all():
            raise EvaluationError("Event anchor disagrees with first eligible NEC; stop event plot")
    return anchors


def _aligned_values(rows, values, features):
    if not isinstance(values, pd.DataFrame) or not values.index.is_unique or not values.columns.is_unique:
        raise EvaluationError("Values/availability must have a unique source-row index")
    keys = pd.Index(rows.row_key)
    if len(values) != len(rows) or not keys.isin(values.index).all() or not values.index.isin(keys).all() or not set(features) <= set(values):
        raise EvaluationError("Values/availability do not exactly cover source rows/features")
    return values.reindex(keys)[features].to_numpy()


@threadpool_limits.wrap(limits=2)
def trajectories(rows, values, feature_definitions, *, frame, alignment, eligible_events=None, availability=None):
    """T3b census distributions, correctly truncated, with 1,000 stay draws.

    rows: unique row_key/stay/day/pod/y/case and (for event alignment) event_day.
    values: original-unit DataFrame indexed by row_key. Definitions map source
    names to kind=continuous/binary, unknown_codes, optional counter=True.
    Optional availability is a boolean table indexed identically. PI cases end
    at their positive retained day (inclusive); A cases end strictly before
    formal first NEC. Existing eligible=False/unlabelled rows are excluded.
    No dates, observations or pseudo-events are created or interpolated.
    """
    required = {"row_key", "stay", "day", "pod", "y", "case"}
    if frame not in SPEC["frames"] or alignment not in ("surgery", "event") or not required <= set(rows):
        raise EvaluationError("Invalid trajectory frame/alignment/metadata")
    rows = rows.reset_index(drop=True).copy()
    if rows.empty or rows.row_key.duplicated().any() or rows.duplicated(["stay", "day"]).any():
        raise EvaluationError("Empty or duplicate census trajectory keys")
    if rows[["row_key", "stay", "day", "pod", "case"]].isna().any().any() or not rows.case.isin([0, 1]).all():
        raise EvaluationError("Invalid trajectory identity/case metadata")
    if any(rows[c].dtype == np.dtype("float32") for c in ("stay", "row_key")):
        raise EvaluationError("Trajectory identifiers must retain source precision")
    calendar = rows[["day", "pod"]].to_numpy(float)
    if not np.isfinite(calendar).all() or not np.equal(calendar, np.floor(calendar)).all():
        raise EvaluationError("Trajectories require integer calendar days")
    if (rows.day - rows.pod).groupby(rows.stay).nunique().gt(1).any():
        raise EvaluationError("Inconsistent surgery day in trajectory metadata")
    if rows.groupby("stay").case.nunique().gt(1).any() or not rows.y.dropna().isin([0, 1]).all():
        raise EvaluationError("Conflicting case status or invalid labels")
    if ((rows.y == 1) & (rows.case == 0)).any():
        raise EvaluationError("Positive trajectory row assigned to a source control")
    features = list(feature_definitions)
    if not features:
        raise EvaluationError("No trajectory features")
    X = _aligned_values(rows, values, features).astype(float)
    if np.isinf(X).any():
        raise EvaluationError("Infinite original-unit trajectory values")
    gates_known = availability is not None
    gates = np.ones_like(X, bool) if availability is None else _aligned_values(rows, availability, features)
    if not np.isin(gates, [0, 1]).all():
        raise EvaluationError("Availability must be explicitly boolean and complete")
    gates = gates.astype(bool)
    if "eligible" in rows and not rows.eligible.isin([0, 1]).all():
        raise EvaluationError("Eligibility must be explicit boolean")
    retained = rows.y.notna().to_numpy() & (rows.eligible.to_numpy(bool) if "eligible" in rows else True)
    if frame == "PI72-CLEAN":
        positive = rows.loc[retained & rows.y.eq(1).to_numpy()]
        cases = set(rows.loc[rows.case.eq(1), "stay"])
        if positive.stay.duplicated().any() or set(positive.stay) != cases:
            raise EvaluationError("Census PI cases must each retain exactly one positive label day")
        boundary = rows.stay.map(positive.set_index("stay").day)
        retained &= (rows.case.eq(0) | rows.day.le(boundary)).to_numpy()
    else:
        if "event_day" not in rows or rows.loc[rows.case.eq(1), "event_day"].isna().any():
            raise EvaluationError("Task A trajectory truncation needs formal first NEC")
        if rows.loc[rows.case.eq(1)].groupby("stay").event_day.nunique().gt(1).any():
            raise EvaluationError("Task A first NEC changes within one stay")
        retained &= (rows.case.eq(0) | rows.day.lt(rows.event_day)).to_numpy()
    anchors = validate_event_anchors(rows.loc[retained], eligible_events, frame=frame) if alignment == "event" else None
    rows, X, gates = rows.loc[retained].reset_index(drop=True), X[retained], gates[retained]
    if rows.empty:
        raise EvaluationError("No retained trajectory rows")
    # Previous means the preceding retained calendar day of this same stay.
    index = pd.MultiIndex.from_frame(rows[["stay", "day"]])
    previous = index.get_indexer(pd.MultiIndex.from_arrays([rows.stay, rows.day - 1]))
    adjacent = previous >= 0
    safe_previous = np.maximum(previous, 0)
    case, pods = rows.case.to_numpy(bool), rows.pod.to_numpy()
    offset = rows.day.to_numpy() - rows.stay.map(anchors).to_numpy() if anchors is not None else pods
    days = range(-14, 0) if alignment == "event" else range(-14, 32)
    plan = StayBootstrap(rows)
    descriptors = [(day, cohort) for day in days for cohort in (("case", "standardized_control") if alignment == "event" else ("case", "control"))]

    # Cache only eligibility/index structure. No bootstrap POD weights or
    # outcome-derived pseudo-event dates are cached or imposed on controls.
    groups = []
    for day, cohort in descriptors:
        case_ix = np.flatnonzero(case & (offset == day))
        if cohort != "standardized_control":
            ix = case_ix if cohort == "case" else np.flatnonzero(~case & (pods == day))
            groups.append((ix, None))
            continue
        case_pods = np.unique(pods[case_ix])
        ix = np.flatnonzero(~case & np.isin(pods, case_pods))
        matches = [(case_ix[pods[case_ix] == pod], np.flatnonzero(pods[ix] == pod)) for pod in case_pods]
        groups.append((ix, matches))

    def weights_for(k, row_weight):
        ix, matches = groups[k]
        if matches is None:
            return row_weight[ix], True
        weights = np.zeros(len(ix))
        valid, total_cases = True, 0.
        for case_ix, control_local in matches:
            case_mass = row_weight[case_ix].sum()
            total_cases += case_mass
            if not case_mass:
                continue
            control_mass = row_weight[ix[control_local]].sum()
            if not control_mass:
                valid = False  # never renormalize away unmatched case PODs
            else:
                weights[control_local] = row_weight[ix[control_local]] * case_mass / control_mass
        return weights, valid and total_cases > 0

    output = []
    for j, feature in enumerate(features):
        definition = feature_definitions[feature]
        if definition.get("kind") not in ("continuous", "binary") or "unknown_codes" not in definition:
            raise EvaluationError("Declare continuous/binary kind and dictionary unknown_codes (possibly empty)")
        x, gate = X[:, j], gates[:, j]
        unknown = np.isin(x, definition.get("unknown_codes", []))
        known = np.isfinite(x) & ~unknown & gate
        if definition["kind"] == "binary" and not np.isin(x[known], [0, 1]).all():
            raise EvaluationError("Nonbinary known values; declare dictionary unknown codes explicitly")
        pair = adjacent & known & known[safe_previous]
        change = x - x[safe_previous]
        numeric_change = pair & (np.abs(change) > 1e-6)
        missing_switch = adjacent & (known != known[safe_previous])
        gate_switch = adjacent & (gate != gate[safe_previous])

        orders = []
        for ix, _ in groups:
            observed = np.flatnonzero(known[ix])
            pairs = np.flatnonzero(pair[ix])
            orders.append((observed[np.argsort(x[ix[observed]], kind="stable")],
                           pairs[np.argsort(change[ix[pairs]], kind="stable")]))

        def sorted_quantiles(v, w, quantiles):
            if not len(w) or not w.sum():
                return np.full(len(quantiles), np.nan)
            cumulative = np.cumsum(w)
            index = np.searchsorted(cumulative, np.asarray(quantiles) * cumulative[-1], side="left")
            return v[np.minimum(index, len(v) - 1)]

        def statistics(k, weights, valid):
            if not valid:
                return np.nan, np.nan, np.full(3, np.nan)
            ix = groups[k][0]
            observed, pairs = orders[k]
            w, v = weights[observed], x[ix[observed]]
            quantiles = sorted_quantiles(v, w, [.25, .5, .75])
            estimate = float(np.dot(w, v) / w.sum()) if definition["kind"] == "binary" and w.sum() else quantiles[1]
            paired_delta = sorted_quantiles(change[ix[pairs]], weights[pairs], [.5])[0]
            return (estimate, paired_delta, quantiles) if valid else (np.nan, np.nan, np.full(3, np.nan))

        summaries = []
        for k, (day, cohort) in enumerate(descriptors):
            weights, valid = weights_for(k, np.ones(len(rows)))
            ix = groups[k][0]
            selected = ix[weights > 0]
            estimate, paired_delta, quantiles = statistics(k, weights, valid)
            denominator = float((weights * known[ix]).sum())
            total_weight = weights.sum()
            summaries.append({"feature": feature, "kind": definition["kind"], "day": day, "cohort": cohort,
                              "unit": definition.get("unit"),
                              "estimate": _nullable(estimate), "q25": _nullable(quantiles[0]), "median": _nullable(quantiles[1]), "q75": _nullable(quantiles[2]),
                              "known_positive_fraction": _nullable(estimate) if definition["kind"] == "binary" else None,
                              "eligible_rows": len(selected), "unique_stays": int(rows.iloc[selected].stay.nunique()),
                              "case_stays": int(rows.iloc[selected[case[selected]]].stay.nunique()),
                              "nonmissing": int(known[selected].sum()), "unknown_codes": int(unknown[selected].sum()),
                              "recorded_nonmissing": int(np.isfinite(x[selected]).sum()),
                              "nan_rows": int(np.isnan(x[selected]).sum()), "unavailable_rows": int((~gate[selected]).sum()),
                              "weighted_unknown_fraction": float(np.dot(weights, unknown[ix]) / total_weight) if total_weight and valid else None,
                              "weighted_nan_fraction": float(np.dot(weights, np.isnan(x[ix])) / total_weight) if total_weight and valid else None,
                              "weighted_unavailable_fraction": float(np.dot(weights, ~gate[ix]) / total_weight) if total_weight and valid else None,
                              "known_weight_denominator": denominator,
                              "adjacent_calendar_pairs": int(adjacent[selected].sum()), "available_pairs": int(pair[selected].sum()),
                              "paired_delta_median": _nullable(paired_delta),
                              "actual_changes": int(numeric_change[selected].sum()) if not definition.get("counter", False) else 0,
                              "counter_advances": int((pair[selected] & (change[selected] > 1e-6)).sum()) if definition.get("counter", False) else 0,
                              "counter_decreases": int((pair[selected] & (change[selected] < -1e-6)).sum()) if definition.get("counter", False) else 0,
                              "missingness_switches": int(missing_switch[selected].sum()),
                              "availability_switches": int(gate_switch[selected].sum()) if gates_known else None,
                              "standardization_defined": bool(valid), "undefined_original": not np.isfinite(estimate)})
        samples = np.full((1000, len(descriptors), 2), np.nan)
        rng = np.random.default_rng(seed(frame, "bootstrap"))
        for b in range(1000):
            weight = plan.row_weights(plan.draw(rng))
            for k in range(len(descriptors)):
                w, valid = weights_for(k, weight)
                estimate, delta, _ = statistics(k, w, valid)
                samples[b, k] = estimate, delta
        for k, item in enumerate(summaries):
            item["estimate_uncertainty"] = _percentile(samples[:, k, 0], attempts=1000, minimum=950)
            item["paired_delta_uncertainty"] = _percentile(samples[:, k, 1], attempts=1000, minimum=950)
            if item["estimate"] is None:
                item["estimate_uncertainty"]["ci95"] = None
            if item["paired_delta_median"] is None:
                item["paired_delta_uncertainty"]["ci95"] = None
            output.append(item)
    return dict(frame=frame, alignment=alignment, population="source_census_retained_rows" if frame == "PI72-CLEAN" else "formal_first_NEC_risk_set",
                input_rows=len(retained), retained_rows=int(retained.sum()), excluded_rows=int((~retained).sum()),
                attempted_draws=1000, summaries=output, quantile_definition="inverse weighted empirical CDF",
                at_risk="retained eligible observations; not necessarily NEC-naive" if frame == "PI72-CLEAN" else "formal eligible observations",
                controls="all observed control days at matching POD; case-POD standardized at each lead",
                retrospective_case_conditioning=True, interpolated=False, causal_interpretation=False)


@threadpool_limits.wrap(limits=2)
def paired_pod_changes(targets, packets, *, frame, feature_universe):
    """T3c later-minus-earlier stay-mean |SHAP| for adjacent POD bins.

    Same held-out fitted model and stays observed in both bins only. This
    descriptive conditioning never changes prediction eligibility. Multiplicity
    is still shared over the evaluated stay union and repeats.
    """
    rows = validate_targets(targets, frame)
    lookup, seen, models = rows.set_index(["repeat", "row_key"]), set(), {}
    features, repeat_values = list(feature_universe), sorted(rows.repeat.unique())
    bin_cells = {f"POD{lo}-{hi}": {} for lo, hi in POD_BINS}
    for packet in packets:
        part = _packet_rows(packet, lookup, seen, models)
        check_shap_additivity(packet["phi"], packet["base"], packet["margin"])
        grouped = group_shap(packet["phi"], packet["layout"], feature_universe=features)["abs_sum"]
        repeat = int(part.repeat.iloc[0])
        fold = int(part.fold.iloc[0]) if "fold" in part else -1
        for name, mask in time_masks(part, "pod").items():
            if name == "preop":
                continue
            for stay, ix in part.loc[mask].groupby("stay", sort=False).indices.items():
                ix = np.flatnonzero(mask)[ix]
                key = (repeat, fold, stay)
                count, sums = bin_cells[name].get(key, (0, np.zeros(len(features))))
                bin_cells[name][key] = count + len(ix), sums + grouped[ix].sum(axis=0)
    if len(seen) != len(rows):
        raise EvaluationError("Paired POD SHAP coverage is incomplete")
    plan, result = StayBootstrap(rows), []
    names = list(bin_cells)
    for earlier, later in zip(names[:-1], names[1:]):
        common = set(bin_cells[earlier]) & set(bin_cells[later])
        cells = {}
        for repeat, fold, stay in common:
            n0, s0 = bin_cells[earlier][repeat, fold, stay]
            n1, s1 = bin_cells[later][repeat, fold, stay]
            if (repeat, stay) in cells:
                raise EvaluationError("Same stay attributed by multiple models in a repeat")
            cells[repeat, stay] = (1, (s1 / n1 - s0 / n0)[None, :])
        counts = {str(r): sum(rep == r for rep, _ in cells) for r in repeat_values}
        compiled = _compile_cells(cells, repeat_values, plan)
        point = _cell_means(compiled)
        samples = np.full((BOOTSTRAPS, len(features)), np.nan)
        if all(n >= 20 for n in counts.values()) and point is not None:
            rng = np.random.default_rng(seed(frame, "bootstrap"))
            for b in range(BOOTSTRAPS):
                means = _cell_means(compiled, plan.draw(rng))
                if means is not None:
                    samples[b] = means.mean(axis=0)[0]
        result.append(dict(earlier=earlier, later=later, paired_stays_by_repeat=counts,
                           features=[dict(feature=f, later_minus_earlier=_nullable(point.mean(axis=0)[0, j]) if point is not None else None,
                                          uncertainty=_percentile(samples[:, j])) for j, f in enumerate(features)]))
    return dict(frame=frame, comparisons=result, conditioning="same fitted model; observed in both adjacent POD bins",
                weighting="equal repeats; equal paired stays", eligibility_filter=False)
