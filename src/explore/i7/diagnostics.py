"""Exploratory I7 slices and learning-curve methods; excluded source-model curves are not fitted."""
from dataclasses import dataclass, replace
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
from pf_nec import contract as c, evaluate as ev, features as f, run, selectors as sel, trees, windows
DIAGNOSTICS = c.SPEC["diagnostics"]
FRAME = "PI72-CLEAN"


@dataclass(frozen=True)
class SliceArm:
    """Frozen diagnostic identities, distinct from the accepted selection menu."""
    recipe: str
    window: int
    learner: str = DIAGNOSTICS["learner"]
    representation: str = DIAGNOSTICS["representation"]
    frame: str = FRAME

    @property
    def id(self):
        return f"DIAG-{self.recipe}-w{self.window}-R1-LGB"

    @property
    def k(self):
        return len(DIAGNOSTICS["feature_slices"][self.recipe])

    @property
    def columns(self):
        return trees.input_columns(self.k, self.window, self.representation)

    @property
    def name(self):
        return "B-only-w1" if self.recipe == "B" else f"{self.recipe}-w{self.window}"


def slice_arms():
    return [SliceArm(recipe, window) for recipe in ("B+C", "B+S", "B+C+S")
            for window in DIAGNOSTICS["windows"]] + [SliceArm("B", DIAGNOSTICS["B_only_window"])]


def validate_slice_matrix(matrix, arm, context, learner):
    declared = {a.id: a for a in slice_arms()}
    if arm not in declared or context.frame != FRAME or learner != DIAGNOSTICS["learner"]:
        raise c.ContractError("Undeclared diagnostic learner/layout")
    config = declared[arm]
    columns = tuple(DIAGNOSTICS["feature_slices"][config.recipe])
    c.validate_keys(matrix.keys, frame=context.frame)
    c.assert_predictors(columns, bank=c.D5_BANK)
    concepts = columns if config.window == 1 else columns * config.window + (None,) * (config.window + 1)
    names = columns if config.window == 1 else tuple(f"{name}__lag{lag}" for lag in range(config.window) for name in columns)
    if config.window > 1:
        names += tuple(f"row_observed__lag{lag}" for lag in range(config.window)) + ("window_observed_day_count",)
    if matrix.values.shape != (len(matrix.keys), config.columns) or tuple(matrix.columns) != names or tuple(matrix.concepts) != concepts:
        raise c.ContractError("Diagnostic columns/order differ from frozen slice")


def fit_slice_tree(train, y, *, learner, context, arm, rounds=None,
                   valid=None, valid_y=None, valid_post=None, guard=None):
    """Frozen LGB recipe for diagnostics omitted from trees.validate_matrix.

    No selection-menu/spec mutation or global monkeypatch. Lifecycle is the
    TreeEngine three inner folds -> median rounds -> outer refit. No scaler:
    like accepted native LGB, train/validation values retain gated NaNs.
    """
    c.reset_threads()
    validate_slice_matrix(train, arm, context, learner)
    y = trees._labels(y, len(train.keys))
    if train.values.dtype != np.float32 or np.isinf(train.values).any():
        raise c.ContractError("Diagnostic inputs must be float32; no infinities")
    if valid is not None:
        validate_slice_matrix(valid, arm, context, learner)
        if context.inner not in (0, 1, 2) or context.pool != "inner-fit" or rounds is not None:
            raise c.ContractError("Diagnostic stopping requires an inner fit")
        if train.keys[c.H].isin(valid.keys[c.H]).any():
            raise c.ContractError("Diagnostic fit/validation stay overlap")
        if valid.values.dtype != np.float32 or np.isinf(valid.values).any():
            raise c.ContractError("Invalid diagnostic validation values")
    if guard:
        guard()
    import lightgbm as lgb
    params = trees.lgb_params(context.seed("model", arm))
    data = lgb.Dataset(train.values, label=y, feature_name=list(train.columns), free_raw_data=True)
    kwargs = {}
    if valid is not None:
        valid_y = trees._labels(valid_y, len(valid.keys))
        trees.primary_auc(valid_y, np.full(len(valid_y), .5), valid_post, context.frame)
        validation = lgb.Dataset(valid.values, label=valid_y, reference=data, free_raw_data=True)
        def auc(probability, dataset):
            return "primary_auc", trees.primary_auc(dataset.get_label(), probability, valid_post, context.frame), True
        kwargs = dict(valid_sets=[validation], valid_names=["inner-validation"], feval=auc,
                      callbacks=[lgb.early_stopping(c.SPEC["learners"]["LGB"]["patience"], verbose=False)])
        count = c.SPEC["learners"]["LGB"]["max_rounds"]
    else:
        if not isinstance(rounds, int) or not 1 <= rounds <= c.SPEC["learners"]["LGB"]["max_rounds"]:
            raise c.ContractError("Diagnostic outer refit requires median inner rounds")
        count = rounds
    model = lgb.train(params, data, num_boost_round=count, **kwargs)
    count = int(model.best_iteration) if valid is not None else rounds
    if count < 1:
        raise c.ContractError("Diagnostic LGB returned no stopping iteration")
    if guard:
        guard()
    return trees.TreeFit(model, learner, count, tuple(train.columns), params)


class SliceEngine(run.TreeEngine):
    def selected(self, rows, recipe, labels, context, arm):
        if recipe not in DIAGNOSTICS["feature_slices"] or arm not in {a.id for a in slice_arms()}:
            raise c.ContractError("Unknown frozen diagnostic slice")
        # These are fixed lists: no full-pool ranking, encoding or imputation.
        return f.select_features(rows, DIAGNOSTICS["feature_slices"][recipe])


    # Accepted TreeEngine fit lifecycle, specialised only at the learner call.
    # The accepted module has no learner-injection API for diagnostics arms.
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
                    model = fit_slice_tree(train, y[fit_indices], learner=arm.learner, context=context, arm=arm.id,
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
                    run.write_json(directory / "result.json", result)
                return result

            results.append(self.jobs.execute(f"inner/{arm.id}/i{inner}", binding, work))
        del rows
        gc.collect()
        return {"arm": arm.id, "score": float(np.mean([r["score"] for r in results])),
                "rounds": trees.median_rounds([r["rounds"] for r in results]), "inner": results,
                "binding": binding}

    def fit(self, arm, inner_result=None):
        # Resume canonical diagnostic inner jobs and the final model job.
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
                model = fit_slice_tree(train, y, learner=arm.learner, context=context, arm=arm.id,
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
                run.write_json(directory / "result.json", info)
            return info

        info = self.jobs.execute(f"outer/{arm.id}", binding, work)
        del rows
        gc.collect()
        return info


def slice_provider(context):
    """Use PI metadata and exact retained rows; project D5 before materialising.

    Masks come directly from accepted build_features(bank=D5SAFE), including
    z_* context columns. This is the same D5 source used by pi_provider.
    """
    _, metadata = run.pi_provider(context)

    def loader(recipe, partition):
        columns = DIAGNOSTICS["feature_slices"][recipe]
        core = c.import_core()
        private_environment()  # reset legacy import's temporary directory/thread side effects
        original = core.load_source_rows(context.repeat, partition)
        keys = original[[c.H, c.DATE, c.SOURCE_ROW]].copy()
        anchors = core._validated_table("stays.parquet", columns=list(c.ANCHORS),
                                        filters=[(c.H, "in", keys[c.H].unique().tolist())])
        values = core._validated_table("D5.parquet", columns=[c.SOURCE_ROW, *columns],
                                       filters=[(c.SOURCE_ROW, "in", keys[c.SOURCE_ROW].tolist())])
        return f.build_features(keys, values.set_index(c.SOURCE_ROW), anchors, bank=c.D5_BANK)

    def scoped_metadata(partition):
        result = metadata(partition)
        private_environment()
        return result
    return loader, scoped_metadata


def subset_provider(loader, metadata, context, fraction):
    train = metadata("train")
    stays = learning_curve_stays(train, context, fraction)
    subset = train.loc[train.stay.isin(stays)].copy()
    def child_loader(recipe, partition):
        rows = loader(recipe, partition)
        if partition == "train":
            return f.take_rows(rows, subset.row_key)
        return rows
    def child_metadata(partition):
        return subset.copy() if partition == "train" else metadata("test")
    return child_loader, child_metadata


def training_counts(meta):
    return dict(train_stays=int(meta.stay.nunique()),
                train_case_stays=int(meta.groupby("stay").case.max().sum()),
                positive_rows=int(meta.y.sum()))


def window_distribution(rows, targets, window):
    """Aggregate every target, including padding and incomplete history."""
    identity = c.validate_keys(rows.keys, frame=rows.frame)
    rows = f.take_rows(rows, targets.row_key)
    if not np.array_equal(rows.keys[c.H], targets.stay):
        raise c.ContractError("Distribution stay alignment differs")
    bins = {"preop": targets.pod.lt(0).to_numpy()}
    bins.update({f"POD{lo}-{hi}": targets.pod.between(lo, hi).to_numpy()
                 for lo, hi in c.SPEC["metrics"]["POD_strata"]})
    accum = {name: dict(rows=0, missing_cells=0, cells=0, observed_missing_cells=0,
                       observed_cells=0, structural_unavailable_cells=0,
                       available_window_days={str(d): 0 for d in range(window + 1)}) for name in bins}
    offset = 0
    for batch in windows.iter_windows(rows, window=window):
        size = len(batch.keys)
        for name, mask in bins.items():
            keep = mask[offset:offset + size]
            values, observed = batch.values[keep], batch.row_observed[keep]
            item = accum[name]
            item["rows"] += int(keep.sum())
            item["cells"] += int(values.size)
            item["missing_cells"] += int(np.isnan(values).sum())
            on_rows = np.broadcast_to(observed[:, :, None], values.shape)
            item["observed_cells"] += int(on_rows.sum())
            item["observed_missing_cells"] += int((np.isnan(values) & on_rows).sum())
            item["structural_unavailable_cells"] += int((~batch.available[keep] & on_rows).sum())
            count = np.bincount(observed.sum(axis=1), minlength=window + 1)
            for d, n in enumerate(count):
                item["available_window_days"][str(d)] += int(n)
        offset += size
    for item in accum.values():
        item["missing_rate_including_padding"] = item["missing_cells"] / item["cells"] if item["cells"] else None
        item["missing_rate_observed_rows"] = item["observed_missing_cells"] / item["observed_cells"] if item["observed_cells"] else None
    return dict(window=window, columns=list(rows.values.columns), strata=accum,
                semantics="retained source days; all targets; no future survival conditioning")


def paired_pod_gains(targets, predictions, contrasts, *, draws=ev.BOOTSTRAPS):
    """Matched POD gains with shared union-stay resampling; descriptive only."""
    rows, aligned = ev.align_predictions(targets, predictions, frame=FRAME)
    repeats = sorted(rows.repeat.unique())
    plan = ev.StayBootstrap(rows)
    rng = np.random.default_rng(ev.seed(FRAME, "bootstrap"))
    prepared, support, counts = {}, {}, {}
    for lo, hi in c.SPEC["metrics"]["POD_strata"]:
        name = f"POD{lo}-{hi}"
        positions = [np.flatnonzero(rows.pod.between(lo, hi) & rows.repeat.eq(r)) for r in repeats]
        counts[name] = [ev.stage_counts(rows.iloc[ix])["total"] for ix in positions]
        support[name] = all(x["positive_stays"] >= 10 and x["negative_stays"] >= 10 for x in counts[name])
        prepared[name] = {arm: [(ix, ev.BinaryMetric(rows.y.to_numpy()[ix], p[ix])) for ix in positions]
                          for arm, p in aligned.items()}
    def scores(weight=None):
        result = []
        for name, metrics in prepared.items():
            means = {arm: np.mean([metric.score(None if weight is None else weight[ix])["auc"]
                                   for ix, metric in parts]) for arm, parts in metrics.items()}
            result.extend([means[a] - means[b] if support[name] else np.nan for a, b in contrasts])
        return np.array(result)
    point = scores()
    samples = np.array([scores(plan.row_weights(plan.draw(rng))) for _ in range(draws)])
    intervals = ev.bootstrap_intervals(point, samples, [])
    result, offset = {}, 0
    for name in prepared:
        result[name] = dict(counts_by_repeat={str(r): count for r, count in zip(repeats, counts[name])},
                            auc_supported=support[name], comparisons=[])
        for a, b in contrasts:
            result[name]["comparisons"].append(dict(arm=a, reference=b, delta=nullable(point[offset]),
                                                    **intervals["ordinary"][offset]))
            offset += 1
    return dict(strata=result, attempted_draws=draws, intervals="descriptive conditional pointwise",
                bootstrap_universe_stays=len(plan.stays))


def nullable(value):
    return float(value) if np.isfinite(value) else None


def learning_slopes(targets, predictions, counts, *, draws=ev.BOOTSTRAPS):
    rows, aligned = ev.align_predictions(targets, predictions, frame=FRAME)
    repeats = sorted(rows.repeat.unique())
    fractions = DIAGNOSTICS["learning_curve"]["fractions"]
    prepared = {}
    for model in ("LGB", "TCN"):
        for repeat in repeats:
            ix = np.flatnonzero(rows.repeat.eq(repeat))
            for fraction in fractions:
                name = f"{model}-{fraction:g}"
                prepared[model, repeat, fraction] = (ix, ev.BinaryMetric(rows.y.to_numpy()[ix], aligned[name][ix],
                                                                       rows.pod.to_numpy()[ix] >= 0))
    def slopes(weight=None):
        values = {}
        for model in ("LGB", "TCN"):
            for repeat in repeats:
                ncases = np.array([counts[str(repeat)][str(frac)]["train_case_stays"] for frac in fractions])
                if (ncases <= 0).any() or len(np.unique(ncases)) < 2:
                    raise c.ContractError("Learning slope requires positive, varying training case counts")
                x = np.log2(ncases)
                y = []
                for frac in fractions:
                    ix, metric = prepared[model, repeat, frac]
                    y.append(metric.score(None if weight is None else weight[ix])["auc"])
                values[model, repeat] = float(np.dot(x - x.mean(), y) / np.dot(x - x.mean(), x - x.mean()))
        return values
    point = slopes()
    plan = ev.StayBootstrap(rows)
    rng = np.random.default_rng(ev.seed(FRAME, "bootstrap"))
    samples = np.array([[np.mean([s[model, r] for r in repeats]) for model in ("LGB", "TCN")]
                        for _ in range(draws) for s in [slopes(plan.row_weights(plan.draw(rng)))]])
    estimates = np.array([np.mean([point[model, r] for r in repeats]) for model in ("LGB", "TCN")])
    intervals = ev.bootstrap_intervals(estimates, samples, [])
    return {model: dict(per_repeat={str(r): nullable(point[model, r]) for r in repeats},
                        mean_slope=nullable(estimates[i]), **intervals["ordinary"][i],
                        units="AUROC_post per doubling of training case stays", attempted_draws=draws,
                        interval_type="descriptive, fixed predictions, shared union-stay bootstrap")
            for i, model in enumerate(("LGB", "TCN"))}

import hashlib


def learning_curve_stays(metadata, context, fraction):
    """Shared tree/network nested prefixes; hash exact typed stay representations."""
    if metadata.groupby("stay").case.nunique().gt(1).any():
        raise c.ContractError("Learning curve source case stratum varies within stay")
    table = metadata.groupby("stay", sort=True).case.max()
    salt = context.seed("learning-curve")
    chosen = []
    for label in (0, 1):
        stays = table.index[table.eq(label)].tolist()
        ordered = sorted(stays, key=lambda stay: hashlib.sha256(
            f"{salt}|{type(stay).__name__}|{stay!r}".encode()).digest())
        chosen.extend(ordered[:int(np.floor(fraction * len(ordered)))])
    return chosen

def private_environment():
    from pf_nec import config
    config.initialize()
    c.reset_threads()


def main():
    import argparse
    from pf_nec import config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--arm", choices=[a.name for a in slice_arms()], default="B-only-w1")
    args = parser.parse_args()
    private_environment()
    ctx = sel.FitContext(FRAME, args.repeat)
    engine = SliceEngine(ctx, *slice_provider(ctx), config.RUN_ROOT / "explore-i7" / ctx.key)
    arm = next(a for a in slice_arms() if a.name == args.arm)
    result = engine.apply(arm)
    run.write_json(engine.directory / f"{arm.name}.json", {**result, "promoted": False})
    print(json.dumps({"arm": arm.id, "status": "completed", "promoted": False}))


if __name__ == "__main__":
    main()
