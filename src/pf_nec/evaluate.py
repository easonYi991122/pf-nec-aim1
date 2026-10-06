"""RX-D1 conditional evaluation on a caller-supplied, complete target universe.

No data loading, fitting, registration or file writes occur here. Public tables
use repeat/row_key/stay/y/pod/case; rename source columns explicitly upstream.
Task A additionally supplies fold (0..4). See docs/EVALUATION.md and docs/en/EVALUATION.md for examples.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

# This existing bridge imports only pure WP0 definitions, not core's driver or
# its output-directory/three-thread side effects. Old metrics' four-arm CI does
# not implement this protocol and is deliberately not called.
from .legacy.wp0 import decomposition as pi_decomposition

SPEC = json.loads(Path(__file__).with_name("spec_v1.json").read_text())
KEY = ["repeat", "row_key"]
META = ["stay", "y", "pod", "case"]
BOOTSTRAPS, MIN_VALID = 2000, 1990
CROSS = "cross_stage_positive_vs_preop_negative_auc"
METRICS = ("AUROC_post", "AUROC_total", CROSS, "AUROC_preop_within_phase",
           "AP_post", "AP_total", "Brier_post_sampled", "Brier_total_sampled")
MATCHED_W1 = "T4-MATCHED-W1"


class EvaluationError(ValueError):
    """An alignment, inference or accounting contract was violated."""


def seed(frame, purpose, *, repeat=-1, fold=-1, arm="shared", inner=-1, trial=-1, init=-1):
    """Frozen RX-D1 seed; shared resampling never uses an individual arm ID."""
    if frame not in SPEC["frames"] or purpose not in SPEC["seeds"]["purpose_values"]:
        raise EvaluationError("Unknown frame or seed purpose")
    value = f"RX-D1-v1|{frame}|{repeat}|{fold}|{purpose}|{arm}|{inner}|{trial}|{init}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little") % (2**31 - 1)


def _require_columns(table, columns):
    if not table.columns.is_unique or not set(columns) <= set(table.columns):
        raise EvaluationError(f"Required unique columns: {columns}")


def validate_targets(targets, frame):
    """Check identity without rounding/casting identifiers or changing risk sets."""
    if frame not in SPEC["frames"]:
        raise EvaluationError("Frames cannot be pooled or inferred")
    _require_columns(targets, KEY + META)
    if targets.empty or targets[KEY + META].isna().any().any():
        raise EvaluationError("Empty targets or missing target metadata")
    for column in ("stay", "row_key"):
        if targets[column].dtype == np.dtype("float32"):
            raise EvaluationError("float32 identifiers have lost source precision")
        if pd.api.types.is_numeric_dtype(targets[column]) and not np.isfinite(targets[column]).all():
            raise EvaluationError("Nonfinite identifiers")
    if targets.duplicated(KEY).any() or targets.duplicated(["repeat", "stay", "pod"]).any():
        raise EvaluationError("Duplicate target key or stay/calendar day")
    if not targets.y.isin([0, 1]).all() or not targets.case.isin([0, 1]).all():
        raise EvaluationError("Labels and source case strata must be binary")
    if ((targets.y == 1) & (targets.case == 0)).any():
        raise EvaluationError("Positive rows cannot belong to source controls")
    if not np.isfinite(targets.pod).all() or not targets.pod.eq(np.floor(targets.pod)).all():
        raise EvaluationError("POD must be a finite calendar-day integer")
    if not set(targets.repeat) <= set(SPEC["repeats"][frame]["confirm"]):
        raise EvaluationError("Wrong repeat numbering for frame")
    if "frame" in targets and not targets.frame.eq(frame).all():
        raise EvaluationError("Mixed evaluation frames")
    if "post_surg" in targets and not targets.post_surg.eq(targets.pod.ge(0)).all():
        raise EvaluationError("POD/stage disagreement (POD0 is postoperative)")
    if targets.groupby("stay").case.nunique().gt(1).any():
        raise EvaluationError("Source case stratum differs across repeats")
    if targets.groupby("row_key")[META].nunique().gt(1).any().any():
        raise EvaluationError("Target metadata differs across repeats")
    if frame == "PI72-CLEAN" and ((targets.pod < 0) & (targets.y == 1)).any():
        raise EvaluationError("PI72-CLEAN cannot have preoperative positive rows")
    if frame == "A-formal":
        _require_columns(targets, ["fold"])
        if not targets.fold.isin(range(5)).all():
            raise EvaluationError("Task A requires the original five OOF folds")
        if targets.groupby(["repeat", "stay"]).fold.nunique().gt(1).any():
            raise EvaluationError("One stay appears in multiple OOF folds")
        if any(set(group.fold) != set(range(5)) for _, group in targets.groupby("repeat")):
            raise EvaluationError("Task A requires all five folds within each repeat")
    return targets.reset_index(drop=True).copy()


def align_predictions(targets, predictions, *, frame):
    """Reindex every arm to independent targets; missing/extra keys always fail.

    Each prediction table carries all target metadata and a probability column.
    Passing vectors, joining arms to one another, or intersecting keys is unsafe.
    """
    rows = validate_targets(targets, frame)
    if not predictions:
        raise EvaluationError("No prediction arms")
    index = pd.MultiIndex.from_frame(rows[KEY])
    metadata = META + (["fold"] if frame == "A-formal" else [])
    aligned = {}
    for arm, table in predictions.items():
        _require_columns(table, KEY + metadata + ["probability"])
        if table[KEY].isna().any().any() or table.duplicated(KEY).any():
            raise EvaluationError(f"{arm}: duplicate/missing prediction keys")
        actual = pd.MultiIndex.from_frame(table[KEY])
        if len(actual) != len(index) or not index.isin(actual).all() or not actual.isin(index).all():
            raise EvaluationError(f"{arm}: prediction coverage differs from targets; no inner join")
        table = table.set_index(KEY).reindex(index).reset_index()
        if not table[metadata].eq(rows[metadata]).all().all():
            raise EvaluationError(f"{arm}: labels/stays/POD/strata/folds differ")
        if "frame" in table and not table.frame.eq(frame).all():
            raise EvaluationError(f"{arm}: evaluation frame differs")
        if "post_surg" in table and not table.post_surg.eq(rows.pod.ge(0)).all():
            raise EvaluationError(f"{arm}: phase differs")
        probability = table.probability.to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise EvaluationError(f"{arm}: invalid probabilities")
        aligned[arm] = probability
    return rows, aligned


class StayBootstrap:
    """Union-stay stratified draws with a single multiplicity per hospitalization.

    ``positions`` maps *all* rows, including repeated appearances, to that union.
    Do not construct a separate object or draw for each repeat/arm/time stratum.
    """
    def __init__(self, rows):
        _require_columns(rows, ["stay", "case"])
        if rows[["stay", "case"]].isna().any().any() or not rows.case.isin([0, 1]).all():
            raise EvaluationError("Invalid bootstrap source strata")
        if rows.stay.dtype == np.dtype("float32") or rows.groupby("stay").case.nunique().gt(1).any():
            raise EvaluationError("Imprecise stay keys or conflicting source strata")
        self.stays = rows[["stay", "case"]].drop_duplicates("stay").sort_values("stay").reset_index(drop=True)
        self.positions = pd.Index(self.stays.stay).get_indexer(rows.stay)
        self.strata = [np.flatnonzero(self.stays.case.to_numpy() == value) for value in (1, 0)]
        # Empty strata are allowed for descriptive, case-only interpretation.
        # Evaluation records their absence and disables inferential intervals;
        # a case-only set can still contain both positive and negative days.

    def draw(self, rng):
        """One attempted draw, including all histories; never redraw invalid AUCs."""
        selected = [rng.choice(s, size=len(s), replace=True) for s in self.strata if len(s)]
        return np.bincount(np.concatenate(selected), minlength=len(self.stays)) if selected else np.zeros(0, int)

    def row_weights(self, multiplicity):
        multiplicity = np.asarray(multiplicity)
        if multiplicity.shape != (len(self.stays),) or (multiplicity < 0).any() or not np.isfinite(multiplicity).all():
            raise EvaluationError("Invalid union-stay multiplicity")
        return multiplicity[self.positions]


class BinaryMetric:
    """Tie-aware weighted AUC/AP/Brier with score sorting cached for bootstrap."""
    def __init__(self, y, probability, mask=None):
        y, p = np.asarray(y), np.asarray(probability, float)
        if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
            raise EvaluationError("Invalid binary metric shape/labels")
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise EvaluationError("Invalid metric probabilities")
        mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != y.shape:
            raise EvaluationError("Metric mask shape differs")
        self.size = len(y)
        self.positions = np.flatnonzero(mask)
        self.positions = self.positions[np.argsort(p[self.positions], kind="stable")]
        self.y, self.p = y[self.positions], p[self.positions]
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p)) + 1] if len(self.p) else np.array([], int)

    def score(self, weight=None):
        """Return internal float NaNs for undefined statistics; public APIs use null."""
        if weight is None:
            w = np.ones(len(self.positions))
        else:
            weight = np.asarray(weight, float)
            if weight.shape != (self.size,) or not np.isfinite(weight).all() or (weight < 0).any():
                raise EvaluationError("Invalid metric weights")
            w = weight[self.positions]
        if not len(w) or not w.sum():
            return dict(auc=np.nan, ap=np.nan, brier=np.nan)
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1 - self.y), self.starts)
        positive, negative = pos.sum(), neg.sum()
        auc = float(np.dot(pos, np.cumsum(neg) - .5 * neg) / (positive * negative)) if positive and negative else np.nan
        # AP has no positive-class estimand if no positive weight is present.
        cp, total = np.cumsum(pos[::-1]), np.cumsum((pos + neg)[::-1])
        precision = np.divide(cp, total, out=np.zeros_like(cp), where=total > 0)
        ap = float(np.dot(pos[::-1], precision) / positive) if positive else np.nan
        return dict(auc=auc, ap=ap, brier=float(np.dot(w, (self.p - self.y) ** 2) / w.sum()))


def _nullable(value):
    return float(value) if np.isfinite(value) else None


def stage_counts(rows):
    """Unweighted denominators; negative/case stay counts are explicitly distinct."""
    result = {}
    for name, mask in (("total", np.ones(len(rows), bool)), ("preop", rows.pod.lt(0)), ("post", rows.pod.ge(0))):
        part = rows.loc[mask]
        result[name] = dict(rows=len(part), positive_rows=int(part.y.sum()), negative_rows=int(part.y.eq(0).sum()),
                            unique_stays=int(part.stay.nunique()), case_stays=int(part.loc[part.case.eq(1), "stay"].nunique()),
                            control_stays=int(part.loc[part.case.eq(0), "stay"].nunique()),
                            positive_stays=int(part.loc[part.y.eq(1), "stay"].nunique()),
                            negative_stays=int(part.loc[part.y.eq(0), "stay"].nunique()))
    n_negative = result["total"]["negative_rows"]
    result["preop_negative_fraction"] = result["preop"]["negative_rows"] / n_negative if n_negative else None
    return result


def _prepared_metrics(rows, probability):
    y, post = rows.y.to_numpy(), rows.pod.ge(0).to_numpy()
    return {"total": BinaryMetric(y, probability), "post": BinaryMetric(y, probability, post),
            "preop": BinaryMetric(y, probability, ~post),
            "cross": BinaryMetric(y, probability, (y == 1) | ~post)}


def _score_prepared(prepared, weight=None):
    value = {name: metric.score(weight) for name, metric in prepared.items()}
    return {"AUROC_total": value["total"]["auc"], "AUROC_post": value["post"]["auc"],
            CROSS: value["cross"]["auc"], "AUROC_preop_within_phase": value["preop"]["auc"],
            "AP_total": value["total"]["ap"], "AP_post": value["post"]["ap"],
            "Brier_total_sampled": value["total"]["brier"], "Brier_post_sampled": value["post"]["brier"]}


@threadpool_limits.wrap(limits=2)
def score_rows(rows, probability, *, frame):
    """Stage metrics and exact AUC decomposition, with null for empty classes.

    PI uses the two-negative-stage identity. Task A can have preop positives;
    its valid identity has four positive-stage x negative-stage components.
    A zero-mass component contributes zero even when its AUC is undefined.
    """
    if frame not in SPEC["frames"]:
        raise EvaluationError("Unknown evaluation frame")
    p = np.asarray(probability, float)
    values = _score_prepared(_prepared_metrics(rows, p))
    y, post = rows.y.to_numpy(), rows.pod.ge(0).to_numpy()
    components = []
    np_, nn = int((y == 1).sum()), int((y == 0).sum())
    if frame == "PI72-CLEAN":
        # Reuse the reviewed definition for its nonempty identity; patch only
        # the explicit zero-mass/undefined-class case at this public boundary.
        legacy = pi_decomposition(y, p, post)
        if np.isfinite(legacy["identity_absolute_error"]) and legacy["identity_absolute_error"] > 1e-9:
            raise EvaluationError("PI decomposition identity failed")
        for negative_stage, is_post in (("preop", False), ("post", True)):
            mask = (y == 1) | ((y == 0) & (post == is_post))
            mass = int(((y == 0) & (post == is_post)).sum()) / nn if nn else np.nan
            auc = BinaryMetric(y, p, mask).score()["auc"]
            term = 0. if mass == 0 and np_ else mass * auc
            components.append(dict(positive_stage="post", negative_stage=negative_stage,
                                   weight=_nullable(mass), auc=_nullable(auc), contribution=_nullable(term)))
    else:
        for ps, pv in (("preop", False), ("post", True)):
            for ns, nv in (("preop", False), ("post", True)):
                pm, nm = (y == 1) & (post == pv), (y == 0) & (post == nv)
                mass = pm.sum() * nm.sum() / (np_ * nn) if np_ and nn else np.nan
                auc = BinaryMetric(y, p, pm | nm).score()["auc"]
                term = 0. if mass == 0 else mass * auc
                components.append(dict(positive_stage=ps, negative_stage=ns, weight=_nullable(mass),
                                       auc=_nullable(auc), contribution=_nullable(term)))
    reconstructed = sum(v["contribution"] for v in components) if all(v["contribution"] is not None for v in components) else None
    error = abs(values["AUROC_total"] - reconstructed) if reconstructed is not None else np.nan
    if np.isfinite(error) and error > 1e-9:
        raise EvaluationError("Stage-decomposition identity failed")
    return {"metrics": {k: _nullable(v) for k, v in values.items()}, "counts": stage_counts(rows),
            "decomposition": {"components": components, "reconstructed_total_auc": reconstructed,
                              "identity_absolute_error": _nullable(error)},
            "Brier_population": "sampled_retained_days; not deployment calibration"}


def bootstrap_intervals(estimate, samples, family_indices):
    """2,000 attempted draws; arbitrary declared max-absolute-error family.

    Ordinary percentiles are descriptive. Simultaneous intervals use only
    jointly finite draws and never infer the family from successful contrasts.
    """
    estimate, samples = np.asarray(estimate, float), np.asarray(samples, float)
    if estimate.ndim != 1 or samples.shape != (BOOTSTRAPS, len(estimate)):
        raise EvaluationError("Inference requires exactly 2,000 attempted draws")
    family = list(family_indices)
    if len(set(family)) != len(family) or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(estimate) for i in family):
        raise EvaluationError("Family indices must be unique valid contrast indices")
    valid = np.isfinite(samples)
    ordinary = []
    for j, point in enumerate(estimate):
        n = int(valid[:, j].sum())
        ci = np.quantile(samples[valid[:, j], j], [.025, .975], method="linear").tolist() if n >= MIN_VALID and np.isfinite(point) else None
        ordinary.append(dict(valid_draws=n, undefined_draws=BOOTSTRAPS - n, percentile_ci95=ci))
    common = valid[:, family].all(axis=1) if family and np.isfinite(estimate[family]).all() else np.zeros(BOOTSTRAPS, bool)
    simultaneous = None
    if common.sum() >= MIN_VALID:
        q = float(np.quantile(np.abs(samples[common][:, family] - estimate[family]).max(axis=1), .95, method="linear"))
        simultaneous = {"q": q, "ci95": [[float(estimate[j] - q), float(estimate[j] + q)] for j in family]}
    return dict(ordinary=ordinary, simultaneous=simultaneous, common_valid_draws=int(common.sum()),
                common_undefined_draws=BOOTSTRAPS - int(common.sum()), family_size=len(family))


def frozen_family(frame, *, matched_tree_reference=MATCHED_W1):
    """Return the declared PI ten-contrast or Task A four-contrast family."""
    if frame not in SPEC["frames"]:
        raise EvaluationError("Unknown evaluation frame")
    key = "PI_simultaneous_family" if frame == "PI72-CLEAN" else "A_simultaneous_family"
    family = [dict(item) for item in SPEC["metrics"][key]]
    if frame == "PI72-CLEAN":
        family[4]["reference"] = matched_tree_reference
    return family


def _contrast_key(item):
    return tuple(item[key] for key in ("arm", "reference", "metric"))


def promotion_rule(delta, simultaneous_ci95, *, frame, repeats, stage, family_complete, common_valid_draws, mdd=.01):
    """Frozen gate; screens, missing repeats, invalid families never promote."""
    if frame not in SPEC["frames"] or stage not in ("screen", "confirm"):
        raise EvaluationError("Unknown frame/stage")
    if not np.isfinite(mdd) or mdd < SPEC["promotion_rule"]["MDD"]:
        raise EvaluationError("MDD cannot be lowered after evaluation")
    threshold = max(.01, float(mdd))
    reasons = []
    if stage != "confirm":
        reasons.append("screen_never_promotes")
    if sorted(repeats) != SPEC["repeats"][frame]["confirm"]:
        reasons.append("five_complete_repeats_required")
    if not family_complete or common_valid_draws < MIN_VALID:
        reasons.append("complete_frozen_family_and_1990_common_draws_required")
    if delta is None or not np.isfinite(delta) or delta < threshold:
        reasons.append("delta_below_effective_threshold_or_undefined")
    if simultaneous_ci95 is None or len(simultaneous_ci95) != 2 or not np.isfinite(simultaneous_ci95).all() or simultaneous_ci95[0] <= 0:
        reasons.append("positive_simultaneous_lower_bound_required")
    return dict(passes=not reasons, effective_threshold=threshold, reasons=reasons)


@threadpool_limits.wrap(limits=2)
def evaluate(targets, predictions, *, frame, stage, contrasts=None, family=None, matched_tree_reference=MATCHED_W1):
    """Paired evaluation with equal-repeat estimates and shared stay bootstrap.

    ``contrasts``/``family`` are lists of {arm, reference, metric}; defaults are
    the frozen family. Additional descriptive contrasts may be supplied, but
    promotion requires the *entire* frozen family, not a successful subset.
    """
    if stage not in ("screen", "confirm"):
        raise EvaluationError("stage must be screen or confirm")
    rows, aligned = align_predictions(targets, predictions, frame=frame)
    expected = frozen_family(frame, matched_tree_reference=matched_tree_reference)
    contrasts = expected if contrasts is None else list(contrasts)
    family = expected if family is None else list(family)
    keys, family_keys = [_contrast_key(c) for c in contrasts], [_contrast_key(c) for c in family]
    if not keys or len(set(keys)) != len(keys) or len(set(family_keys)) != len(family_keys):
        raise EvaluationError("Contrasts and family must be explicit and unique")
    if not set(family_keys) <= set(keys):
        raise EvaluationError("Declared family members are missing; cannot shrink family")
    for arm, ref, metric in keys:
        if arm not in aligned or ref not in aligned or metric not in METRICS or arm == ref:
            raise EvaluationError("Unknown/missing arm, reference or metric, or self-contrast")
    repeats, arms = sorted(rows.repeat.unique().tolist()), list(aligned)
    positions = [np.flatnonzero(rows.repeat.to_numpy() == r) for r in repeats]
    prepared = [[_prepared_metrics(rows.iloc[pos], aligned[a][pos]) for a in arms] for pos in positions]
    arm_index = {a: j for j, a in enumerate(arms)}
    metric_index = {m: j for j, m in enumerate(METRICS)}

    def compute(weight=None):
        # np.mean intentionally propagates any undefined repeat, never nanmean.
        per_repeat = np.array([[[scored[m] for m in METRICS]
                                for item in per_arm
                                for scored in [_score_prepared(item, None if weight is None else weight[pos])]]
                               for pos, per_arm in zip(positions, prepared)])
        mean = per_repeat.mean(axis=0)
        differences = np.array([mean[arm_index[a], metric_index[m]] - mean[arm_index[b], metric_index[m]] for a, b, m in keys])
        return differences, mean

    estimate, arm_estimate = compute()
    plan = StayBootstrap(rows)
    source_strata_complete = all(len(s) for s in plan.strata)
    rng = np.random.default_rng(seed(frame, "bootstrap"))
    samples = np.empty((BOOTSTRAPS, len(keys)))
    arm_samples = np.empty((BOOTSTRAPS, len(arms), len(METRICS)))
    for b in range(BOOTSTRAPS):
        weight = plan.row_weights(plan.draw(rng))
        if source_strata_complete:
            samples[b], arm_samples[b] = compute(weight)
        else:
            samples[b], arm_samples[b] = np.nan, np.nan
    indices = [keys.index(k) for k in family_keys]
    intervals = bootstrap_intervals(estimate, samples, indices)
    arm_intervals = bootstrap_intervals(arm_estimate.ravel(), arm_samples.reshape(BOOTSTRAPS, -1), [])
    family_complete = set(family_keys) == {_contrast_key(c) for c in expected}
    comparisons = []
    for j, contrast in enumerate(contrasts):
        ci = None
        if j in indices and intervals["simultaneous"] is not None:
            ci = intervals["simultaneous"]["ci95"][indices.index(j)]
        delta = _nullable(estimate[j])
        comparisons.append({**contrast, "delta": delta, **intervals["ordinary"][j], "simultaneous_ci95": ci,
                            "promotion": promotion_rule(delta, ci, frame=frame, repeats=repeats, stage=stage,
                                                        family_complete=family_complete and keys[j] in family_keys,
                                                        common_valid_draws=intervals["common_valid_draws"])})
    by_key = {_contrast_key(c): c for c in comparisons}
    claims = {}
    for arm in dict.fromkeys(c["arm"] for c in expected):
        relevant = [c for c in expected if c["arm"] == arm]
        passed = {c.get("claim", "local"): by_key.get(_contrast_key(c), {}).get("promotion", {}).get("passes", False) for c in relevant}
        claims[arm] = {"local_improvement": passed.get("local", False)}
        if frame == "PI72-CLEAN":
            claims[arm]["PI_champion_candidate"] = passed.get("local", False) and passed.get("champion_competition", False)
    return dict(frame=frame, stage=stage, primary_metric=SPEC["frames"][frame]["primary_metric"], repeats=repeats,
                complete_repeats=repeats == SPEC["repeats"][frame][stage],
                per_repeat={str(r): {a: score_rows(rows.iloc[pos], aligned[a][pos], frame=frame) for a in arms}
                            for r, pos in zip(repeats, positions)},
                arm_intervals={a: {m: {"estimate": _nullable(arm_estimate[i, j]),
                                       **arm_intervals["ordinary"][i * len(METRICS) + j]}
                                   for j, m in enumerate(METRICS)} for i, a in enumerate(arms)},
                comparisons=comparisons, claims=claims, simultaneous_family=family,
                family_complete=family_complete, common_valid_draws=intervals["common_valid_draws"],
                attempted_draws=BOOTSTRAPS, seed=seed(frame, "bootstrap"),
                bootstrap_universe_stays=len(plan.stays), bootstrap_case_stays=len(plan.strata[0]),
                bootstrap_control_stays=len(plan.strata[1]), quantile_method="linear",
                bootstrap_source_strata_complete=source_strata_complete,
                conditional_on_fitted_models=True, training_selection_uncertainty_included=False,
                independent_confirmation=False)


def account_ledger(declarations, fits, *, required_fit_keys=()):
    """Count declared slots separately from every attempted optimization fit.

    Declarations: id/frame/kind (configuration or procedure), optional stage.
    Fits: attempt_id/fit_key/arm/kind/status. fit_key is the caller's canonical
    fit-pool + data/gate + repeat/fold/trial/init identity; retries need new
    attempt_id and are charged, while two successful executions of one key
    are rejected. Alias views never have their own fitting executions.
    """
    declarations, fits, required_fit_keys = list(declarations), list(fits), tuple(required_fit_keys)
    ids = [d["id"] for d in declarations]
    if len(set(ids)) != len(ids):
        raise EvaluationError("Duplicate fork declaration")
    aliases = {v["view"]: v["canonical_arm"] for v in SPEC["windows"]["nominal_views"] if v["view"] != v["canonical_arm"]}
    count, recipes = Counter(), set()
    pi_configurations = {a["id"] for a in SPEC["arms"]}
    a_fixed = {a["id"] for a in SPEC["Task_A_arms"]}
    for d in declarations:
        if d["frame"] not in SPEC["frames"] or d["kind"] not in ("configuration", "procedure") or d["id"] in aliases:
            raise EvaluationError("Unknown fork type/frame or alias charged as a new fork")
        if d["kind"] == "procedure":
            programs = SPEC["selection_programs"] if d["frame"] == "PI72-CLEAN" else SPEC["Task_A_programs"]
            if d["id"] not in programs:
                raise EvaluationError("Unregistered selection procedure")
        elif d["frame"] == "PI72-CLEAN":
            if d["id"] not in pi_configurations:
                raise EvaluationError("Unregistered PI configuration")
        elif d["id"] not in a_fixed:
            # The spec gives Task A's nested recipes/profile slots, not literal
            # IDs. Caller names are allowed only with the exact declared recipe.
            recipe = (d.get("features"), d.get("learner"), d.get("window"), d.get("representation"), d.get("profile_slot"))
            valid_tree = recipe[0] == "GAIN50" and recipe[1] == "LGB" and recipe[2] in (3, 5, 7, 14) and recipe[3] in ("R1", "R3") and recipe[4] is None
            valid_network = recipe[0] == "GAIN50" and recipe[1] in ("TCN", "GRU") and recipe[2] is None and recipe[3] is None and recipe[4] in (1, 2)
            if not (valid_tree or valid_network) or recipe in recipes:
                raise EvaluationError("Unknown or duplicated Task A nested recipe/profile slot")
            recipes.add(recipe)
        count[d["frame"], d["kind"]] += 1
    caps = {("PI72-CLEAN", "configuration"): 85, ("PI72-CLEAN", "procedure"): 5,
            ("A-formal", "configuration"): 19, ("A-formal", "procedure"): 4}
    if any(count[k] > limit for k, limit in caps.items()):
        raise EvaluationError("Frozen configuration/procedure slot budget exceeded")
    attempts, successes, attempted_keys, status, kinds = set(), set(), set(), Counter(), Counter()
    for fit in fits:
        if fit["attempt_id"] in attempts or fit["arm"] not in ids or fit["arm"] in aliases:
            raise EvaluationError("Duplicate fit attempt or undeclared/alias arm")
        if not fit["fit_key"] or fit["status"] not in ("completed", "failed", "incomplete", "running"):
            raise EvaluationError("Invalid fit execution record")
        if fit["kind"] not in ("inner", "outer", "selector", "epoch", "final"):
            raise EvaluationError("Unaccounted fit kind")
        attempts.add(fit["attempt_id"])
        attempted_keys.add(fit["fit_key"])
        status[fit["status"]] += 1
        kinds[fit["kind"]] += 1
        if fit["status"] == "completed":
            if fit["fit_key"] in successes:
                raise EvaluationError("Canonical fit (including w1 aliases) fitted twice")
            successes.add(fit["fit_key"])
    missing = sorted(set(required_fit_keys) - successes)
    unresolved = sorted(attempted_keys - successes)
    return dict(declared_configuration_slots=sum(count[f, "configuration"] for f in SPEC["frames"]),
                declared_procedures=sum(count[f, "procedure"] for f in SPEC["frames"]),
                declared_by_frame={f: {k: count[f, k] for k in ("configuration", "procedure")} for f in SPEC["frames"]},
                declared_by_stage=dict(Counter(d.get("stage", "unspecified") for d in declarations)),
                all_frozen_slots_declared=all(count[k] == limit for k, limit in caps.items()),
                attempted_fits=len(fits), fits_by_status=dict(status), fits_by_kind=dict(kinds),
                successful_canonical_fits=len(successes), missing_required_fit_keys=missing,
                unresolved_fit_keys=unresolved,
                complete=not missing and not unresolved and not status["running"] and not status["incomplete"],
                completeness_scope="submitted executions and supplied required_fit_keys only",
                required_fit_manifest_supplied=bool(required_fit_keys),
                global_historical_fork_budget_verified=False)


@threadpool_limits.wrap(limits=2)
def time_stratified_metrics(targets, predictions, *, frame):
    """POD-bin discrimination with >=10 positive and >=10 negative stays.

    Counts are always reported. Gate AUC on the original support, then use
    2,000 shared union-stay draws for descriptive pointwise intervals. No
    case-only lead AUC, threshold search, or promotion is performed here.
    """
    rows, aligned = align_predictions(targets, predictions, frame=frame)
    repeats, plan = sorted(rows.repeat.unique()), StayBootstrap(rows)
    masks = {"preop": rows.pod.lt(0).to_numpy()}
    masks.update({f"POD{lo}-{hi}": rows.pod.between(lo, hi).to_numpy() for lo, hi in SPEC["metrics"]["POD_strata"]})
    result = {}
    for name, mask in masks.items():
        positions = [np.flatnonzero(mask & rows.repeat.eq(r).to_numpy()) for r in repeats]
        counts = [stage_counts(rows.iloc[ix])["total"] for ix in positions]
        auc_supported = [c["positive_stays"] >= 10 and c["negative_stays"] >= 10 for c in counts]
        prepared = {a: [BinaryMetric(rows.y.to_numpy()[ix], p[ix]) for ix in positions] for a, p in aligned.items()}
        arms, metric_names = list(aligned), ("auc", "ap", "brier")

        def scores(multiplicity=None):
            w = None if multiplicity is None else plan.row_weights(multiplicity)
            out = []
            for arm in arms:
                per_repeat = []
                for i, (ix, metric) in enumerate(zip(positions, prepared[arm])):
                    scored = metric.score(None if w is None else w[ix])
                    if not auc_supported[i]:
                        scored["auc"] = np.nan
                    per_repeat.append([scored[k] for k in metric_names])
                out.append(per_repeat)
            return np.array(out)

        point = scores()
        samples = np.full((BOOTSTRAPS, len(arms), 3), np.nan)
        if mask.any():
            rng = np.random.default_rng(seed(frame, "bootstrap"))
            for b in range(BOOTSTRAPS):
                samples[b] = scores(plan.draw(rng)).mean(axis=1)
        mean = point.mean(axis=1)
        intervals = bootstrap_intervals(mean.ravel(), samples.reshape(BOOTSTRAPS, -1), [])
        result[name] = dict(counts_by_repeat={str(r): c for r, c in zip(repeats, counts)},
                            auc_supported_by_repeat={str(r): ok for r, ok in zip(repeats, auc_supported)},
                            arms={a: {m: {"estimate": _nullable(mean[i, j]), **intervals["ordinary"][i * 3 + j],
                                           "per_repeat": [_nullable(v) for v in point[i, :, j]]}
                                      for j, m in enumerate(metric_names)} for i, a in enumerate(arms)})
    return dict(frame=frame, strata=result, intervals="descriptive pointwise; not promotion", attempted_draws=BOOTSTRAPS)
