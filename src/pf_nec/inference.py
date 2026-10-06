"""Single-arm descriptive intervals using the frozen evaluator's primitives.

No fitting, contrasts, promotion decision, path access or file writes. This
adapter uses the same stay universe, random draws and arm scoring as evaluate.
"""
import numpy as np
from threadpoolctl import threadpool_limits
from . import evaluate as ev


@threadpool_limits.wrap(limits=2)
def single_arm_intervals(targets, predictions, *, frame, model):
    rows, aligned = ev.align_predictions(targets, {model: predictions}, frame=frame)
    repeats = sorted(rows.repeat.unique().tolist())
    positions = [np.flatnonzero(rows.repeat.to_numpy() == r) for r in repeats]
    prepared = [ev._prepared_metrics(rows.iloc[pos], aligned[model][pos]) for pos in positions]

    def compute(weight=None):
        per_repeat = [ev._score_prepared(item, None if weight is None else weight[pos])
                      for pos, item in zip(positions, prepared)]
        # Propagate undefined phases; never silently drop a repeat.
        return np.array([[score[m] for m in ev.METRICS] for score in per_repeat]).mean(axis=0)

    estimate = compute()
    plan = ev.StayBootstrap(rows)
    complete = all(len(s) for s in plan.strata)
    rng = np.random.default_rng(ev.seed(frame, "bootstrap"))
    samples = np.empty((ev.BOOTSTRAPS, len(ev.METRICS)))
    for draw in range(ev.BOOTSTRAPS):
        weight = plan.row_weights(plan.draw(rng))
        samples[draw] = compute(weight) if complete else np.nan
    intervals = ev.bootstrap_intervals(estimate, samples, [])
    return dict(frame=frame, model=model, repeats=repeats,
                arm_intervals={m: dict(estimate=ev._nullable(estimate[j]), **intervals["ordinary"][j])
                               for j, m in enumerate(ev.METRICS)},
                attempted_draws=ev.BOOTSTRAPS, seed=ev.seed(frame, "bootstrap"),
                bootstrap_universe_stays=len(plan.stays), bootstrap_case_stays=len(plan.strata[0]),
                bootstrap_control_stays=len(plan.strata[1]), bootstrap_source_strata_complete=complete,
                quantile_method="linear", conditional_on_fitted_models=True,
                training_selection_uncertainty_included=False,
                independent_confirmation=False, promotion_assessed=False)
