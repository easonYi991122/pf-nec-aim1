"""Fixed-capacity trees and bounded float32 calendar-window inputs."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import contract as c, windows


def lgb_params(seed):
    spec = c.SPEC["learners"]["LGB"]
    names = ("objective", "learning_rate", "num_leaves", "min_child_samples",
             "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2",
             "max_bin", "num_threads", "deterministic", "force_col_wise")
    return {**{name: spec[name] for name in names}, "seed": seed,
            "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed,
            "verbosity": -1, "metric": "None"}






def input_columns(k, window, representation):
    if window not in c.WINDOW_LENGTHS or representation not in ("R1", "R3"):
        raise c.ContractError("Unregistered tree input")
    count = k if window == 1 else k * window + window + 1 if representation == "R1" else 7 * k + 1
    if count > c.SPEC["windows"]["actual_engineered_column_cap"]:
        raise c.ContractError("Engineering cap exceeded: stop, never reduce k or w")
    return count


def matrix(rows, *, window, representation, batch_size=1024, guard=None):
    """One float32 arm matrix, constructed in bounded windows; never all arms."""
    c.assert_predictors(rows.values.columns, bank=rows.bank, dependencies=rows.dependencies)
    count = input_columns(len(rows.values.columns), window, representation)
    if guard:
        guard(allocation=len(rows.keys) * count * 4)
    output = np.empty((len(rows.keys), count), dtype=np.float32)
    layout, offset = None, 0
    for batch in windows.iter_windows(rows, window=window, batch_size=batch_size):
        if guard:
            guard()
        block = windows.tree_matrix(batch, representation)
        output[offset:offset + len(block.keys)] = block.values
        offset += len(block.keys)
        if layout is None:
            layout = (block.columns, block.concepts, block.observation_columns)
    if layout is None or offset != len(rows.keys):
        raise c.ContractError("Empty/incomplete tree matrix")
    return windows.TreeMatrix(rows.keys.copy(), output, *layout)


def _labels(y, n):
    y = np.asarray(y)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise c.ContractError("Tree fit/evaluation needs aligned binary labels and both classes")
    return y


def validate_matrix(matrix, arm, context, learner):
    """Fail closed on forged final layouts as well as on source dependencies."""
    c.validate_keys(matrix.keys, frame=context.frame)
    entries = c.SPEC["arms"] if context.frame == "PI72-CLEAN" else c.SPEC["Task_A_arms"]
    entry = next((entry for entry in entries if entry["id"] == arm), None)
    if entry is None and context.frame == "A-formal" and arm.startswith("A-T4-GAIN50-w"):
        pieces = arm.split("-")
        entry = {"features": "GAIN50", "window": int(pieces[3][1:]), "representation": pieces[4]}
    if entry is None or learner != "LGB" or entry.get("learner", "LGB") != learner:
        raise c.ContractError("Tree layout belongs to an undeclared arm")
    recipe, window = entry["features"], entry.get("window", 1)
    representation = "R1" if entry.get("representation") in (None, "single_day") else entry["representation"]
    source = c.SPEC["feature_sets"][recipe]
    concepts = tuple(dict.fromkeys(concept for concept in matrix.concepts if concept is not None))
    if "selector" in source:
        c.feature_columns(recipe, selected=concepts, frame=context.frame)
    elif concepts != tuple(source["columns"]):
        raise c.ContractError("Fixed recipe concept list/order changed")
    expected = input_columns(source["nominal_k"], window, representation)
    if matrix.values.shape != (len(matrix.keys), expected) or len(matrix.columns) != len(matrix.concepts):
        raise c.ContractError("Final tree layout dimensions differ from frozen arm")
    if window == 1:
        names = concepts
    elif representation == "R1":
        names = tuple(f"{name}__lag{lag}" for lag in range(window) for name in concepts)
        names += tuple(f"row_observed__lag{lag}" for lag in range(window)) + ("window_observed_day_count",)
    else:
        names = tuple(f"{name}__{stat}" for name in concepts for stat in c.SPEC["windows"]["R3"]["per_feature"])
        names += ("window_observed_day_count",)
    if tuple(matrix.columns) != names:
        raise c.ContractError("Unlisted/forbidden derived predictor in final tree layout")


def primary_auc(y, probability, post, frame):
    mask = np.asarray(post, bool) if frame == "PI72-CLEAN" else np.ones(len(y), bool)
    y = np.asarray(y)
    if mask.shape != y.shape or len(np.unique(y[mask])) != 2:
        raise c.ContractError("Single-class primary validation phase; no redraw")
    return float(roc_auc_score(y[mask], np.asarray(probability)[mask]))


@dataclass
class TreeFit:
    model: object
    learner: str
    rounds: int
    columns: tuple
    parameters: dict

    def predict(self, X):
        if X.shape[1] != len(self.columns):
            raise c.ContractError("Apply columns differ from fitted tree")
        if self.learner == "LGB":
            probability = self.model.predict(X, num_iteration=self.rounds, num_threads=2)
        else:
            raise c.ContractError("Only the exported LightGBM learner is available")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise c.ContractError("Nonfinite/out-of-range tree probabilities")
        return np.asarray(probability, dtype=np.float64)

    def save(self, path):
        if self.learner == "LGB":
            self.model.save_model(str(path), num_iteration=self.rounds)
        else:
            self.model.save_model(str(path))


def load_fit(path, info):
    c.reset_threads()
    if info["learner"] == "LGB":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
    else:
        raise c.ContractError("Only the exported LightGBM learner is available")
    return TreeFit(model, info["learner"], info["rounds"], tuple(info["columns"]), info["parameters"])


def fit_tree(train, y, *, learner, context, arm, rounds=None,
             valid=None, valid_y=None, valid_post=None, guard=None):
    """No outer-test argument exists. LGB stopping is inner-validation-only."""
    c.reset_threads()  # torch before learner imports, including after CORE
    validate_matrix(train, arm, context, learner)
    if valid is not None:
        validate_matrix(valid, arm, context, learner)
        if context.inner not in (0, 1, 2) or context.pool != "inner-fit":
            raise c.ContractError("Early-stopping/validation input requires an explicit inner fit context")
        if train.keys[c.H].isin(valid.keys[c.H]).any():
            raise c.ContractError("Fit/validation hospitalization overlap")
    y = _labels(y, len(train.keys))
    if np.isinf(train.values).any() or train.values.dtype != np.float32:
        raise c.ContractError("Tree input must be float32, NaNs allowed, infinities forbidden")
    input_columns(len(train.columns), 1, "R1")
    if guard:
        guard()
    seed = context.seed("model", arm)
    if learner == "LGB":
        import lightgbm as lgb
        params = lgb_params(seed)
        data = lgb.Dataset(train.values, label=y, feature_name=list(train.columns), free_raw_data=True)
        kwargs = {}
        if valid is not None:
            if rounds is not None or tuple(valid.columns) != tuple(train.columns):
                raise c.ContractError("Inner stopping cannot use fixed rounds or mismatched columns")
            valid_y = _labels(valid_y, len(valid.keys))
            primary_auc(valid_y, np.full(len(valid_y), .5), valid_post, context.frame)
            validation = lgb.Dataset(valid.values, label=valid_y, reference=data, free_raw_data=True)

            def auc(probability, dataset):
                return "primary_auc", primary_auc(dataset.get_label(), probability, valid_post, context.frame), True

            kwargs = {"valid_sets": [validation], "valid_names": ["inner-validation"], "feval": auc,
                      "callbacks": [lgb.early_stopping(c.SPEC["learners"]["LGB"]["patience"], verbose=False)]}
            count = c.SPEC["learners"]["LGB"]["max_rounds"]
        else:
            if not isinstance(rounds, int) or not 1 <= rounds <= c.SPEC["learners"]["LGB"]["max_rounds"]:
                raise c.ContractError("Outer LGB requires frozen median inner stopping rounds")
            count = rounds
        model = lgb.train(params, data, num_boost_round=count, **kwargs)
        # Requested rounds remain the stopping budget even if no legal split is
        # possible; LightGBM may physically emit fewer trees on a tiny fixture.
        count = int(model.best_iteration) if valid is not None else rounds
        if count < 1:
            raise c.ContractError("LGB returned no valid stopping iteration")
    else:
        raise c.ContractError("Unregistered tree learner")
    if guard:
        guard()
    return TreeFit(model, learner, count, tuple(train.columns), params)


def median_rounds(values):
    if len(values) != 3 or any(not isinstance(v, int) or v < 1 for v in values):
        raise c.ContractError("Exactly three valid inner round counts required")
    return int(np.median(values))
