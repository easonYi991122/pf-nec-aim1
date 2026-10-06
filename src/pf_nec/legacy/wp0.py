"""Pure local clean-table row/split/metric contract; no model access."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
common = SimpleNamespace(H="hospitalizationidnew", ROW="__harness_row_index")


H, ROW, LABEL = common.H, common.ROW, "outcome_3d"


CASE_COUNT, CASE_TRAIN = 356, 285


CONTROL_COUNT, CONTROL_TRAIN = 1039, 831


BUDGET_SECONDS, MAX_UNMATCHED, MAX_POSITIVE_LOSS = 3600, 0.05, 0.02


KEYS = [H, "actual_date"]


CLEAN_COLUMNS = [H, "cardsurgdtshift_index", "day", LABEL]


CACHE_META = KEYS + ["surg", "pod_raw", "post_surg", "group"]


def row_counts(frame, case_stays):
    case = frame[H].isin(case_stays)
    positive, negative, pre = frame[LABEL].eq(1), frame[LABEL].eq(0), frame.day.lt(0)
    return dict(rows=len(frame), stays=frame[H].nunique(), case_stays=frame.loc[case, H].nunique(),
                control_stays=frame.loc[~case, H].nunique(), positive_rows=int(positive.sum()),
                stays_with_positive_rows=frame.loc[positive, H].nunique(),
                negative_rows=int(negative.sum()), unlabeled_rows=int(frame[LABEL].isna().sum()),
                preoperative_rows=int(pre.sum()),
                postoperative_rows=int((~pre).sum()), preoperative_positive_rows=int((pre & positive).sum()),
                preoperative_negative_rows=int((pre & negative).sum()),
                postoperative_negative_rows=int((~pre & negative).sum()),
                case_history_negative_rows=int((case & negative).sum()),
                f_pre=float((pre & negative).sum() / negative.sum()) if negative.any() else None)


def reconcile_rows(clean, cache_meta):
    """Audit keys before reading features; never use clean.actual_date.

    Strata use the untouched clean label BEFORE exclusion. Return matched rows,
    original stay strata, and an aggregate audit.
    """
    frame = clean[CLEAN_COLUMNS].copy()
    values = frame[[H, "cardsurgdtshift_index", "day"]].to_numpy(dtype=float)
    day_values = frame[["cardsurgdtshift_index", "day"]].to_numpy(dtype=float)
    # De-identified stay IDs may have fractional parts; preserve them verbatim.
    if frame.empty or not np.isfinite(values).all() or not np.equal(day_values, np.floor(day_values)).all():
        raise ValueError("Clean keys must be finite; surgery date and day must be integers")
    if not frame[LABEL].dropna().isin([0, 1]).all() or (frame.day.lt(0) & frame[LABEL].eq(1)).any():
        raise ValueError("Observed clean labels must be binary with no preoperative positives")
    if frame.groupby(H).cardsurgdtshift_index.nunique().gt(1).any():
        raise ValueError("Each clean stay must have one index surgery date")
    # Missing row labels stay missing. A stay is a case iff it has an actual positive.
    stay_labels = frame[LABEL].eq(1).groupby(frame[H], sort=True).max().astype(int)
    if frame.groupby(H)[LABEL].sum().gt(1).any():
        raise ValueError("The PI row design requires one positive row per case stay")
    case_stays = stay_labels.index[stay_labels.eq(1)].to_numpy()
    frame["actual_date"] = frame.cardsurgdtshift_index + frame.day
    frame[ROW] = np.arange(len(frame), dtype=np.int64)
    if frame.duplicated(KEYS).any() or cache_meta.duplicated(KEYS).any():
        raise ValueError("Duplicate stay/date keys; no many-to-many joins are permitted")
    if cache_meta[CACHE_META].isna().any().any() or cache_meta.groupby(H).group.nunique().gt(1).any():
        raise ValueError("Invalid cache keys/phase/group mapping")
    groups = cache_meta.drop_duplicates(H).set_index(H).group
    control_groups = stay_labels.index[stay_labels.eq(0)].to_series().map(groups).fillna("not in v2.6")
    composition = control_groups.value_counts().sort_index().to_dict()
    joined = frame.merge(cache_meta[CACHE_META], on=KEYS, how="left", sort=False,
                         validate="one_to_one", indicator=True)
    missing = joined._merge.eq("left_only")
    matched = joined.loc[~missing].drop(columns="_merge").copy()
    disagreements = {
        "pod_raw": int(matched.day.ne(matched.pod_raw).sum()),
        "surgery_date": int(matched.cardsurgdtshift_index.ne(matched.surg).sum()),
        "post_surg": int(matched.day.ge(0).ne(matched.post_surg).sum()),
    }
    missing_stays, retained_stays = set(joined.loc[missing, H]), set(matched[H])
    missing_fraction = float(missing.mean())
    source_counts, matched_counts = row_counts(frame, case_stays), row_counts(matched, case_stays)
    lost_positive = joined.loc[missing & joined[LABEL].eq(1)]
    positive_loss_fraction = len(lost_positive) / source_counts["positive_rows"] if source_counts["positive_rows"] else 0.
    first_nec = (lost_positive[H].map(cache_meta.drop_duplicates(H).set_index(H).nec)
                 if "nec" in cache_meta else pd.Series(np.nan, index=lost_positive.index))
    truncated = lost_positive.actual_date.ge(first_nec) & first_nec.lt(lost_positive.cardsurgdtshift_index)
    absent = ~lost_positive[H].isin(groups.index)
    reasons = {"at_or_after_preoperative_first_NEC": int(truncated.sum()),
               "stay_absent_from_v26": int(absent.sum()),
               "other_or_insufficient_cache_metadata": int((~truncated & ~absent).sum())}
    matched_control_groups = control_groups.loc[control_groups.index.isin(retained_stays)]
    audit = {
        "source": source_counts, "matched": matched_counts,
        "f_pre_comparison": {"full_table": source_counts["f_pre"], "matched_rows": matched_counts["f_pre"],
                             "negative_definition": "outcome_3d == 0; missing labels are excluded from both counts"},
        "positive_rows_lost": {"rows": len(lost_positive), "fraction": positive_loss_fraction,
                               "by_cache_observable_reason": reasons,
                               "controller_attribution": "Amendment 1 将真实数据的 7 个阳性损失归因于术前后均有 NEC、"
                               "而 v2.6 在首次 NEC 截断；缓存缺失住院无法由本脚本独立核实首次 NEC 日期。"},
        "unmatched": {**row_counts(joined.loc[missing], case_stays), "row_fraction": missing_fraction,
                      "fully_unmatched_stays": len(missing_stays - retained_stays),
                      "partially_unmatched_stays": len(missing_stays & retained_stays),
                      "rows_by_v26_group": joined.loc[missing, H].map(groups).fillna("not in v2.6")
                      .value_counts().sort_index().to_dict()},
        "control_pool_composition": composition,
        "matched_control_pool_composition": matched_control_groups.value_counts().sort_index().to_dict(),
        "control_pool_unexpected_groups": {k: v for k, v in composition.items()
                                           if k not in {"none", "c", "d", "not in v2.6"}},
        "join_disagreements": disagreements,
        "gate": {"amendment": 1, "maximum_unmatched_fraction": MAX_UNMATCHED, "denominator": "all_clean_rows",
                 "maximum_positive_loss_fraction": MAX_POSITIVE_LOSS, "positive_denominator": "all_clean_positive_rows",
                 "unmatched_rows_passed": missing_fraction <= MAX_UNMATCHED,
                 "positive_loss_passed": positive_loss_fraction <= MAX_POSITIVE_LOSS,
                 "passed": (missing_fraction <= MAX_UNMATCHED and positive_loss_fraction <= MAX_POSITIVE_LOSS
                            and not any(disagreements.values()))},
    }
    return matched.reset_index(drop=True), stay_labels, audit


def make_split(stay_labels, seed, n_controls=CONTROL_COUNT, case_train=CASE_TRAIN, control_train=CONTROL_TRAIN,
               matched_stays=None):
    """Cases first; sorted control-pool choice; permute the returned draw order.

    The first requested number of each permutation goes to training. No redraws.
    """
    if not stay_labels.index.is_unique or not stay_labels.isin([0, 1]).all():
        raise ValueError("Stay strata must be unique and binary")
    cases = np.sort(stay_labels.index[stay_labels.eq(1)].to_numpy())
    controls = np.sort(stay_labels.index[stay_labels.eq(0)].to_numpy())
    if matched_stays is not None:
        controls = controls[np.isin(controls, matched_stays)]
    if not (0 < case_train < len(cases) and 0 < control_train < n_controls <= len(controls)):
        raise ValueError("Insufficient case/control stays for the specified split")
    cases = np.random.default_rng(seed + 1).permutation(cases)
    sampled = np.random.default_rng(seed).choice(controls, size=n_controls, replace=False)
    controls = np.random.default_rng(seed + 2).permutation(sampled)
    return {"train": np.concatenate([cases[:case_train], controls[:control_train]]),
            "test": np.concatenate([cases[case_train:], controls[control_train:]]),
            "sampled_controls": sampled, "case_stays": cases}


def decomposition(y, probability, post_surg, weight=None):
    """Row AUROCs with half credit for ties (sklearn's weighted rank statistic)."""
    y, probability, post_surg = np.asarray(y), np.asarray(probability), np.asarray(post_surg)
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    if not (y.shape == probability.shape == post_surg.shape == weight.shape and y.ndim == 1):
        raise ValueError("Metric arrays must be aligned one-dimensional vectors")
    if (not np.isin(y, [0, 1]).all() or not np.isin(post_surg, [0, 1]).all()
            or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()
            or not np.isfinite(weight).all() or (weight < 0).any()):
        raise ValueError("Invalid labels, phases, probabilities or weights")
    positive, pre = y == 1, post_surg == 0
    if (positive & pre).any():
        raise ValueError("The specified design has no preoperative positives")

    def auc(mask):
        selected = mask & (weight > 0)
        if len(np.unique(y[selected])) != 2:
            return float("nan")
        return float(roc_auc_score(y[selected], probability[selected], sample_weight=weight[selected]))

    negative_weight = weight[~positive].sum()
    f_pre = float(weight[pre & ~positive].sum() / negative_weight) if negative_weight else float("nan")
    total, before, after = auc(np.ones(len(y), bool)), auc(positive | pre), auc(positive | ~pre)
    pre_component, post_component = f_pre * before, (1 - f_pre) * after
    error = abs(total - (pre_component + post_component))
    if np.isfinite(error) and error > 1e-9:
        raise AssertionError("AUROC decomposition identity failed")
    return dict(AUROC_total=total, AUROC_pre=before, AUROC_post=after, f_pre=f_pre,
                AUROC_postoperative_rows_only=auc(~pre), pre_weighted_component=pre_component,
                post_weighted_component=post_component, preop_inclusion_delta=total - after,
                identity_absolute_error=error)
