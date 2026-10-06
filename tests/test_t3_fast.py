"""RX-I11 exact JSON/order, native learner and bounded-storage acceptance."""
import json

import numpy as np
import pandas as pd
import pytest

from pf_nec import contract as c
from explore.t3 import interpret as it, t3_fast as fast


def _rows(frame, *, stays=45):
    records = []
    repeats = (1, 2) if frame == "PI72-CLEAN" else (0, 1)
    pods = (-8, -1, 0, 1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 23, 24, 25, 26)
    for repeat in repeats:
        for h in range(stays):
            case = h < 25
            for pod in pods:
                if pod == -8 and h >= 7:
                    continue  # >31 lead is suppressed, not silently discarded
                item = dict(repeat=repeat, row_key=f"{h}:{pod}", stay=h + .125,
                            y=int(case and pod == 24), pod=pod, case=int(case),
                            day=100 + pod, event_day=127. if case else np.nan)
                if frame == "A-formal":
                    item["fold"] = (h + repeat) % 5
                records.append(item)
    return pd.DataFrame(records)


def _layout(representation):
    if representation == "R1":
        return [dict(column="signal__lag0", feature="signal", lag=0, kind="value"),
                dict(column="signal__lag2", feature="signal", lag=2, kind="value"),
                dict(column="signal_missing__lag2", feature="signal", lag=2, kind="missingness"),
                dict(column="noise__lag0", feature="noise", lag=0, kind="value"),
                dict(column="window_flag", feature="__window_structure__", lag=None, kind="structural")]
    return [dict(column=f"signal__{name}", feature="signal", lag=None,
                 kind="observation" if name == "n_observed" else "value")
            for name in ("last", "mean", "min", "max", "sd", "slope", "n_observed")] + [
        dict(column="noise_last", feature="noise", lag=None, kind="value"),
        dict(column="window_flag", feature="__window_structure__", lag=None, kind="structural")]


FEATURES = ["signal", "noise", "unselected", "__window_structure__"]


def _packets(rows, representation="R1", *, values=True, chunk=97):
    layout = _layout(representation)
    rng = np.random.default_rng(793)
    records = []
    fields = ["repeat", "fold"] if "fold" in rows else ["repeat"]
    for context, part in rows.groupby(fields, sort=False):
        for start in range(0, len(part), chunk):
            block = part.iloc[start:start + chunk]
            phi = rng.normal(size=(len(block), len(layout)))
            phi[:, 0] *= 1e5  # expose reassociation; cancellations across columns
            phi[:, 1] = -phi[:, 0] + phi[:, 1] * 1e-5
            record = dict(rows=block, phi=phi, base=np.zeros(len(block)),
                          margin=phi.sum(axis=1), layout=layout, model_id=f"toy-{context}")
            if values:
                x = rng.normal(size=phi.shape)
                x[::3, :2] = np.nan
                record["values"] = pd.DataFrame(x, columns=[item["column"] for item in layout])
            records.append(record)
    return records


def _legacy(rows, packets, frame, features=FEATURES):
    views = ("overall", "pod", "lead") if frame == "PI72-CLEAN" else ("pod", "lead")
    reports = {}
    for view in views:
        strata = {}
        for name in it.time_masks(rows, view):
            result = it.shap_importance(rows, iter(packets), frame=frame, feature_universe=features,
                                        view=view, strata=[name])
            strata.update(result.pop("strata"))
        reports[view] = {**result, "strata": strata}
    paired = it.paired_pod_changes(rows, iter(packets), frame=frame, feature_universe=features)
    return reports, paired


def _new(rows, packets, frame, directory, **kwargs):
    calls = []
    def factory():
        calls.append(1)
        return iter(packets)
    result = fast.shap_views(rows, factory, frame=frame, feature_universe=kwargs.pop("features", FEATURES),
                            views=("overall", "pod", "lead") if frame == "PI72-CLEAN" else ("pod", "lead"),
                            paired=True, scratch_dir=directory, **kwargs)
    assert list(directory.iterdir()) == []  # private arrays never survive success
    return result, len(calls)


def _exact(left, right):
    # hook._write also converts NumPy scalar counts to their native JSON type.
    native = lambda value: value.item()
    assert json.dumps(left, sort_keys=False, allow_nan=False, default=native) == json.dumps(right, sort_keys=False, allow_nan=False, default=native)


@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(it, "BOOTSTRAPS", 11)


@pytest.fixture(autouse=True)
def private_runtime(tmp_path, monkeypatch):
    for name, leaf in (("MPLCONFIGDIR", "mpl"), ("TMPDIR", "tmp")):
        directory = tmp_path / leaf
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))


@pytest.mark.parametrize("frame", ["PI72-CLEAN", "A-formal"])
@pytest.mark.parametrize("representation", ["R1", "R3"])
@pytest.mark.parametrize("learner", ["LGB"])
def test_native_learners_layouts_exact_single_pass(tmp_path, few_draws, frame, representation, learner):
    c.reset_threads()  # torch must precede either native learner
    rows, layout = _rows(frame), _layout(representation)
    rng = np.random.default_rng(27)
    train = rng.normal(size=(180, len(layout))).astype(np.float32)
    y = (train[:, 0] + train[:, 1] > 0).astype(int)
    if learner == "LGB":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=5, num_leaves=4, min_child_samples=3, n_jobs=2, verbosity=-1)
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, n_jobs=2, tree_method="exact")
    model.fit(train, y)
    packets = []
    fields = ["repeat", "fold"] if "fold" in rows else ["repeat"]
    for context, part in rows.groupby(fields, sort=False):
        x = rng.normal(size=(len(part), len(layout))).astype(np.float32)
        x[::3, :2] = np.nan
        for block in it.native_shap_chunks(model, x, learner=learner, chunk_rows=97):
            sl = slice(block["start"], block["stop"])
            packets.append({**block, "rows": part.iloc[sl], "values": x[sl],
                            "layout": layout, "model_id": f"native-{context}"})
    expected = _legacy(rows, packets, frame)
    actual, streams = _new(rows, packets, frame, tmp_path / "native")
    _exact(expected, actual)
    assert streams == 1
    assert not actual[0]["lead"]["strata"][">31"]["rank_ci_support_ok"]
    assert actual[0]["lead"]["strata"]["1"]["rank_ci_support_ok"]
    feature = actual[0]["pod"]["strata"]["POD0-2"]["features"][2]
    assert feature["mean_abs"] == 0 and feature["included_in_any_model"] is False


@pytest.mark.parametrize("frame", ["PI72-CLEAN", "A-formal"])
def test_full_frozen_bootstrap_serial_and_processes_exact(tmp_path, frame):
    rows = _rows(frame)
    packets = _packets(rows)
    expected = _legacy(rows, packets, frame)
    serial, streams = _new(rows, packets, frame, tmp_path / "serial")
    parallel, parallel_streams = _new(rows, packets, frame, tmp_path / "parallel", workers=2)
    _exact(expected, serial)
    _exact(serial, parallel)
    assert streams == parallel_streams == 1
    assert serial[0]["lead"]["strata"]["1"]["features"][0]["ci"]["mean_abs"]["valid_draws"] == 2000


@pytest.mark.parametrize("values", [False, True])
def test_reappearing_repeats_missing_values_and_grouped_fallback(tmp_path, few_draws, values):
    rows = _rows("PI72-CLEAN", stays=30)
    packets = _packets(rows, values=values)
    # Alternate repeats to force exact spill -> reload -> update cycles.
    packets = [packet for pair in zip(packets[:len(packets) // 2], packets[len(packets) // 2:]) for packet in pair]
    expected = _legacy(rows, packets, "PI72-CLEAN")
    actual, streams = _new(rows, packets, "PI72-CLEAN", tmp_path / "spills")
    _exact(expected, actual)
    assert streams == 1
    grouped, streams = _new(rows, packets, "PI72-CLEAN", tmp_path / "groups", cell_limit=10000)
    _exact(expected, grouped)
    assert 1 < streams < 15


def test_one_feature_empty_strata_variable_selections_and_packet_order(tmp_path, few_draws):
    rows = _rows("A-formal", stays=15)
    rows = rows.loc[rows.pod.isin([0, 1, 3, 4])].reset_index(drop=True)
    packets = _packets(rows, chunk=3)
    for i, packet in enumerate(packets):
        width = 1 if packet["rows"].fold.iloc[0] % 2 else 2
        packet["phi"] = packet["phi"][:, :width]
        packet["layout"] = packet["layout"][:width]
        packet["values"] = packet["values"].iloc[:, :width]
        packet["margin"] = packet["phi"].sum(axis=1)
    expected = _legacy(rows, packets, "A-formal", features=["signal"])
    actual, _ = _new(rows, packets, "A-formal", tmp_path / "one", features=["signal"])
    _exact(expected, actual)
    assert actual[0]["pod"]["strata"]["preop"]["explanation_coverage"] is None
    assert actual[0]["lead"]["strata"]["1"]["features"][0]["mean_abs"] is None


@pytest.mark.parametrize("defect", ["missing", "duplicate", "model", "infinity", "column_order", "additivity", "selection", "lineage"])
def test_fail_closed_like_legacy_and_clean_temporary_files(tmp_path, few_draws, defect):
    rows = _rows("PI72-CLEAN", stays=10)
    packets = _packets(rows, chunk=37)
    if defect == "missing":
        packets = packets[:-1]
    elif defect == "duplicate":
        packets.append(packets[0])
    elif defect == "model":
        packets[1]["model_id"] = "substituted"
    elif defect == "infinity":
        packets[0]["values"].iloc[0, 0] = np.inf
    elif defect == "column_order":
        packets[0]["values"] = packets[0]["values"].iloc[:, ::-1]
    elif defect == "additivity":
        packets[0]["margin"] = packets[0]["margin"] + 1
    elif defect == "selection":
        packets[1]["layout"] = [dict(item, feature="unselected") for item in packets[1]["layout"]]
    elif defect == "lineage":
        packets[0]["layout"] = [dict(item, dependencies=["event_day"]) for item in packets[0]["layout"]]
    with pytest.raises(it.EvaluationError):
        _legacy(rows, packets, "PI72-CLEAN")
    directory = tmp_path / "bad"
    with pytest.raises(it.EvaluationError):
        _new(rows, packets, "PI72-CLEAN", directory)
    assert list(directory.iterdir()) == []


def test_future_packet_changes_do_not_change_earlier_pod_attribution(tmp_path, few_draws):
    rows = _rows("PI72-CLEAN")
    packets = _packets(rows)
    original, _ = _new(rows, packets, "PI72-CLEAN", tmp_path / "before")
    for packet in packets:
        future = packet["rows"].pod.ge(15).to_numpy()
        packet["phi"][future] *= -3.25
        packet["margin"] = packet["phi"].sum(axis=1)
        packet["values"].iloc[np.flatnonzero(future)] = 900.
    changed, _ = _new(rows, packets, "PI72-CLEAN", tmp_path / "after")
    for name in ("preop", "POD0-2", "POD3-7", "POD8-14"):
        _exact(original[0]["pod"]["strata"][name], changed[0]["pod"]["strata"][name])
    _exact(original[1]["comparisons"][0], changed[1]["comparisons"][0])


def test_explicit_worker_and_memory_guards(tmp_path, few_draws):
    rows = _rows("PI72-CLEAN", stays=10)
    packets = _packets(rows)
    for workers in (-1, True, 1.5):
        with pytest.raises(it.EvaluationError, match="workers"):
            _new(rows, packets, "PI72-CLEAN", tmp_path / "workers", workers=workers)
    with pytest.raises(it.EvaluationError, match="RSS"):
        _new(rows, packets, "PI72-CLEAN", tmp_path / "rss", rss_limit=1)
    with pytest.raises(it.EvaluationError, match="cell memory"):
        _new(rows, packets, "PI72-CLEAN", tmp_path / "cells", cell_limit=1)


def test_mixed_layouts_fold_selections_and_partially_unavailable_values(tmp_path, few_draws):
    rows = _rows("A-formal")
    packets = _packets(rows, "R3", chunk=37)
    for packet in packets:
        fold = int(packet["rows"].fold.iloc[0])
        if fold % 2:
            packet["layout"] = [dict(item, feature="unselected" if item["feature"] == "signal" else item["feature"],
                                     lag=0 if item["kind"] == "value" else None) for item in packet["layout"]]
    packets[0].pop("values")
    expected = _legacy(rows, packets, "A-formal")
    actual, streams = _new(rows, packets, "A-formal", tmp_path / "mixed", workers=2)
    _exact(expected, actual)
    assert streams == 1
    feature = actual[0]["pod"]["strata"]["POD0-2"]["features"][0]
    assert 0 < feature["selection_frequency"] < 1
    assert feature["nonmissing_row_fraction"] is None


def test_empty_packet_stream_and_budget_errors_leave_no_arrays(tmp_path, few_draws):
    rows = _rows("PI72-CLEAN", stays=10)
    directory = tmp_path / "empty"
    with pytest.raises(it.EvaluationError, match="coverage"):
        _new(rows, [], "PI72-CLEAN", directory)
    assert list(directory.iterdir()) == []


def test_disk_cap_replays_clean_groups_without_rounding_or_double_counting(tmp_path, few_draws, monkeypatch):
    rows = _rows("PI72-CLEAN", stays=30)
    packets = _packets(rows)
    expected = _legacy(rows, packets, "PI72-CLEAN")
    write = fast._BoundedFile.write
    rejected = []
    def bounded(stream, data):
        try:
            return write(stream, data)
        except fast._SpillLimit:
            rejected.append(stream.stream.tell())
            assert stream.stream.tell() <= stream.limit
            raise
    monkeypatch.setattr(fast._BoundedFile, "write", bounded)
    actual, streams = _new(rows, packets, "PI72-CLEAN", tmp_path / "disk", disk_limit=20000)
    _exact(expected, actual)
    assert rejected and streams > 1


def test_single_stratum_disk_failure_is_explicit_and_cleans_up(tmp_path, few_draws):
    rows = _rows("PI72-CLEAN", stays=10)
    directory = tmp_path / "too_small"
    with pytest.raises(it.EvaluationError, match="temporary disk"):
        _new(rows, _packets(rows), "PI72-CLEAN", directory, disk_limit=100)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("width", [1, 2, 4, 558])
def test_bootstrap_projection_preserves_reduction_bits_including_scalar_kernel(width):
    rng = np.random.default_rng(415)
    blocks = [(np.arange(129), rng.integers(1, 31, 129).astype(float),
               rng.normal(size=(129, 5, width)) * rng.choice([1e-12, 1., 1e12], size=(129, 5, width))) for _ in range(5)]
    draws, sensitivity = fast._bootstrap_cells(blocks)
    for trial in range(8):
        weight = None if trial == 0 else np.ones(129) if trial == 1 else rng.integers(0, 5, 129)
        expected = it._cell_means(blocks, weight).mean(axis=0)[:4]
        actual = it._cell_means(draws, weight).mean(axis=0)
        expected_stay = it._cell_means(blocks, weight, equal_stays=True).mean(axis=0)[:1]
        actual_stay = it._cell_means(sensitivity, weight).mean(axis=0)[:1]
        assert actual.tobytes() == expected.tobytes()
        assert actual_stay.tobytes() == expected_stay.tobytes()
