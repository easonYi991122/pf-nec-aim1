import json
import numpy as np
import pandas as pd
from explore.aim1b import stage12, stage34
from explore.i7 import diagnostics
from pf_nec import selectors
from test_gsafe import sample


def test_aim1b_synthetic_pipeline_and_embedded_causality_tests():
    result = stage12.synthetic_run()
    assert result["synthetic_only"]
    stage12.test_causality_future_changes_leave_prefix_unchanged()
    stage12.test_first_interval_selected_without_future_end()
    result = stage34.run_diagnostics(stage34.synthetic_tables(120), pods=(1,), graces=(1,))
    assert result["index_stays"] == 120
    assert len(result["diagnostics"]) > 0
    json.dumps(result, allow_nan=False)


def test_i7_nested_case_control_subsets_and_background_definition():
    _, meta = sample(80)
    ctx = selectors.FitContext()
    a = set(diagnostics.learning_curve_stays(meta, ctx, .25))
    b = set(diagnostics.learning_curve_stays(meta, ctx, .5))
    d = set(diagnostics.learning_curve_stays(meta, ctx, 1.))
    assert len(a) == 20 and len(b) == 40 and len(d) == 80 and a < b < d
    background = next(arm for arm in diagnostics.slice_arms() if arm.recipe == "B")
    assert background.k == 8 and background.window == 1


def test_i7_background_model_synthetic_end_to_end(tmp_path):
    train, test = sample(120), sample(30, 200)
    loader = lambda recipe, partition: test[0] if partition == 'test' else train[0]
    metadata = lambda partition: test[1] if partition == 'test' else train[1]
    engine = diagnostics.SliceEngine(selectors.FitContext(), loader, metadata, tmp_path / 'i7')
    arm = next(arm for arm in diagnostics.slice_arms() if arm.recipe == 'B')
    output = engine.apply(arm)
    predictions = pd.read_parquet(output['prediction_path'])
    assert len(predictions) == len(test[1]) and np.isfinite(predictions.probability).all()
    assert len(engine.ledger.records) == 4
