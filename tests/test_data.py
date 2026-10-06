import numpy as np
import pandas as pd
import pytest
from pf_nec import data, contract as c
from pf_nec.legacy import build_v26, line_S, wp0


def test_private_mapping_reconstruction_uses_codes_not_outcomes():
    clean = pd.DataFrame({c.H: [1., 2., 3.], "funddiagnosis_a": [1, 0, 1],
                          "procprimary_b": [0, 1, 0], "PreopRiskFactor_c": [1, 1, 0]})
    raw = {"IndexSurgHosp": pd.DataFrame({"hospitalizationidNEW": ["1", "2", "3"],
        "FundDiagnosis": ["10", "20", "30"], "ProcPrimary": ["40", "50", "60"]}),
        "PreopRiskFactor": pd.DataFrame({"hospitalizationidNEW": ["1", "2", "3"], "preopfactor": ["70", "70", "80"]})}
    defs, checks = data.mapping_inputs(clean, raw)
    assert defs["funddiagnosis_a"]["members"] == [{"code": "10"}, {"code": "30"}]
    assert checks["risk_minimum_compatible_groups"]["PreopRiskFactor_c"]["minimal_compatible_code_sets"] == [["70"]]
    # Duplicate clean days and unrelated outcome changes cannot change mapping.
    changed = pd.concat([clean, clean]).assign(outcome_3d=[1, 0, 0, 0, 1, 1])
    assert data.mapping_inputs(changed, raw) == (defs, checks)


def test_split_is_stay_disjoint_reproducible_and_preserves_fractional_ids():
    strata = pd.Series([1] * 8 + [0] * 16, index=np.arange(24, dtype=float) + .125)
    a = wp0.make_split(strata, 73, n_controls=12, case_train=6, control_train=9)
    b = wp0.make_split(strata, 73, n_controls=12, case_train=6, control_train=9)
    assert set(a["train"]).isdisjoint(a["test"])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert all(v % 1 == .125 for v in a["train"])


def test_support_features_ignore_unobserved_future_intervals():
    keys = pd.DataFrame({c.H: [1., 1., 1.], c.DATE: [100., 101., 102.]})
    intervals = {name: {"1": np.array([[100., 105., 104., 1.]])} for name in line_S.SOURCES}
    before = line_S.build_features(keys, intervals)
    altered = {name: {"1": np.array([[100., 200., 199., 1.], [110., 113., 112., 1.]])} for name in line_S.SOURCES}
    after = line_S.build_features(keys, altered)
    pd.testing.assert_frame_equal(before, after)
    assert before.shape == (3, 15)
