# Evaluation contract and executable examples

[fact] Numerical logic in frozen `evaluate.py` and harness v3/v2 is unchanged. `pf_nec.inference.single_arm_intervals` only composes the original evaluator's alignment, sampling and scoring functions; it does not fabricate a second arm.

|Field|Meaning / check|
|---|---|
|repeat, row_key|Unique row key per repeat; predictions must completely match independent targets|
|stay, case|Stay key and case-stay indicator; case is not the daily y|
|y, pod|Daily binary label and original POD; never infer the target universe from predictions|
|fold|Task A requires folds 0–4; pool complete OOF within each repeat before scoring|
|probability|Predicted probability, finite and within [0, 1]|

Do not hide lost rows with an inner join or treat repeated days as independent observations; weight repeat metrics equally. AP/Brier describe sampled retained days, not deployment calibration.

## Single-arm descriptive intervals

Complete every required model fit in README, then run from the repository root. `--descriptive-ci` uses case/control-stratified bootstrap of the union of stays; `--harness` can also be included, but label the two interval sets separately.

```sh
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --harness
```

## Paired evaluation of the two PI72-CLEAN handoff references

Save the following outside the repository and execute it with the configured interpreter and `PYTHONPATH`. It compares G-safe with D5-safe. **The explicit family contains only this descriptive contrast**, not the full historical promotion family; a positive local interval cannot establish promotion. A future candidate compared against both references needs a separately predeclared set of contrasts and family; this example cannot replace the complete historical family. To reuse accepted private outputs, replace the loading lines with `pd.read_parquet` of independent targets and both prediction tables; the same fields and completeness checks apply, and targets must not be reconstructed from predictions.

```python
from pf_nec import cli, config, evaluate as ev, run
import pandas as pd

repeats = [1, 2, 3, 4, 5]
targets, g = cli.load_evaluation_inputs("GSAFE-LGB", repeats)
other_targets, d = cli.load_evaluation_inputs("T8-D5SAFE-LGB", repeats)
pd.testing.assert_frame_equal(targets, other_targets)
family = [{"arm": "GSAFE-LGB", "reference": "T8-D5SAFE-LGB", "metric": "AUROC_post"}]
result = ev.evaluate(targets, {"GSAFE-LGB": g, "T8-D5SAFE-LGB": d},
                     frame="PI72-CLEAN", stage="confirm", contrasts=family, family=family)
assert not result["family_complete"]
assert not result["comparisons"][0]["promotion"]["passes"]
run.write_json(config.writable(config.RUN_ROOT / "reference_pair.json"), result)
```

## Interval algorithms and outputs

|Entry|Bootstrap|Interpretation|
|---|---|---|
|Task A `--harness`|Default 400 draws, unstratified stay resampling|Frozen harness's earlier descriptive convention, not the interval algorithm in R11's main table|
|`--descriptive-ci` / `ev.evaluate`|2000 attempts, stratified by case/control; shared multiplicity for the same stay across repeats, phases and arms|Percentile descriptive intervals conditional on fixed predictions; at least 1990 valid draws per item; no redrawing invalid samples|
|Simultaneous intervals for explicit family|At least 1990 jointly valid draws out of 2000; maximum absolute centred error|Only the complete frozen family may support promotion; a subfamily never replaces it|

R9/R11 main tables use the latter conventions. The gate remains MDD=0.01, 5 complete repeats and simultaneous lower bound>0. A 3-repeat screen never promotes; reusing it in 5 repeats does not provide independent confirmation.

Output: `PF_RUN_DIR/<model>/evaluation.json`; paired example: `PF_RUN_DIR/reference_pair.json`. The full single-arm JSON also contains `per_repeat` (metrics/counts/decomposition per repeat), all phase metrics and conditionality flags. Below is only the **aggregate-field projection** to check; numbers are from internal `r9/report.md`, not shipped; key numbers summarised here:

```json
{
  "model": "GSAFE-LGB",
  "frame": "PI72-CLEAN",
  "repeats": [1, 2, 3, 4, 5],
  "equal_repeat_mean": {"AUROC_post": 0.7329, "AUROC_total": 0.8095},
  "promotion_assessed": false,
  "descriptive_ci": {
    "attempted_draws": 2000,
    "arm_intervals": {
      "AUROC_post": {"estimate": 0.7329, "valid_draws": 2000,
                     "undefined_draws": 0, "percentile_ci95": [0.7020, 0.7622]}
    }
  }
}
```

[suggestion] Counts, keys and repeat sets must match exactly. For the 4-decimal values here, compare with JSON using absolute tolerance 0.00005; for 5 decimals use 0.000005. These are display-rounding tolerances only. Repeated computation with identical inputs / versions should be deterministic; cross-platform differences require checking models, parameters, stopping rounds and predictions, not hiding training changes within display tolerance. `null` means undefined or insufficient support, not 0; do not drop that repeat before averaging. Paired JSON must include `comparisons`, `arm_intervals`, `simultaneous_family`, `family_complete=false`, `attempted_draws` and valid-draw counts.

## Phase decomposition

[fact] PI72-CLEAN pre-op labels are all negative, so `AUROC_total = f_pre × cross_stage_AUC + (1−f_pre) × AUROC_post`, with all negative days as the denominator of f_pre. R9 f_pre ranges from 0.270–0.301; cross-stage arm-mean AUC was **measured** at 0.999918–1.000000, not inferred from all-negative labels. Within-pre-op AUROC is null.

[fact] Task A includes pre-op positives. R11's A-D5-LGB decomposition below contains 5-repeat means; the weighted contributions reconstruct total 0.71751. The general `cross_stage_positive_vs_preop_negative_auc` field compares all positives against pre-op negatives, including pre-op positives; it is not just the post-op-positive cell.

|Positive-day phase / negative-day phase|Weight|AUC|
|---|---|---|
|Pre / pre|0.02653|0.54227|
|Pre / post|0.07666|0.14612|
|Post / pre|0.23059|0.96123|
|Post / post|0.66622|0.70588|

Sources: `r9/report.md`, `r11/report.md` (internal reports, not shipped; key numbers summarised here). Check the identity at full precision; rounded table entries cannot provide bitwise equality.
