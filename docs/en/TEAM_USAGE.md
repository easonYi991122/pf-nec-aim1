# bounded tabular and missingness-modeling task (T-2) / early-postoperative ordering task (T-3): starting with synthetic interfaces

Compact fields: `fit_pool_sha256` is the Secure Hash Algorithm digest (SHA-256) of fitting-pool identity; `gestagewks` is gestational age in weeks; `RUN_ROOT` is the resolved output root; `validation_auc` is inner-validation area under the receiver operating characteristic curve (AUROC); the file and integrity-hash inventory (MANIFEST) checks delivered-file identity.

If you want to learn the interfaces, start with this synthetic example; these are the steps we use to check a single fit. Run from the repository root with the configured interpreter. The example verifies a single fit/apply, not complete new tasks. Source entries: [input object with daily-row identities, predictors and availability metadata (FeatureRows)](../../src/pf_nec/features.py), [fit-context object binding training-pool identity and seeds (FitContext)](../../src/pf_nec/selectors.py); full causality / separation tests: [test_team.py](../../tests/test_team.py), [test_team_sequence.py](../../tests/test_team_sequence.py), [test_team_m1.py](../../tests/test_team_m1.py).

Names used in the example: interpreter module-search-path environment variable (`PYTHONPATH`); interpreter environment variable disabling bytecode caches (`PYTHONDONTWRITEBYTECODE`); environment variable for the authorized private run-output directory (`PF_RUN_DIR`).

```sh
PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_team.py tests/test_team_sequence.py tests/test_team_m1.py -p no:cacheprovider --basetemp="$PF_RUN_DIR/team-tests"
```

Time notation: postoperative day (POD), counted from zero on surgery day.

|Object / stage|Use and separation|
|---|---|
|FeatureRows|keys contain stay / date / exact daily-row identity; values contain predictors gated on each historical day; available, gates and admission are metadata. Create through build_features or a reviewed adapter, without outcomes. Keep labels in a separate row-identifier (ID)-indexed Series, never indexed by stay or implicit position.|
|FitContext / child|FitContext records frame, repeat, fold, inner and pool and binds seeds. A child is the current inner-training subset or outer-refit training pool; feature ranking by tree gain within the current training pool (GAIN) and scaling fit only this pool.|
|inner|Separate train / validation by stay; index validation_labels and validation_score_mask by exact daily-row ID. The mask defines inner scoring rows; never provide outer-test labels. T-3 POD0–2 adaptation remains disabled.|
|refit|After inner selection, rerank GAIN and fit scaling / models on the full outer-training pool. Use selected epochs, without outer-test labels or held-out early stopping.|
|apply|Accept selected-column predictors and row identity, without labels. Historical neural-model selection family (M1) also requires preprocessing and fit_pool_sha256 from the refit package; never refit on apply rows. Prediction order follows packaged row_key.|

The block runs a synthetic chain once for sequence gated recurrent unit (GRU) and once for M1 gated recurrent unit with missingness decay (GRU-D). Execute it with `PYTHONPATH=src:tests`; `sample` and `extract` are shipped test fixtures. The fixed 2 epochs are for smoke only, not inner-selection results; temporary outputs are written under PF_RUN_DIR and cleaned up.

Names used in the example: Light Gradient Boosting Machine (`LightGBM`); local cleaned-table evaluation contract (`PI72-CLEAN`); configuration key for the missingness-decay window sequence learner (`GRUD-WINDOW`); top fifty columns ranked by training-pool gain (`GAIN50`).

```python
import tempfile
from pathlib import Path
import torch  # Import before LightGBM on Mac.
import pandas as pd
from pf_nec import config, selectors
from explore.team import inputs, sequence, sequence_m1
from test_gsafe import sample
from test_team import extract

config.initialize()
train, meta = sample(120)  # Entirely synthetic FeatureRows and metadata.
train.values.loc[:, "gestagewks"] = 30 + 6 * meta.case.to_numpy()
labels = pd.Series(meta.y.to_numpy(), index=meta.row_key)
held, _ = sample(12, 100)
context = selectors.FitContext(frame="PI72-CLEAN", repeat=1, pool="outer")
for runner, learner in ((sequence, "GRU"), (sequence_m1, "GRUD-WINDOW")):
    with tempfile.TemporaryDirectory(dir=config.RUN_ROOT, prefix="team-smoke-") as folder:
        out = Path(folder)
        rows, selected = inputs.gain50(train, labels, context)
        held_rows = inputs.selected_rows(held, selected)
        common = dict(frame="PI72-CLEAN", repeat=1, recipe="GAIN50",
                      window=7, learner=learner, synthetic=True)
        fit_cfg = runner.make_config(**common, stage="refit", chosen_epochs=2)
        packed = runner.build_fit_package(out/"fit.zip", rows, labels, fit_cfg)
        fit_dir = extract(packed["path"], out/"fit")
        receipt = runner.fit_package(fit_dir, out/"models", device="cpu", synthetic_smoke=True)
        state = {} if runner is sequence else dict(
            preprocessing=receipt["preprocessing"], fit_pool_sha256=receipt["config"]["fit_pool_sha256"])
        apply_cfg = runner.make_config(**common, stage="apply")
        packed = runner.build_apply_package(out/"apply.zip", held_rows, apply_cfg, **state)
        apply_dir = extract(packed["path"], out/"apply")
        probability = runner.apply_package(apply_dir, out/"models", out/"predictions.npz", device="cpu")
        assert probability.shape == (len(held_rows.keys),)
```

## Configuration decisions

|Item|Already specified|Configuration to decide together|
|---|---|---|
|T-2 menu and inner summary|GAIN50-three-day calendar window (w3)/seven-day calendar window (w7); summarize_inner averages validation_auc equally across 3 folds and takes the integer median of 3×3 best_epoch values for refit|Record the new program’s window ranking, tie tolerance / tiebreak in its configuration appendix; no real window selection until frozen, and no automatic inheritance of the full historical M1 menu|
|T-2 references|Shipped full availability-gated tree reference for the local cleaned-table contract (T8-D5SAFE-LGB), encoding-enhanced Light Gradient Boosting Machine reference (GSAFE-LGB) and frozen configurations are located in the final spec|The new comparison driver must bind identical rows / labels and jointly recorded fixed configurations; synthetic examples do not establish clinical readiness|
|T-3 inputs and controls|training-pool gain-selected configuration with a seven-day calendar window (GAIN50-w7); temporal convolutional network (TCN) / GRU versus orderless multilayer perceptron (MLP), window tree and clock-only tree|Separately freeze POD0–2 rows / scoring / receipt adaptation, exact new-task lag-flattened-window tree model (R1-LGB) and clock-only Light Gradient Boosting Machine learner code (LGB) configurations, and the complete joint family; current interfaces do not execute T-3|
|Extension location|Add independently identified drivers / adapters and synthetic tests beside src/explore/team|Changing an existing learner, base runner or team spec invites joint review, synchronized versions / binding hashes, a rebuilt MANIFEST and tests to preserve comparability; do not modify final-model spec or harness|

parameter-sharing ensemble of small tabular models (TabM) is lag-flattened configuration key for the parameter-sharing tabular model (TABM-R1) in sequence_m1; test_team_m1.py covers its call chain and exact dependencies, without library substitution. Full historical reconstructed feature-bank name (D5) neural inputs, complete task scheduling, new POD0–2 evaluation and alert drivers remain not shipped. We discuss research questions and run arrangements together and use authorized hosts; see [TEAM_TASKS](TEAM_TASKS.md) for gates and budgets.

## Scope of exploration-specification gate values

The false gates and field names in the [exploration specification](../../src/explore/team/spec.json) retain their handoff values. The original builder writes them as constants; the shipped test checks only that real runs are not automatically authorized. No per-run initialization or update mechanism is defined, so this package cannot certify that they are defaults for each new run. These static gates do not replace [current status](STATUS_AND_NEXT.md). Every real run still requires prior recorded task-card review, configuration freeze, input-timing checks and data authorization; freezing a research addendum does not establish clinical endorsement.
