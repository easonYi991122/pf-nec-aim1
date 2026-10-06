"""Held-out final-model SHAP packets using the extracted G-safe hook.

Caller assembles all required repeats/folds before passing packets to
interpret.shap_views or t3_fast.shap_views. No fitting occurs in this module.
"""
import json
import numpy as np
import pandas as pd
from pf_nec import contract as c, features as f, trees, run, gsafe as hook
from . import interpret as it


def packets(engine, output):
    if output["arm"] == hook.ARM:
        info = hook._json(output["fit_info_path"])
        rows = engine.base("D5SAFE", "test")
        load_inputs = getattr(engine, "gsafe_inputs", lambda data: hook._load_encoding_inputs(data, hook._cache(engine)))
        inputs = load_inputs(rows)
        state = hook._json(info["encoding_path"])
        matrix = hook._g_matrix(rows, hook._transform(inputs, state))
    else:
        arm = run.arm_from_id(output["arm"], engine.context.frame)
        path = engine.directory / "jobs" / "outer" / arm.id / "result.json"
        info = json.loads(path.read_text())
        rows = f.select_features(engine.base(arm.recipe, "test"), info["selected"])
        matrix = trees.matrix(rows, window=arm.window, representation=arm.representation)
    if info["parameters"]["seed"] != engine.context.seed("model", output["arm"]):
        raise c.ContractError("Interpretation model belongs to another context")
    model = trees.load_fit(info["model_path"], info)
    targets = engine.metadata("test").reset_index(drop=True)
    saved = pd.read_parquet(output["prediction_path"])
    if saved.row_key.duplicated().any() or set(saved.row_key) != set(targets.row_key):
        raise c.ContractError("Interpretation prediction keys differ")
    saved = saved.set_index("row_key").loc[targets.row_key].reset_index()
    if not saved[targets.columns].equals(targets):
        raise c.ContractError("Interpretation metadata changed")
    if not np.allclose(model.predict(matrix.values), saved.probability, rtol=0, atol=1e-10):
        raise c.ContractError("Saved input/model cannot reproduce held-out predictions")
    targets = targets.assign(day=rows.keys[c.DATE].to_numpy())
    layout = it.tree_layout(matrix, window=1, representation="R1")
    for block in it.native_shap_chunks(model.model, matrix.values, learner="LGB"):
        sl = slice(block["start"], block["stop"])
        yield {**block, "rows": targets.iloc[sl], "values": matrix.values[sl],
               "layout": layout, "model_id": engine.context.key + "/" + output["arm"]}
