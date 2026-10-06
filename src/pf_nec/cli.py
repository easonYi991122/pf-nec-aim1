"""Explicit local commands for the three exported final-model paths."""
import argparse
import gc
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from . import config, contract as c, evaluate as ev, run, selectors as sel

MODELS = ("GSAFE-LGB", "T8-D5SAFE-LGB", "A-D5-LGB")


def context(model, repeat, fold):
    if model not in MODELS:
        raise c.ContractError("Model is excluded from this handoff")
    if model.startswith("A-"):
        if repeat not in range(5) or fold not in range(5):
            raise c.ContractError("Task A requires repeat 0..4 and fold 0..4")
        return sel.FitContext("A-formal", repeat, fold)
    if repeat not in range(1, 6) or fold != -1:
        raise c.ContractError("PI72-CLEAN uses repeats 1..5 and one held-out split, fold=-1")
    return sel.FitContext("PI72-CLEAN", repeat)


def engine_for(model, repeat, fold, guard=None):
    ctx = context(model, repeat, fold)
    provider = run.task_a_provider(ctx) if ctx.frame == "A-formal" else run.pi_provider(ctx)
    directory = config.RUN_ROOT / model / ctx.key
    engine = run.TreeEngine(ctx, *provider[:2], directory, guard=guard, gate=run.disk_gate)
    engine.i2b_cache = config.CACHE_ROOT / "gsafe" / f"r{repeat}"
    return engine


def train(model, repeat, fold=-1):
    config.require_data()
    config.initialize()
    c.reset_threads()
    guard = run.ResourceGuard(config.CACHE_ROOT, rss_limit=8 * 2**30, cache_limit=8 * 2**30)
    engine = engine_for(model, repeat, fold, guard)
    run.freeze_versions(config.RUN_ROOT)
    started = time.monotonic()
    if model == "GSAFE-LGB":
        from .gsafe import gsafe
        result = gsafe(engine)
    else:
        result = engine.apply(run.Arm("D5SAFE", frame=engine.context.frame))
    guard()
    result.update(wall_seconds=time.monotonic() - started, peak_rss_bytes=guard.peak,
                  frame=engine.context.frame, repeat=repeat, fold=fold)
    run.write_json(config.RUN_ROOT / model / f"r{repeat}-f{fold}.json", result)
    return result


def evaluate(model, repeats, *, harness=False):
    """Use frozen row scoring; full Task A repeats require all five OOF folds."""
    config.require_data()
    config.initialize()
    c.reset_threads()
    frame = "A-formal" if model.startswith("A-") else "PI72-CLEAN"
    expected, predicted = [], []
    for repeat in repeats:
        for fold in range(5) if frame == "A-formal" else [-1]:
            ctx = context(model, repeat, fold)
            provider = run.task_a_provider(ctx) if frame == "A-formal" else run.pi_provider(ctx)
            expected.append(provider[1]("test"))
            result = json.loads((config.RUN_ROOT / model / f"r{repeat}-f{fold}.json").read_text())
            predicted.append(pd.read_parquet(result["prediction_path"]))
            del provider
            gc.collect()
    targets = pd.concat(expected, ignore_index=True)
    predictions = pd.concat(predicted, ignore_index=True)
    rows, aligned = ev.align_predictions(targets, {model: predictions}, frame=frame)
    per_repeat = {str(r): ev.score_rows(rows.loc[rows.repeat.eq(r)], aligned[model][rows.repeat.eq(r)], frame=frame)
                  for r in repeats}
    mean = {metric: ev._nullable(np.mean([np.nan if per_repeat[str(r)]["metrics"][metric] is None
                                         else per_repeat[str(r)]["metrics"][metric] for r in repeats]))
            for metric in ev.METRICS}
    result = {"model": model, "frame": frame, "repeats": repeats, "per_repeat": per_repeat,
              "equal_repeat_mean": mean, "promotion_assessed": False,
              "note": "Final reference reproduction; no reduced-family promotion claim"}
    if harness:
        if frame != "A-formal":
            raise c.ContractError("harness_v3 applies only to the Task A contract here")
        from .legacy.common import load_frames
        from .harness import harness_v3 as hv
        formal = load_frames("A", "v2.6", columns=[])[0]["A"]
        matrix = np.column_stack([predictions.loc[predictions.repeat.eq(r)].set_index("row_key")
                                  .loc[formal[c.HARNESS_ROW], "probability"].to_numpy() for r in repeats])
        from .legacy.core import json_safe
        result["harness_v3"] = json_safe(hv.evaluate_A(formal, matrix))
    run.write_json(config.RUN_ROOT / model / "evaluation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-data")
    build.add_argument("--batch-stays", type=int, default=128)
    fit = commands.add_parser("train")
    fit.add_argument("--model", choices=MODELS, required=True)
    fit.add_argument("--repeat", type=int, required=True)
    fit.add_argument("--fold", type=int, default=-1)
    score = commands.add_parser("evaluate")
    score.add_argument("--model", choices=MODELS, required=True)
    score.add_argument("--repeats", type=int, nargs="+", required=True)
    score.add_argument("--harness", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build-data":
        from .data import build_frames
        config.initialize()
        guard = run.ResourceGuard(config.CACHE_ROOT, rss_limit=8 * 2**30, cache_limit=8 * 2**30)
        result = build_frames(batch_stays=args.batch_stays, guard=guard)
    elif args.command == "train":
        result = train(args.model, args.repeat, args.fold)
    else:
        result = evaluate(args.model, args.repeats, harness=args.harness)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
