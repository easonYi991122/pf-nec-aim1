"""NX-H4 sequence interface on the D5-safe bank.

The controller owns grouped selection, budgets and real-data runs.
See docs/TEAM_TASKS.md; this reduced menu is a new program.
"""
import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPEC_SHA256 = "43d3c6ada7a5242165d97e7fc88375198e9b5a99a112191e969583f28f340ca7"
FORMAT = "NX-H4-SEQ-v1"
LEARNERS = ("TCN", "GRU", "ORDERLESS_MLP")
EPOCHS_MAX, PATIENCE, TOLERANCE = 60, 8, 1e-4
BATCH_SIZE, INITIALIZATIONS, PARAMETER_CAP = 512, 3, 100000
CONFIG_FIELDS = {"format", "spec_sha256", "frame", "repeat", "fold", "recipe",
                 "window", "learner", "arm", "inner_fold", "stage", "features",
                 "synthetic", "chosen_epochs"}
KEY_FIELDS = ("row_key", "stay", "actual_date")
PACKAGE_MEMBERS = {"sequence.py", "rx_metric.py", "config.json", "arrays.npz"}


class SequenceError(ValueError):
    """An input, fit or portability contract failed; no silent repairs."""


def configure_torch(device="cpu"):
    """CPU fallback on Mac; no MPS/AMP; one controller-managed Windows GPU job."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        if torch.get_num_interop_threads() > 2:
            raise SequenceError("Set torch inter-op threads <=2 before starting work")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise SequenceError("Device must be available cpu/cuda")
    return torch.device(device)


@dataclass
class Preprocessor:
    """Statistics from unique, masked CURRENT training source rows, not windows."""
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not x.shape[1] or np.isinf(x).any():
            raise SequenceError("Preprocessing requires finite/NaN unique source rows")
        median, mean, scale = np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.ones(x.shape[1])
        for j in range(x.shape[1]):
            observed = ~np.isnan(x[:, j])
            if not observed.any():
                continue
            z = np.sign(x[observed, j]) * np.log1p(np.abs(x[observed, j]))
            median[j] = np.median(z)
            filled = np.full(len(x), median[j])
            filled[observed] = z
            mean[j] = filled.mean()
            sd = filled.std(ddof=0)
            scale[j] = sd if sd > 0 else 1.0
        return cls(median, mean, scale)

    def transform(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.median) or np.isinf(x).any():
            raise SequenceError("Preprocessor input shape/infinity")
        missing = np.isnan(x)
        safe = np.where(missing, 0, x)
        z = np.sign(safe) * np.log1p(np.abs(safe))
        return np.clip((np.where(missing, self.median, z) - self.mean) / self.scale,
                       -6, 6).astype(np.float32)

    def r4(self, raw):
        """Apply fixed rowwise state; leave all missing bits and row bits intact."""
        x = np.asarray(raw, dtype=np.float32)
        k = len(self.median)
        if x.ndim != 3 or x.shape[2] != 2 * k + 1:
            raise SequenceError("Expected raw R4 [targets,window,2*k+1]")
        out = x.copy()
        out[..., :k] = self.transform(x[..., :k].reshape(-1, k)).reshape(x.shape[:2] + (k,))
        out[..., :k][x[..., -1] == 0] = 0
        if not np.isfinite(out).all():
            raise SequenceError("Nonfinite standardized R4")
        return out

    def state(self):
        return {name: getattr(self, name).tolist() for name in ("median", "mean", "scale")}

    @classmethod
    def from_state(cls, state):
        if set(state) != {"median", "mean", "scale"}:
            raise SequenceError("Unexpected preprocessing state")
        arrays = [np.asarray(state[name], dtype=np.float64) for name in ("median", "mean", "scale")]
        if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
            raise SequenceError("Invalid preprocessing state")
        if not len(arrays[0]) or any(a.shape != arrays[0].shape for a in arrays) or (arrays[2] <= 0).any():
            raise SequenceError("Invalid preprocessing dimensions/scale")
        return cls(*arrays)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.left_pad = 2 * dilation
        self.conv = nn.Conv1d(32, 32, 3, dilation=dilation)
        self.drop = nn.Dropout(.15)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        z = x + self.drop(F.gelu(self.conv(F.pad(x, (self.left_pad, 0)))))
        return self.norm(z.transpose(1, 2)).transpose(1, 2)


class TCN(nn.Module):
    receptive_field = 15

    def __init__(self, channels):
        super().__init__()
        self.input = nn.Linear(channels, 32)
        self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4)))
        self.head = nn.Linear(32, 1)

    def sequence_logits(self, x):
        h = self.blocks(self.input(x).transpose(1, 2)).transpose(1, 2)
        return self.head(h).squeeze(-1)

    def forward(self, x):
        return self.sequence_logits(x)[:, -1]


class GRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input = nn.Linear(channels, 32)
        self.drop = nn.Dropout(.15)
        self.gru = nn.GRU(32, 32, num_layers=1, batch_first=True,
                          bidirectional=False, dropout=0)
        self.head = nn.Linear(32, 1)

    def sequence_logits(self, x):
        h, _ = self.gru(self.drop(F.gelu(self.input(x))))
        return self.head(h).squeeze(-1)

    def forward(self, x):
        return self.sequence_logits(x)[:, -1]


class OrderlessMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.day = nn.Sequential(nn.Linear(channels, 32), nn.GELU(), nn.Dropout(.15),
                                 nn.Linear(32, 32), nn.GELU(), nn.Dropout(.15))
        self.head = nn.Linear(65, 1)

    def forward(self, x):
        h = self.day(x)
        observed = x[:, :-1, -1:]
        count = observed.sum(dim=1)
        history = (h[:, :-1] * observed).sum(dim=1) / count.clamp_min(1)
        return self.head(torch.cat((history, h[:, -1], count), dim=-1)).squeeze(-1)


def make_model(learner, channels, prevalence):
    if learner not in LEARNERS or not isinstance(channels, int) or channels < 3:
        raise SequenceError("Unknown learner/input dimension")
    if not 0 < prevalence < 1:
        raise SequenceError("Training pool must contain both current-label classes")
    model = {"TCN": TCN, "GRU": GRU, "ORDERLESS_MLP": OrderlessMLP}[learner](channels)
    if parameter_count(model) > PARAMETER_CAP:
        raise SequenceError("Network exceeds 100,000 parameters")
    nn.init.constant_(model.head.bias, float(np.log(prevalence / (1 - prevalence))))
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def network_seed(config, init):
    # Exactly evaluate.seed(... purpose='network', trial=-1); no profile-slot seed.
    value = (f"RX-D1-v1|{config['frame']}|{config['repeat']}|{config['fold']}|network|"
             f"{config['arm']}|{config['inner_fold']}|-1|{init}")
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little") % (2**31 - 1)


def make_config(*, frame, repeat, recipe, window, learner, fold=-1, inner_fold=-1,
                stage="apply", chosen_epochs=None, synthetic=False):
    config = dict(format=FORMAT, spec_sha256=SPEC_SHA256, frame=frame, repeat=repeat,
                  fold=fold, recipe=recipe, window=window, learner=learner,
                  arm=f"NX-T3-SEQ-{recipe}-w{window}-{learner}", inner_fold=inner_fold,
                  stage=stage, features=[], synthetic=synthetic, chosen_epochs=chosen_epochs)
    _validate_config(config, require_features=False)
    return config


def _validate_config(config, *, require_features=True):
    if set(config) != CONFIG_FIELDS or config["format"] != FORMAT or config["spec_sha256"] != SPEC_SHA256:
        raise SequenceError("Unknown config fields or frozen specification identity")
    frame, recipe, w, learner = (config[k] for k in ("frame", "recipe", "window", "learner"))
    if frame not in ("PI72-CLEAN", "A-formal") or learner not in LEARNERS:
        raise SequenceError("Unknown frame/learner")
    recipes = ("GAIN50",)
    lengths = (3, 7)
    if recipe not in recipes or w not in lengths:
        raise SequenceError("Profile outside frozen menu")
    if config["arm"] != f"NX-T3-SEQ-{recipe}-w{w}-{learner}":
        raise SequenceError("Profile slot must use canonical fit identity")
    for field in ("repeat", "fold", "inner_fold"):
        if type(config[field]) is not int:
            raise SequenceError("Repeat/fold indices must be integers")
    if config["repeat"] not in (range(1, 6) if frame == "PI72-CLEAN" else range(5)):
        raise SequenceError("Wrong repeat numbering")
    if config["fold"] not in ((-1,) if frame == "PI72-CLEAN" else range(5)):
        raise SequenceError("Wrong outer fold")
    if type(config["synthetic"]) is not bool:
        raise SequenceError("Synthetic marker must be explicit boolean")
    stage = config["stage"]
    if stage not in ("inner", "refit", "apply"):
        raise SequenceError("Stage must be inner/refit/apply")
    if config["inner_fold"] not in (range(3) if stage == "inner" else (-1,)):
        raise SequenceError("Wrong inner fold/stage")
    epochs = config["chosen_epochs"]
    if stage == "refit":
        if type(epochs) is not int or not 1 <= epochs <= EPOCHS_MAX:
            raise SequenceError("Refit requires the frozen median chosen epoch")
    elif epochs is not None:
        raise SequenceError("Chosen epochs only belong to final refit")
    features = config["features"]
    if not isinstance(features, list) or len(features) != len(set(features)):
        raise SequenceError("Features must be unique ordered names")
    expected_k = {"GAIN50": 50}[recipe]
    if require_features and len(features) != expected_k:
        raise SequenceError("Incomplete recipe feature list")
    # The builder additionally uses the bank's full dependency-closure validator.
    for name in features:
        if not isinstance(name, str) or any(token in name.lower() for token in
                ("nec", "outcome", "label", "discharge", "hospdisch", "death", "mortality",
                 "stay_length", "last_date", "lead", "rsfeedpostop", "preopriskfactor_320",
                 "preopriskfactor_330")) or name.lower() in {"y", "y3", "y30", "yb", "case", "c_h", "diff", "dis", "died", "group", "set"}:
            raise SequenceError("Forbidden feature name in package")


def _key_array(series):
    """NPZ without pickle; preserve float64/int/string identities exactly."""
    a = series.to_numpy(copy=True)
    if a.dtype.kind == "O":
        if not all(isinstance(v, str) for v in a):
            raise SequenceError("Mixed object keys cannot be packaged")
        a = np.asarray(a.tolist(), dtype=str)
    return a


def _pack_rows(rows, config):
    from pf_nec import contract as c, windows
    c.validate_keys(rows.keys, frame=config["frame"])
    if rows.frame != config["frame"]:
        raise SequenceError("Frame mismatch")
    bank = c.D5_BANK
    if rows.bank != bank:
        raise SequenceError("Recipe/source bank mismatch")
    expected = c.feature_columns(config["recipe"], frame=rows.frame,
                                 selected=list(rows.values.columns) if config["recipe"] == "GAIN50" else None)
    if tuple(rows.values.columns) != expected:
        raise SequenceError("Feature order differs from recipe")
    c.assert_predictors(expected, bank=rows.bank, dependencies=rows.dependencies)
    n, w, k = len(rows.keys), config["window"], len(expected)
    # This is a single current fit pool/profile, never an all-arm cache.
    if n * w * (2 * k + 1) * 4 + n * w * k > 1.5 * 2**30:
        raise SequenceError("One R4 package would exceed 1.5 GiB; controller must stream pools")
    x = np.empty((n, w, 2 * k + 1), np.float32)
    availability = np.empty((n, w, k), bool)
    start = 0
    for batch in windows.iter_windows(rows, window=w):
        stop = start + len(batch.keys)
        x[start:stop] = windows.r4_tensor(batch)
        availability[start:stop] = batch.available
        start = stop
    row_name = c.SOURCE_ROW if rows.frame == "PI72-CLEAN" else c.HARNESS_ROW
    arrays = {"X": x, "availability": availability,
              "row_key": _key_array(rows.keys[row_name]), "stay": _key_array(rows.keys[c.H]),
              "actual_date": _key_array(rows.keys[c.DATE])}
    return arrays, list(expected)


def _labels(labels, rows):
    from pf_nec import contract as c
    row_name = c.SOURCE_ROW if rows.frame == "PI72-CLEAN" else c.HARNESS_ROW
    # Require indexed labels: plain positional vectors can silently misalign.
    if not hasattr(labels, "index") or not labels.index.is_unique:
        raise SequenceError("Current labels must be a Series indexed by exact row identity")
    if len(labels) != len(rows.keys) or set(labels.index) != set(rows.keys[row_name]):
        raise SequenceError("Label coverage differs from feature rows")
    y = labels.reindex(rows.keys[row_name]).to_numpy()
    if y.shape != (len(rows.keys),) or not np.isin(y, [0, 1]).all():
        raise SequenceError("Only current training binary labels are allowed")
    return y.astype(np.float32)


def _metric_source():
    from pf_nec.evaluate import BinaryMetric
    # Export the EXACT accepted metric, avoiding sklearn/pandas/pyarrow on Windows.
    return "import numpy as np\n\nclass EvaluationError(ValueError):\n    pass\n\n" + inspect.getsource(BinaryMetric)


def _write_package(path, config, arrays):
    _validate_arrays(config, arrays)
    if sum(a.nbytes for a in arrays.values()) > 1.5 * 2**30:
        raise SequenceError("Combined job tensors exceed 1.5 GiB; stop and notify controller")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SequenceError("Refuse to overwrite an existing package")
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as z:
        z.writestr("sequence.py", Path(__file__).read_bytes())
        z.writestr("rx_metric.py", _metric_source())
        z.writestr("config.json", json.dumps(config, ensure_ascii=False, allow_nan=False, indent=2))
        with z.open("arrays.npz", "w", force_zip64=True) as payload:
            np.savez_compressed(payload, **arrays)
    return {"path": str(path), "sha256": file_sha256(path), "stage": config["stage"],
            "arm": config["arm"], "bytes": path.stat().st_size}


def build_apply_package(path, rows, config):
    """Build label-free target tensors+keys only; no anchors/endpoints in ZIP."""
    config = dict(config)
    if config["stage"] != "apply":
        raise SequenceError("Apply package requires apply stage")
    _validate_config(config, require_features=False)
    arrays, config["features"] = _pack_rows(rows, config)
    return _write_package(path, config, arrays)


def build_fit_package(path, train_rows, train_labels, config, *, validation_rows=None,
                      validation_labels=None, validation_score_mask=None):
    """One inner fit or full outer refit; each GAIN list is caller's child-fit list.

    PI validation_score_mask is an indexed boolean Series for post_surg==1
    (POD0 included), computed upstream. It is used ONLY for epoch scoring.
    Task A scores all validation rows. No outer test input enters this API.
    """
    config = dict(config)
    _validate_config(config, require_features=False)
    if config["stage"] not in ("inner", "refit"):
        raise SequenceError("Fit package requires inner/refit stage")
    train, config["features"] = _pack_rows(train_rows, config)
    train["y"] = _labels(train_labels, train_rows)
    train["source_values"] = train_rows.values.to_numpy(np.float32, copy=True)
    arrays = {"train_" + name: value for name, value in train.items()}
    if config["stage"] == "inner":
        if validation_rows is None or validation_labels is None:
            raise SequenceError("Inner epoch selection requires separate validation rows/labels")
        val, features = _pack_rows(validation_rows, config)
        if features != config["features"]:
            raise SequenceError("Train/validation feature layout differs")
        val["y"] = _labels(validation_labels, validation_rows)
        if config["frame"] == "PI72-CLEAN":
            if validation_score_mask is None:
                raise SequenceError("PI requires an independently indexed postoperative scoring mask")
            mask = _labels(validation_score_mask, validation_rows)
        else:
            if validation_score_mask is not None:
                raise SequenceError("Task A primary metric scores all rows")
            mask = np.ones(len(val["y"]), bool)
        val["score_mask"] = mask.astype(bool)
        arrays.update({"validation_" + name: value for name, value in val.items()})
    elif any(v is not None for v in (validation_rows, validation_labels, validation_score_mask)):
        raise SequenceError("Full refit must not read validation/test labels or tensors")
    return _write_package(path, config, arrays)


def _validate_keys(arrays, prefix):
    key = arrays[prefix + "row_key"]
    stay, date = (arrays[prefix + name] for name in ("stay", "actual_date"))
    n = len(key)
    if not n or key.shape != (n,) or key.dtype.kind not in "iu" or len(np.unique(key)) != n:
        raise SequenceError("Missing/duplicate/noninteger target row keys")
    if stay.shape != (n,) or stay.dtype.kind not in "iuUf" or stay.dtype == np.dtype("float32"):
        raise SequenceError("Unsafe hospitalization identity")
    if stay.dtype.kind == "U":
        if any(not s.strip() for s in stay):
            raise SequenceError("Empty stay identity")
    elif not np.isfinite(stay).all():
        raise SequenceError("Nonfinite stay identity")
    if date.shape != (n,) or not np.isfinite(date).all() or not np.equal(date, np.floor(date)).all():
        raise SequenceError("Dates must be integer calendar keys")
    if len(set(zip(stay.tolist(), date.tolist()))) != n:
        raise SequenceError("Duplicate stay/date")
    return n


def _validate_r4(arrays, prefix, config):
    n = _validate_keys(arrays, prefix)
    k, w = len(config["features"]), config["window"]
    x, available = arrays[prefix + "X"], arrays[prefix + "availability"]
    if x.shape != (n, w, 2 * k + 1) or x.dtype != np.dtype("float32"):
        raise SequenceError("Raw R4 shape/dtype differs")
    if available.shape != (n, w, k) or available.dtype != np.dtype("bool"):
        raise SequenceError("Availability shape/dtype differs")
    values, missing, observed = x[..., :k], x[..., k:2 * k], x[..., -1]
    if np.isinf(values).any() or not np.isin(missing, [0, 1]).all() or not np.isin(observed, [0, 1]).all():
        raise SequenceError("Invalid R4 value/mask")
    padding = observed == 0
    if not (values[padding] == 0).all() or not (missing[padding] == 1).all() or available[padding].any():
        raise SequenceError("Padding is not value0/missing1/row0")
    real = ~padding
    if not np.array_equal(np.isnan(values[real]), missing[real].astype(bool)):
        raise SequenceError("Missing flags differ from raw observed values")
    if np.isfinite(values[~available & real[..., None]]).any():
        raise SequenceError("Structurally unavailable value is finite")
    if not (observed[:, -1] == 1).all():
        raise SequenceError("Target row must be a retained observed row")


def _validate_arrays(config, arrays):
    _validate_config(config)
    common = {"X", "availability", *KEY_FIELDS}
    expected = common if config["stage"] == "apply" else {"train_" + name for name in common | {"y", "source_values"}}
    if config["stage"] == "inner":
        expected |= {"validation_" + name for name in common | {"y", "score_mask"}}
    if set(arrays) != expected or any(a.dtype.kind == "O" for a in arrays.values()):
        raise SequenceError("Unexpected package fields (outcomes allowed only as current fit y)")
    prefixes = ("",) if config["stage"] == "apply" else (("train_", "validation_") if config["stage"] == "inner" else ("train_",))
    for prefix in prefixes:
        _validate_r4(arrays, prefix, config)
        if prefix:
            y = arrays[prefix + "y"]
            if y.shape != arrays[prefix + "row_key"].shape or not np.isin(y, [0, 1]).all():
                raise SequenceError("Invalid current-label vector")
    if config["stage"] != "apply":
        source = arrays["train_source_values"]
        if source.shape != (len(arrays["train_y"]), len(config["features"])) or source.dtype != np.dtype("float32"):
            raise SequenceError("Unique fit source table dimensions")
        if not np.array_equal(source, arrays["train_X"][:, -1, :source.shape[1]], equal_nan=True):
            raise SequenceError("Fit source rows must equal target-day masked values")
        if np.unique(arrays["train_y"]).size != 2:
            raise SequenceError("Single-class training pool")
    if config["stage"] == "inner":
        if np.intersect1d(arrays["train_stay"], arrays["validation_stay"]).size:
            raise SequenceError("Train/validation hospitalization overlap")
        mask = arrays["validation_score_mask"]
        if mask.dtype != np.dtype("bool") or mask.shape != arrays["validation_y"].shape:
            raise SequenceError("Invalid validation scoring mask")
        if config["frame"] == "A-formal" and not mask.all():
            raise SequenceError("Task A primary metric requires all rows")
        if np.unique(arrays["validation_y"][mask]).size != 2:
            raise SequenceError("Single-class validation primary stratum; no redraw/substitution")


def load_package(directory):
    directory = Path(directory)
    if {p.name for p in directory.iterdir() if p.is_file()} != PACKAGE_MEMBERS:
        raise SequenceError("Extract into a clean input directory with exactly four files")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    _validate_arrays(config, arrays)
    return config, arrays


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _binary_metric(y, probability, mask=None):
    if __package__:
        from pf_nec.evaluate import BinaryMetric
    else:
        from rx_metric import BinaryMetric
    return BinaryMetric(y, probability, mask).score()


def _predict(model, raw, preprocessing, device):
    model.eval()
    probability = np.empty(len(raw), np.float64)
    with torch.inference_mode():
        for start in range(0, len(raw), BATCH_SIZE):
            x = torch.from_numpy(preprocessing.r4(raw[start:start + BATCH_SIZE])).to(device)
            probability[start:start + len(x)] = torch.sigmoid(model(x)).cpu().numpy()
    return probability


def _fit_one(config, arrays, preprocessing, init, device, *, smoke=False):
    seed = network_seed(config, init)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    y = arrays["train_y"]
    model = make_model(config["learner"], arrays["train_X"].shape[-1], float(y.mean())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1., device=device))
    limit = EPOCHS_MAX if config["stage"] == "inner" else config["chosen_epochs"]
    if smoke:
        limit = min(limit, 2)
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(len(y))
        loss_sum = 0.
        for start in range(0, len(y), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            x = torch.from_numpy(preprocessing.r4(arrays["train_X"][ids])).to(device)
            target = torch.from_numpy(y[ids].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            if not torch.isfinite(loss):
                raise SequenceError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(ids)
        record = {"epoch": epoch, "loss": loss_sum / len(y)}
        if config["stage"] == "inner":
            p = _predict(model, arrays["validation_X"], preprocessing, device)
            auc = _binary_metric(arrays["validation_y"], p, arrays["validation_score_mask"])["auc"]
            if not np.isfinite(auc):
                raise SequenceError("Undefined primary validation AUROC")
            record["validation_auc"] = auc
            if auc > best_auc + TOLERANCE:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if config["stage"] == "inner" and stale >= PATIENCE:
            break
    if config["stage"] == "inner":
        model.load_state_dict(best_state)
    else:
        best_epoch = limit
    return model.cpu(), {"init": init, "seed": seed, "best_epoch": best_epoch,
                         "epochs_ran": len(history), "parameters": parameter_count(model), "history": history}


def fit_package(directory, output, *, device="auto", synthetic_smoke=False):
    """Fit three initializations; final refit reads no external/apply inputs."""
    config, arrays = load_package(directory)
    if config["stage"] not in ("inner", "refit"):
        raise SequenceError("Cannot fit an apply package")
    if synthetic_smoke and not config["synthetic"]:
        raise SequenceError("Micro-smoke is permitted only for declared synthetic inputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SequenceError("Fit output directory must be empty")
    device = configure_torch(device)
    started = time.perf_counter()
    preprocessing = Preprocessor.fit(arrays["train_source_values"])
    members, predictions = [], []
    for init in range(INITIALIZATIONS):
        model, record = _fit_one(config, arrays, preprocessing, init, device, smoke=synthetic_smoke)
        torch.save({"format": FORMAT, "config": config, "init": init,
                    "status": "synthetic_smoke" if synthetic_smoke else "completed",
                    "preprocessing": preprocessing.state(),
                    "state_dict": model.state_dict()}, output / f"init{init}.pt")
        members.append(record)
        if config["stage"] == "inner":
            predictions.append(_predict(model.to(device), arrays["validation_X"], preprocessing, device))
    receipt = {"format": FORMAT, "status": "synthetic_smoke" if synthetic_smoke else "completed",
               "config": config, "members": members, "elapsed_seconds": time.perf_counter() - started,
               "device": str(device), "torch_version": str(torch.__version__),
               "numpy_version": np.__version__, "preprocessing": preprocessing.state()}
    if predictions:
        probability = np.mean(predictions, axis=0)
        _write_predictions(output / "validation_predictions.npz", config, arrays, probability, prefix="validation_")
        receipt["validation_auc"] = _binary_metric(arrays["validation_y"], probability,
                                                    arrays["validation_score_mask"])["auc"]
        receipt["AUROC_total"] = _binary_metric(arrays["validation_y"], probability)["auc"]
        receipt["AUROC_post"] = receipt["validation_auc"] if config["frame"] == "PI72-CLEAN" else None
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return receipt


def summarize_inner(receipts):
    """Three folds x three best epochs; integer median (nine values), fold mean."""
    if len(receipts) != 3 or any(r["status"] != "completed" or r["config"]["stage"] != "inner" for r in receipts):
        raise SequenceError("Three completed inner receipts required; smoke is not selection")
    if {r["config"]["inner_fold"] for r in receipts} != {0, 1, 2}:
        raise SequenceError("Inner folds must be exactly 0/1/2")
    identity = ("frame", "repeat", "fold", "arm", "synthetic")
    if any(any(r["config"][k] != receipts[0]["config"][k] for k in identity) for r in receipts):
        raise SequenceError("Cannot average different outer contexts/profiles")
    epochs = []
    for receipt in receipts:
        if len(receipt["members"]) != 3 or {m["init"] for m in receipt["members"]} != {0, 1, 2}:
            raise SequenceError("Three initialization receipts required per fold")
        epochs.extend(m["best_epoch"] for m in receipt["members"])
    if any(type(e) is not int or not 1 <= e <= 60 for e in epochs):
        raise SequenceError("Invalid best epochs")
    scores = [r["validation_auc"] for r in receipts]
    if not np.isfinite(scores).all():
        raise SequenceError("Undefined inner primary score")
    return {"arm": receipts[0]["config"]["arm"], "chosen_epochs": int(np.median(epochs)),
            "mean_inner_auc": float(np.mean(scores)), "best_epochs": epochs}


def _write_predictions(path, config, arrays, probability, *, prefix=""):
    if probability.shape != arrays[prefix + "row_key"].shape or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise SequenceError("Invalid prediction vector")
    if Path(path).suffix != ".npz":
        raise SequenceError("Prediction file must have .npz suffix")
    keys = {name: arrays[prefix + name] for name in KEY_FIELDS}
    keys["repeat"] = np.full(len(probability), config["repeat"], np.int64)
    if config["frame"] == "A-formal":
        keys["fold"] = np.full(len(probability), config["fold"], np.int64)
    np.savez_compressed(path, **keys, probability=probability)


def apply_package(directory, models, output, *, device="auto"):
    """Predictions in exact packaged row order, three-init mean probabilities."""
    config, arrays = load_package(directory)
    if config["stage"] != "apply":
        raise SequenceError("Apply must receive a label-free apply package")
    output, models = Path(output), Path(models)
    if output.exists():
        raise SequenceError("Refuse to overwrite predictions")
    device = configure_torch(device)
    probabilities = []
    for init in range(INITIALIZATIONS):
        saved = torch.load(models / f"init{init}.pt", map_location="cpu", weights_only=True)
        fitted = saved["config"]
        if saved["format"] != FORMAT or fitted["stage"] != "refit":
            raise SequenceError("Only final full-pool refit models may score apply rows")
        if saved["init"] != init or saved["status"] not in ("completed", "synthetic_smoke"):
            raise SequenceError("Missing/duplicated initialization identity")
        if saved["status"] == "synthetic_smoke" and not config["synthetic"]:
            raise SequenceError("Smoke weights cannot score real apply rows")
        for field in ("frame", "repeat", "fold", "arm", "features", "spec_sha256", "synthetic"):
            if fitted[field] != config[field]:
                raise SequenceError(f"Model/apply context differs: {field}")
        model = make_model(config["learner"], arrays["X"].shape[-1], .5)
        model.load_state_dict(saved["state_dict"], strict=True)
        preprocessing = Preprocessor.from_state(saved["preprocessing"])
        probabilities.append(_predict(model.to(device), arrays["X"], preprocessing, device))
    probability = np.mean(probabilities, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(output, config, arrays, probability)
    return probability


def join_predictions(path, targets):
    """Controller-side exact join to evaluation metadata; NEVER runs in apply.

    Returns the same columns as the tree arms (targets + probability), preserving
    target order. Full-repeat/phase validation remains evaluate's responsibility.
    A single Task A fit may cover one fold; combine all five before evaluation.
    """
    import pandas as pd
    from pf_nec.evaluate import KEY
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    expected = {*KEY_FIELDS, "repeat", "probability"}
    if "fold" in targets:
        expected.add("fold")
    if set(data) != expected or not {*KEY, "stay"} <= set(targets):
        raise SequenceError("Prediction/target key schema differs")
    prediction = pd.DataFrame(data)
    if targets[KEY].isna().any().any() or targets.duplicated(KEY).any() or prediction.duplicated(KEY).any():
        raise SequenceError("Missing/duplicate prediction or target keys")
    target_index = pd.MultiIndex.from_frame(targets[KEY])
    prediction_index = pd.MultiIndex.from_frame(prediction[KEY])
    order = prediction_index.get_indexer(target_index)
    if len(prediction) != len(targets) or (order < 0).any():
        raise SequenceError("Prediction coverage differs; no inner join")
    aligned = prediction.iloc[order].reset_index(drop=True)
    for name in ("stay", "fold", "actual_date"):
        if name in targets:
            if targets[name].dtype == np.dtype("float32") or not np.array_equal(aligned[name].to_numpy(), targets[name].to_numpy()):
                raise SequenceError(f"Prediction target identity differs: {name}")
    p = aligned.probability.to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise SequenceError("Invalid prediction probabilities")
    return targets.assign(probability=p)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--package", required=True, help="Clean extracted job directory")
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    fit.add_argument("--synthetic-smoke", action="store_true")
    apply = sub.add_parser("apply")
    apply.add_argument("--package", required=True)
    apply.add_argument("--models", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.command == "fit":
        receipt = fit_package(args.package, args.output, device=args.device, synthetic_smoke=args.synthetic_smoke)
        print(json.dumps({"status": receipt["status"], "arm": receipt["config"]["arm"],
                          "elapsed_seconds": receipt["elapsed_seconds"]}, allow_nan=False))
    else:
        p = apply_package(args.package, args.models, args.output, device=args.device)
        print(json.dumps({"status": "completed", "predictions": len(p)}))


if __name__ == "__main__":
    main()
