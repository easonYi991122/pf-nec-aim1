"""NX-H4 M1 learner interfaces, with a new GAIN-only identity.

Historical menus and orchestration are not shipped.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import types
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ADDENDUM_SHA256 = "43d3c6ada7a5242165d97e7fc88375198e9b5a99a112191e969583f28f340ca7"
BASE_RUNNER_SHA256 = "948b772d38551197c6bf1c139023a3fe0b250c1e13ca474f425af87a4c4dc6d3"
LEARNERS = ("TABM-R1", "GRUD-WINDOW", "CAUSAL-MS-CNN")
PROGRAMS = dict(zip(LEARNERS, ("NX-T2-TABM-GAIN50-v1", "NX-T2-GRUD-GAIN50-v1", "NX-H4-CMSCN-INTERFACE-v1")))
CAPS = dict(zip(LEARNERS, (100000, 30000, 30000)))
FORMAT = "NX-H4-M1-v1"

_base_path = Path(__file__).with_name("sequence.py" if __package__ else "sequence_base.py")
_base_source = globals().get("_PACKAGED_BASE_SOURCE") or _base_path.read_bytes()
if hashlib.sha256(_base_source).hexdigest() != BASE_RUNNER_SHA256:
    raise ValueError("Accepted I3 runner identity changed; controller review required")
_name = (__package__ + "." if __package__ else "") + "_i6_private_sequence"
b = types.ModuleType(_name)
b.__file__ = str(_base_path)
b.__package__ = __package__ or ""
sys.modules[_name] = b
exec(compile(_base_source, str(_base_path), "exec"), b.__dict__)
SequenceError = b.SequenceError
# Preserve original validator globals before specializing the private instance.
_original_config = types.FunctionType(b._validate_config.__code__, b.__dict__.copy())
_original_pack = b._pack_rows
_original_r4 = b._validate_r4
BasePreprocessor = b.Preprocessor
EXTRA_FIELDS = {"addendum_sha256", "implementation_sha256", "representation",
                "preprocessing", "fit_pool_sha256"}
CONFIG_FIELDS = b.CONFIG_FIELDS | EXTRA_FIELDS


def implementation_sha256():
    return globals().get("_PACKAGED_IMPLEMENTATION_SHA256") or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def make_config(*, frame, repeat, recipe, window, learner, fold=-1, inner_fold=-1,
                stage="apply", chosen_epochs=None, synthetic=False):
    config = dict(format=FORMAT, spec_sha256=b.SPEC_SHA256, frame=frame, repeat=repeat,
                  fold=fold, recipe=recipe, window=window, learner=learner,
                  arm=f"NX-T2-SEQ-{recipe}-w{window}-{learner}", inner_fold=inner_fold,
                  stage=stage, features=[], synthetic=synthetic, chosen_epochs=chosen_epochs,
                  addendum_sha256=ADDENDUM_SHA256, implementation_sha256=implementation_sha256(),
                  representation="R1" if learner == "TABM-R1" else "R4",
                  preprocessing=None, fit_pool_sha256=None)
    _validate_config(config, require_features=False)
    return config


def _validate_config(config, *, require_features=True):
    if set(config) != CONFIG_FIELDS or config["format"] != FORMAT:
        raise SequenceError("Unknown M1 config fields/format")
    if config["addendum_sha256"] != ADDENDUM_SHA256 or config["implementation_sha256"] != implementation_sha256():
        raise SequenceError("M1 specification/implementation identity differs")
    if config["frame"] != "PI72-CLEAN" or config["learner"] not in LEARNERS:
        raise SequenceError("M1 is PI72-CLEAN only with three frozen candidates")
    r, w, learner = config["recipe"], config["window"], config["learner"]
    if r != "GAIN50":
        raise SequenceError("M1 handoff requires child-local GAIN50 from D5-safe")
    if config["arm"] != f"NX-T2-SEQ-{r}-w{w}-{learner}":
        raise SequenceError("Wrong M1 canonical identity")
    if config["representation"] != ("R1" if learner == "TABM-R1" else "R4"):
        raise SequenceError("Candidate representation differs")
    normalized = {k: config[k] for k in CONFIG_FIELDS - EXTRA_FIELDS}
    normalized.update(format="NX-H4-SEQ-v1", learner="TCN", arm=f"NX-T3-SEQ-{r}-w{w}-TCN")
    _original_config(normalized, require_features=require_features)
    if learner == "TABM-R1" and len(config["features"]) * w + w + 1 > 800:
        raise SequenceError("R1 exceeds 800 input columns")
    if require_features:
        pp = Preprocessor.from_state(config["preprocessing"])
        if len(pp.mean) != len(config["features"]) or not isinstance(config["fit_pool_sha256"], str) or len(config["fit_pool_sha256"]) != 64:
            raise SequenceError("Missing frozen train-only preprocessing/fit identity")


class Preprocessor(BasePreprocessor):
    """Train-only R4 statistics, GRU-D baseline and R1 sentinel adaptation."""
    @classmethod
    def fit(cls, values):
        base = BasePreprocessor.fit(values)
        obj = cls(base.median, base.mean, base.scale)
        observed = ~np.isnan(values)
        transformed = obj.transform(values)
        obj.mu = np.array([transformed[observed[:, j], j].mean() if observed[:, j].any()
                           else 0. for j in range(values.shape[1])], dtype=np.float32)
        # Canonical bytes: original NaNs replaced with 0 plus separate missing bits.
        canonical = np.asarray(values, dtype="<f4")
        obj.source_sha256 = hashlib.sha256(np.nan_to_num(canonical, nan=0).tobytes()
                                          + np.isnan(canonical).tobytes()).hexdigest()
        return obj

    def r4(self, raw):
        if raw.ndim == 3:
            return super().r4(raw)
        k = len(self.median)
        w = (raw.shape[1] - 1) // (k + 1)
        values = raw[:, :k*w].reshape(-1, k)
        out = raw.copy()
        z = self.transform(values)
        z[np.isnan(values)] = -7
        out[:, :k*w] = z.reshape(len(raw), -1)
        out[:, -1] /= w
        if not np.isfinite(out).all():
            raise SequenceError("Nonfinite R1")
        return out

    def state(self):
        return dict(super().state(), mu=self.mu.tolist(), source_sha256=self.source_sha256)

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or set(state) != {"median", "mean", "scale", "mu", "source_sha256"}:
            raise SequenceError("Unexpected M1 preprocessing state")
        base = BasePreprocessor.from_state({k: state[k] for k in ("median", "mean", "scale")})
        obj = cls(base.median, base.mean, base.scale)
        obj.mu = np.asarray(state["mu"], np.float32)
        obj.source_sha256 = state["source_sha256"]
        if obj.mu.shape != obj.mean.shape or not np.isfinite(obj.mu).all() or (not isinstance(obj.source_sha256,str) or len(obj.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in obj.source_sha256)):
            raise SequenceError("Invalid GRU-D baseline/provenance")
        return obj


class GRUD(nn.Module):
    def __init__(self, k, mu=None):
        super().__init__()
        self.k = k
        self.register_buffer("mu", torch.zeros(k) if mu is None else torch.as_tensor(mu).clone())
        self.a_x = nn.Parameter(torch.full((k,), .1))
        self.b_x = nn.Parameter(torch.zeros(k))
        self.hidden_decay = nn.Linear(k, 32)
        nn.init.constant_(self.hidden_decay.weight, .1 / k)
        nn.init.zeros_(self.hidden_decay.bias)
        self.gates = nn.ModuleDict({name: nn.Linear(2*k+1, 32) for name in ("z", "r", "n")})
        self.recurrent = nn.ModuleDict({name: nn.Linear(32, 32, bias=False) for name in ("z", "r", "n")})
        self.drop = nn.Dropout(.15)
        self.head = nn.Linear(32, 1)
        for layer in [*self.gates.values(), *self.recurrent.values(), self.head]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def sequence_logits(self, x, *, return_decay=False):
        k = self.k
        h = x.new_zeros((len(x), 32))
        last = self.mu.expand(len(x), -1)
        delta = x.new_zeros((len(x), k))
        previous_mask = x.new_zeros((len(x), k))
        logits, deltas = [], []
        for s in range(x.shape[1]):
            if s:
                delta = 1 + (1 - previous_mask) * delta
            row = x[:, s, -1:]
            m = (1 - x[:, s, k:2*k]) * row
            gamma_x = torch.exp(-F.relu(self.a_x * delta + self.b_x))
            gamma_h = torch.exp(-F.relu(self.hidden_decay(delta)))
            decayed = gamma_h * h
            x_hat = m*x[:, s, :k] + (1-m)*(gamma_x*last + (1-gamma_x)*self.mu)
            last = torch.where(m.bool(), x[:, s, :k], last)  # before dropout
            u = torch.cat((self.drop(x_hat), m, row), dim=-1)
            z = torch.sigmoid(self.gates["z"](u) + self.recurrent["z"](decayed))
            r = torch.sigmoid(self.gates["r"](u) + self.recurrent["r"](decayed))
            n = torch.tanh(self.gates["n"](u) + self.recurrent["n"](r*decayed))
            h = torch.where(row.bool(), (1-z)*decayed + z*n, decayed)
            previous_mask = m
            logits.append(self.head(h).squeeze(-1))
            deltas.append(delta)
        result = torch.stack(logits, dim=1)
        return (result, torch.stack(deltas, dim=1)) if return_decay else result

    def forward(self, x):
        return self.sequence_logits(x)[:, -1]


class MultiScaleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(32, 8, kernel) for kernel in (1, 3, 7)])
        self.fusion = nn.Linear(24, 32)
        self.drop = nn.Dropout(.15)
        self.norm = nn.LayerNorm(32)

    def forward(self, h):
        x = h.transpose(1, 2)
        branches = [conv(F.pad(x, (conv.kernel_size[0]-1, 0))) for conv in self.branches]
        fused = self.fusion(F.gelu(torch.cat(branches, dim=1).transpose(1, 2)))
        return self.norm(h + self.drop(fused))


class CausalMSCNN(nn.Module):
    receptive_field = 19

    def __init__(self, k):
        super().__init__()
        self.input = nn.Linear(2*k+1, 32)
        self.blocks = nn.Sequential(*(MultiScaleBlock() for _ in range(3)))
        self.head = nn.Linear(32, 1)

    def sequence_logits(self, x):
        return self.head(self.blocks(F.gelu(self.input(x)))).squeeze(-1)

    def forward(self, x):
        return self.sequence_logits(x)[:, -1]


class TabMR1(nn.Module):
    def __init__(self, columns):
        super().__init__()
        if columns > 800:
            raise SequenceError("TabM R1 input exceeds 800 columns")
        try:
            version = importlib.metadata.version("tabm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SequenceError("Frozen dependency tabm==0.0.3 is unavailable; no substitute") from exc
        if version != "0.0.3":
            raise SequenceError("Frozen dependency must be tabm==0.0.3")
        try:
            embedding_version = importlib.metadata.version("rtdl_num_embeddings")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SequenceError("Frozen dependency rtdl_num_embeddings==0.0.12 is unavailable") from exc
        if embedding_version != "0.0.12":
            raise SequenceError("Frozen dependency must be rtdl_num_embeddings==0.0.12")
        import tabm
        self.network = tabm.TabM.make(n_num_features=columns, cat_cardinalities=None,
            num_embeddings=None, d_out=1, n_blocks=2, d_block=64, dropout=.15,
            activation="ReLU", k=16, arch_type="tabm", start_scaling_init="random-signs")
        self.head = self.network.output

    def forward(self, x):
        logits = self.network(x).squeeze(-1)
        if logits.shape != (len(x), 16):
            raise SequenceError("TabM must return sixteen member logits per row")
        return logits


def make_model(config, prevalence, preprocessing=None):
    if not 0 < prevalence < 1:
        raise SequenceError("Single-class training")
    k, learner = len(config["features"]), config["learner"]
    if learner == "TABM-R1":
        model = TabMR1(k*config["window"] + config["window"] + 1)
    elif learner == "GRUD-WINDOW":
        model = GRUD(k, None if preprocessing is None else preprocessing.mu)
    elif learner == "CAUSAL-MS-CNN":
        model = CausalMSCNN(k)
    else:
        raise SequenceError("Unknown M1 learner")
    if b.parameter_count(model) > CAPS[learner]:
        raise SequenceError("M1 parameter cap exceeded")
    nn.init.constant_(model.head.bias, float(np.log(prevalence/(1-prevalence))))
    return model


def member_loss(logits, target):
    if logits.ndim == 2:
        target = target[:, None].expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def _predict(model, raw, preprocessing, device):
    model.eval()
    probability = np.empty(len(raw), np.float64)
    with torch.inference_mode():
        for start in range(0, len(raw), b.BATCH_SIZE):
            x = torch.from_numpy(raw[start:start+b.BATCH_SIZE]).to(device)
            p = torch.sigmoid(model(x))
            if p.ndim == 2:
                p = p.mean(dim=1)
            probability[start:start+len(x)] = p.cpu().numpy()
    return probability


def _fit_one(config, arrays, preprocessing, init, device, *, smoke=False):
    seed = b.network_seed(config, init)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    y = arrays["train_y"]
    model = make_model(config, float(y.mean()), preprocessing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    limit = b.EPOCHS_MAX if config["stage"] == "inner" else config["chosen_epochs"]
    if smoke:
        limit = min(limit, 2)
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    for epoch in range(1, limit+1):
        model.train()
        order, loss_sum = rng.permutation(len(y)), 0.
        for start in range(0, len(y), b.BATCH_SIZE):
            ids = order[start:start+b.BATCH_SIZE]
            x = torch.from_numpy(arrays["train_X"][ids]).to(device)
            target = torch.from_numpy(y[ids].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = member_loss(model(x), target)
            if not torch.isfinite(loss):
                raise SequenceError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(ids)
        record = dict(epoch=epoch, loss=loss_sum/len(y))
        if config["stage"] == "inner":
            p = _predict(model, arrays["validation_X"], preprocessing, device)
            auc = b._binary_metric(arrays["validation_y"], p, arrays["validation_score_mask"])["auc"]
            if not np.isfinite(auc):
                raise SequenceError("Undefined primary validation AUROC")
            record["validation_auc"] = auc
            if auc > best_auc + b.TOLERANCE:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if config["stage"] == "inner" and stale >= b.PATIENCE:
            break
    if config["stage"] == "inner":
        model.load_state_dict(best_state)
    else:
        best_epoch = limit
    return model.cpu(), dict(init=init, seed=seed, best_epoch=best_epoch,
        epochs_ran=len(history), parameters=b.parameter_count(model), history=history)


def _pack_rows(rows, config):
    arrays, features = _original_pack(rows, config)
    if config["learner"] == "TABM-R1":
        x, k = arrays["X"], len(features)
        values = x[..., :k].copy()
        values[x[..., -1] == 0] = np.nan
        row = x[:, ::-1, -1]
        arrays["X"] = np.concatenate((values[:, ::-1].reshape(len(x), -1), row,
                                      row.sum(axis=1, keepdims=True)), axis=1).astype(np.float32)
        arrays["availability"] = arrays["availability"][:, ::-1].copy()
    return arrays, features


def _validate_r4(arrays, prefix, config):
    if config["learner"] != "TABM-R1":
        return _original_r4(arrays, prefix, config)
    n = b._validate_keys(arrays, prefix)
    k, w = len(config["features"]), config["window"]
    x, available = arrays[prefix+"X"], arrays[prefix+"availability"]
    if x.shape != (n, k*w+w+1) or x.dtype != np.dtype("float32"):
        raise SequenceError("Raw R1 shape/dtype differs")
    values = x[:, :k*w].reshape(n, w, k)
    row = x[:, k*w:k*w+w]
    if available.shape != values.shape or available.dtype != np.dtype("bool"):
        raise SequenceError("R1 availability differs")
    if np.isinf(values).any() or not np.isin(row, [0, 1]).all() or not (row[:, 0] == 1).all():
        raise SequenceError("Invalid R1 values/row bits")
    if not np.array_equal(x[:, -1], row.sum(axis=1)):
        raise SequenceError("R1 day count differs")
    if np.isfinite(values[~available]).any() or available[row == 0].any() or np.isfinite(values[row == 0]).any():
        raise SequenceError("R1 padding/gates violated")


def _validate_arrays(config, arrays):
    """GPU receives only pretransformed inputs and minimal keyed fit metadata."""
    _validate_config(config)
    apply = config["stage"] == "apply"
    expected = {"X", "row_key"} if apply else {"train_X", "train_row_key", "train_stay", "train_y", "train_post"}
    if config["stage"] == "inner":
        expected |= {"validation_X", "validation_row_key", "validation_stay", "validation_y", "validation_score_mask"}
    if set(arrays) != expected or any(a.dtype.kind == "O" for a in arrays.values()):
        raise SequenceError("Unexpected M1 package fields; apply allows X/row_key only")
    prefixes = ("",) if apply else (("train_", "validation_") if config["stage"] == "inner" else ("train_",))
    k, w = len(config["features"]), config["window"]
    for prefix in prefixes:
        key, x = arrays[prefix+"row_key"], arrays[prefix+"X"]
        if key.ndim != 1 or not len(key) or key.dtype != np.dtype("int64") or len(np.unique(key)) != len(key):
            raise SequenceError("Missing/duplicate/non-int64 M1 row IDs")
        n = len(key)
        shape = (n, k*w+w+1) if config["representation"] == "R1" else (n, w, 2*k+1)
        if x.shape != shape or x.dtype != np.dtype("float32") or not np.isfinite(x).all():
            raise SequenceError("M1 transformed input shape/dtype/nonfinite")
        if config["representation"] == "R1":
            values, row = x[:, :k*w], x[:, k*w:k*w+w]
            if not (((values >= -6) & (values <= 6)) | (values == -7)).all():
                raise SequenceError("R1 clipped values/sentinel differ")
            if not np.isin(row, [0,1]).all() or not (row[:, 0] == 1).all() or not np.allclose(x[:, -1], row.sum(axis=1)/w, rtol=0, atol=1e-7):
                raise SequenceError("R1 row bits/normalized day count differ")
            if not (values.reshape(n,w,k)[row == 0] == -7).all():
                raise SequenceError("R1 padding is not sentinel")
        else:
            values, missing, row = x[..., :k], x[..., k:2*k], x[..., -1]
            if (np.abs(values) > 6).any() or not np.isin(missing,[0,1]).all() or not np.isin(row,[0,1]).all() or not (row[:,-1] == 1).all():
                raise SequenceError("R4 standardized values/masks differ")
            if not (values[row == 0] == 0).all() or not (missing[row == 0] == 1).all():
                raise SequenceError("R4 padding differs")
        if prefix:
            stay, y = arrays[prefix+"stay"], arrays[prefix+"y"]
            if stay.shape != (n,) or stay.dtype.kind not in "iuUf" or stay.dtype == np.dtype("float32"):
                raise SequenceError("Unsafe fit hospitalization identity")
            if (stay.dtype.kind == "U" and any(not value.strip() for value in stay)) or (stay.dtype.kind != "U" and not np.isfinite(stay).all()):
                raise SequenceError("Invalid fit hospitalization identity")
            if y.shape != (n,) or y.dtype != np.dtype("uint8") or not np.isin(y,[0,1]).all():
                raise SequenceError("Invalid M1 current-label vector")
    if not apply:
        phase = arrays["train_post"]
        if phase.shape != arrays["train_y"].shape or phase.dtype != np.dtype("uint8") or not np.isin(phase,[0,1]).all():
            raise SequenceError("Invalid training phase metadata")
    if not apply and np.unique(arrays["train_y"]).size != 2:
        raise SequenceError("Single-class training pool")
    if config["stage"] == "inner":
        if np.intersect1d(arrays["train_stay"],arrays["validation_stay"]).size:
            raise SequenceError("Train/validation hospitalization overlap")
        mask = arrays["validation_score_mask"]
        if mask.dtype != np.dtype("bool") or mask.shape != arrays["validation_y"].shape:
            raise SequenceError("Invalid postoperative scoring mask")
        if np.unique(arrays["validation_y"][mask]).size != 2:
            raise SequenceError("Single-class validation primary stratum")


def build_fit_package(path, train_rows, train_labels, config, *, validation_rows=None,
                      validation_labels=None, validation_score_mask=None):
    """Fit preprocessing on Mac on unique masked child source rows only."""
    config = dict(config)
    _validate_config(config, require_features=False)
    if config["stage"] not in ("inner", "refit"):
        raise SequenceError("Fit requires inner/refit stage")
    raw, config["features"] = _pack_rows(train_rows, config)
    _validate_r4(raw, "", config)
    source = train_rows.values.to_numpy(np.float32, copy=True)
    pp = Preprocessor.fit(source)
    config["preprocessing"] = pp.state()
    digest = hashlib.sha256()
    for name in ("row_key", "stay", "actual_date", "availability"):
        digest.update(raw[name].dtype.str.encode())
        digest.update(raw[name].tobytes())
    digest.update(pp.source_sha256.encode())
    digest.update(json.dumps(config["features"]).encode())
    config["fit_pool_sha256"] = digest.hexdigest()
    arrays = dict(train_X=pp.r4(raw["X"]), train_row_key=raw["row_key"].astype(np.int64),
                  train_stay=raw["stay"], train_y=b._labels(train_labels,train_rows).astype(np.uint8),
                  train_post=train_rows.gates.post_surg.to_numpy(np.uint8))
    if config["stage"] == "inner":
        if validation_rows is None or validation_labels is None or validation_score_mask is None:
            raise SequenceError("Inner requires validation and postoperative scoring mask")
        validation, features = _pack_rows(validation_rows,config)
        if features != config["features"]:
            raise SequenceError("Train/validation feature layout differs")
        _validate_r4(validation,"",config)
        arrays.update(validation_X=pp.r4(validation["X"]), validation_row_key=validation["row_key"].astype(np.int64),
                      validation_stay=validation["stay"], validation_y=b._labels(validation_labels,validation_rows).astype(np.uint8),
                      validation_score_mask=b._labels(validation_score_mask,validation_rows).astype(bool))
    elif any(value is not None for value in (validation_rows,validation_labels,validation_score_mask)):
        raise SequenceError("Refit must not receive external validation inputs")
    return _write_package(path,config,arrays)


def build_apply_package(path, rows, config, *, preprocessing, fit_pool_sha256):
    """Label-free apply: X and row ID only, using final refit's frozen state."""
    config = dict(config,preprocessing=preprocessing,fit_pool_sha256=fit_pool_sha256)
    _validate_config(config,require_features=False)
    if config["stage"] != "apply":
        raise SequenceError("Apply stage required")
    raw, config["features"] = _pack_rows(rows,config)
    _validate_r4(raw,"",config)
    pp = Preprocessor.from_state(preprocessing)
    return _write_package(path,config,dict(X=pp.r4(raw["X"]),row_key=raw["row_key"].astype(np.int64)))


def _write_package(path, config, arrays):
    _validate_arrays(config, arrays)
    if sum(a.nbytes for a in arrays.values()) > 1.5 * 2**30:
        raise SequenceError("Combined M1 tensors exceed 1.5 GiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as z:
        runner = (f"_PACKAGED_IMPLEMENTATION_SHA256 = {implementation_sha256()!r}\n"
                  f"_PACKAGED_BASE_SOURCE = {_base_source!r}\n").encode() + Path(__file__).read_bytes()
        z.writestr("sequence.py", runner)
        z.writestr("rx_metric.py", b._metric_source())
        z.writestr("config.json", json.dumps(config, ensure_ascii=False, allow_nan=False, indent=2))
        with z.open("arrays.npz", "w", force_zip64=True) as payload:
            np.savez_compressed(payload, **arrays)
    return dict(path=str(path), sha256=b.file_sha256(path), stage=config["stage"],
                arm=config["arm"], bytes=path.stat().st_size)


def _device(config, device):
    if not config["synthetic"] and device not in ("cuda", "auto"):
        raise SequenceError("Real M1 jobs require CUDA; CPU is synthetic only")
    if b.os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise SequenceError("Frozen CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    resolved = b.configure_torch(device)
    if resolved.type == "cuda":
        torch.cuda.set_device(0)
        resolved = torch.device("cuda:0")
    if not config["synthetic"] and resolved.type != "cuda":
        raise SequenceError("Real M1 jobs require CUDA; no CPU fallback")
    return resolved


def fit_package(directory, output, *, device="auto", synthetic_smoke=False):
    config, arrays = b.load_package(directory)
    if config["stage"] not in ("inner","refit") or (synthetic_smoke and not config["synthetic"]):
        raise SequenceError("Fit stage/synthetic smoke marker differs")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise SequenceError("Fit output directory must be empty")
    resolved = _device(config,device)
    started = time.perf_counter()
    pp = Preprocessor.from_state(config["preprocessing"])  # never fit on GPU
    members, predictions = [], []
    status = "synthetic_smoke" if synthetic_smoke else "completed"
    for init in range(3):
        model, record = _fit_one(config,arrays,pp,init,resolved,smoke=synthetic_smoke)
        torch.save(dict(format=FORMAT,config=config,init=init,status=status,
                        preprocessing=pp.state(),state_dict=model.state_dict()),output/f"init{init}.pt")
        members.append(record)
        if config["stage"] == "inner":
            predictions.append(_predict(model.to(resolved),arrays["validation_X"],pp,resolved))
    software = dict(python=platform.python_version(),torch_cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(),tabm=None,rtdl_num_embeddings=None,tabm_sha256=None)
    if config["learner"] == "TABM-R1":
        import tabm
        software.update(tabm=importlib.metadata.version("tabm"),rtdl_num_embeddings=importlib.metadata.version("rtdl_num_embeddings"),
                        tabm_sha256=file_sha256(Path(tabm.__file__)))
    receipt = dict(format=FORMAT,status=status,config=config,members=members,elapsed_seconds=time.perf_counter()-started,
                   device=str(resolved),torch_version=str(torch.__version__),numpy_version=np.__version__,
                   preprocessing=pp.state(),software=software)
    if predictions:
        probability = np.mean(predictions,axis=0)
        _write_predictions(output/"validation_predictions.npz",config,arrays,probability,prefix="validation_")
        receipt["validation_auc"] = b._binary_metric(arrays["validation_y"],probability,arrays["validation_score_mask"])["auc"]
        receipt["AUROC_post"] = receipt["validation_auc"]
        receipt["AUROC_total"] = b._binary_metric(arrays["validation_y"],probability)["auc"]
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8")
    return receipt


def _write_predictions(path, config, arrays, probability, *, prefix=""):
    key = arrays[prefix+"row_key"]
    if probability.shape != key.shape or not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
        raise SequenceError("Invalid M1 prediction vector")
    np.savez_compressed(path,row_key=key,repeat=np.full(len(key),config["repeat"],np.int64),probability=probability)


def apply_package(directory, models, output, *, device="auto"):
    config, arrays = b.load_package(directory)
    if config["stage"] != "apply" or Path(output).exists():
        raise SequenceError("Apply stage required; refuse prediction overwrite")
    resolved = _device(config, device)
    probabilities = []
    for init in range(3):
        saved = torch.load(Path(models)/f"init{init}.pt", map_location="cpu", weights_only=True)
        fitted = saved["config"]
        _validate_config(fitted)
        if saved["format"] != FORMAT or fitted["stage"] != "refit" or saved["init"] != init:
            raise SequenceError("Only three identified final refits may score apply")
        if saved["status"] not in ("completed", "synthetic_smoke") or (saved["status"] == "synthetic_smoke" and not config["synthetic"]):
            raise SequenceError("Checkpoint status differs")
        for field in CONFIG_FIELDS - {"stage", "inner_fold", "chosen_epochs"}:
            if fitted[field] != config[field]:
                raise SequenceError(f"Model/apply identity differs: {field}")
        if saved["preprocessing"] != config["preprocessing"]:
            raise SequenceError("Checkpoint/apply preprocessing state differs")
        preprocessing = Preprocessor.from_state(saved["preprocessing"])
        model = make_model(config, .5, preprocessing)
        model.load_state_dict(saved["state_dict"], strict=True)
        probabilities.append(_predict(model.to(resolved), arrays["X"], preprocessing, resolved))
    probability = np.mean(probabilities, axis=0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(output, config, arrays, probability)
    return probability


# Specialize only the private runner. Accepted sequence.py remains unmodified.
b.FORMAT = FORMAT
b.CONFIG_FIELDS = CONFIG_FIELDS
b.PACKAGE_MEMBERS = {"sequence.py", "rx_metric.py", "config.json", "arrays.npz"}
b._validate_config = _validate_config
b._validate_arrays = _validate_arrays
b._validate_r4 = _validate_r4
b._pack_rows = _pack_rows
b._write_package = _write_package
b.Preprocessor = Preprocessor
b._fit_one = _fit_one
b._predict = _predict
summarize_inner = b.summarize_inner
network_seed = b.network_seed
parameter_count = b.parameter_count
file_sha256 = b.file_sha256


def join_predictions(path, targets):
    import pandas as pd
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files) != {"repeat","row_key","probability"}:
            raise SequenceError("M1 output field whitelist differs")
        key, repeat, probability = (arrays[name] for name in ("row_key","repeat","probability"))
        if key.ndim != 1 or key.dtype != np.dtype("int64") or repeat.dtype != np.dtype("int64") or repeat.shape != key.shape or probability.shape != key.shape or probability.dtype != np.dtype("float64"):
            raise SequenceError("M1 prediction identity/dtype differs")
        prediction = pd.DataFrame({name:arrays[name] for name in arrays.files})
    keys = ["repeat","row_key"]
    if prediction.duplicated(keys).any() or targets.duplicated(keys).any():
        raise SequenceError("Duplicate M1 prediction/target keys")
    if set(map(tuple,prediction[keys].to_numpy())) != set(map(tuple,targets[keys].to_numpy())):
        raise SequenceError("Missing/extra M1 prediction keys")
    if not np.isfinite(prediction.probability).all() or not prediction.probability.between(0,1).all():
        raise SequenceError("Invalid M1 probabilities")
    return targets.merge(prediction,on=keys,how="left",validate="one_to_one",sort=False)


def main(argv=None):
    # I3's CLI dispatch uses our entry points and retains its exact arguments.
    cli = types.FunctionType(b.main.__code__, dict(b.__dict__, fit_package=fit_package,
                                                apply_package=apply_package))
    return cli(argv)


if __name__ == "__main__":
    main()
