import os
import sys
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import torch
torch.set_num_threads(2)

import pytest
from pf_nec import config, contract


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "private")
    monkeypatch.setattr(config, "RUN_ROOT", tmp_path / "private/runs")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "read-only-input")
    config.initialize()
    contract.reset_threads()
