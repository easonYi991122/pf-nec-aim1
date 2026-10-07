# Environment installation and resources

Installation terms: Python virtual environment (venv); `cu128` is the frozen torch package build tag for the corresponding NVIDIA parallel-computing platform (CUDA). Preserve the actual version string. In historical spot checks, `r0/f0` means repeat 0 / outer fold 0 and `r1` means repeat 1, under each model’s repeat convention.

Existing execution environments: Mac with an Apple M4 chip, 16 gigabyte in decimal units (GB), Python 3.9.6; Linux Python 3.12.3. Final models here run on central processing unit (CPU). The commands below start with an empty Python virtual environment (venv), assuming the relevant Python is installed and the shell is at the repository root; replace placeholder paths with writable private locations. These are installation steps for authorised partners; this revision did not install online or check a fresh environment or Linux.

## Mac

Suggestion: Match Python 3.9.6 and use the installed-version snapshot `env/requirements-mac.txt`; it includes historical tools and is not a minimal dependency lock. Import torch before Light Gradient Boosting Machine (LightGBM) on Mac to load the OpenMP numerical runtime (libomp); entry points already do so.

Names used in the example: private environment-directory variable in installation examples (`PF_ENV`); interpreter module-search-path environment variable (`PYTHONPATH`); interpreter environment variable disabling bytecode caches (`PYTHONDONTWRITEBYTECODE`); environment variable for the authorized read-only data root (`PF_DATA_ROOT`); environment variable for the authorized private cache root (`PF_CACHE_ROOT`); environment variable for the authorized private run-output directory (`PF_RUN_DIR`).

```sh
export PF_ENV=/private/pf-nec-venv
python3.9 -m venv "$PF_ENV"
. "$PF_ENV/bin/activate"
python -m pip install -r env/requirements-mac.txt
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export PF_DATA_ROOT=/authorized/pc4
export PF_CACHE_ROOT=/private/pf-nec-cache
export PF_RUN_DIR=/private/pf-nec-cache/runs
python -c 'import torch; import lightgbm; print(torch.__version__, lightgbm.__version__)'
python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
python -m pf_nec.verify
```

## Linux

Suggestion: Match Python 3.12.3. The recorded torch version is 2.8.0+cu128; install it from its official wheel index before the remaining pinned direct dependencies. A CPU-wheel substitution or upgrade requires a new environment and numerical checks; it is not assumed equivalent here. The Linux file pins direct dependencies, not a full transitive snapshot.

```sh
export PF_ENV=/private/pf-nec-venv
python3.12 -m venv "$PF_ENV"
. "$PF_ENV/bin/activate"
python -m pip install 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r env/requirements-linux.txt
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export PF_DATA_ROOT=/authorized/pc4
export PF_CACHE_ROOT=/private/pf-nec-cache
export PF_RUN_DIR=/private/pf-nec-cache/runs
python -c 'import torch; import lightgbm; print(torch.__version__, lightgbm.__version__)'
python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
python -m pf_nec.verify
```

## Execution and historical timings

Entry points set numeric threads to 2 and single-process resident set size (RSS) limit to 8 gibibyte in binary units (GiB); run sequentially. The resource guard uses Unix resource, and native Windows entry points have not been adapted. If resources are insufficient, retain completed records and stop that configuration so we can review the next budget together; reproduction retains its original repeats, menu and settings.

First code-handoff and equivalence-check record (H1) measurements (internal `h1/report.md` and equivalence receipts, not shipped; key numbers summarised here):

|Step|Wall time|Peak RSS|Check scope|
|---|---|---|---|
|Build from clean and 20 tables|156.9 seconds|2.47 GiB|13 output tables matched values, types and frame hashes|
|daily reference (A-D5-LGB) r0/f0|146.5 seconds|3.11 GiB|61,386 predictions; maximum absolute difference 6.94e-18; byte-identical model text|
|encoding-enhanced Light Gradient Boosting Machine reference (GSAFE-LGB) r1|15.6 seconds|1.34 GiB|6,787 predictions identical|

The tiny A-D5-LGB difference is **consistent with rounding**, not proof of its cause from magnitude alone. H1 did not rerun all repeats; this revision's checks do not establish full retraining of all final models or cross-platform validation. Suggestion: Add external resource monitoring for long jobs; build inputs once and retain complete shorthand for the authorized private cache root (CACHE) / shorthand for the private run-output directory (RUN) bindings for training.
