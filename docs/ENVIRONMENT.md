# 环境安装与资源

〔事实〕已有运行环境：Mac M4、16 GB、Python 3.9.6；Linux Python 3.12.3。本包最终模型使用 CPU。以下从空 venv 开始，假定相应 Python 已安装且 shell 位于仓库根；将占位路径换为有权限的私有目录。这些是给获授权队友的安装命令，本次修订未联网安装、未验收全新环境或 Linux。

## Mac

〔建议〕匹配 Python 3.9.6，使用实际包版本快照 `env/requirements-mac.txt`；快照包含历史工具包，不是最小依赖锁。Mac 在 LightGBM 前导入 torch，加载 libomp；入口已这样处理。

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

〔建议〕匹配 Python 3.12.3。原记录的 torch 为 2.8.0+cu128；先从对应官方 wheel 源安装，再安装其余锁定直接依赖。CPU wheel 替代或版本升级应建立新环境并重新核对数值，不在本手册中默认为等价。Linux 文件只锁定直接依赖，不是完整传递依赖快照。

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

## 执行和历史计时

〔事实〕入口将数值线程设为 2，单进程 RSS 上限为 8 GiB；顺序运行。资源保护使用 Unix 的 resource 模块，Windows 原生入口尚未适配。不要靠删重复、改菜单或改模型配置处理资源不足。

H1 的实测（内部 `h1/report.md` 和等价性回执，不随包分发；关键数字已摘入）：

|步骤|墙钟|峰值 RSS|检查范围|
|---|---|---|---|
|从 clean 和 20 表构建|156.9 秒|2.47 GiB|13 张输出表逐值、类型与帧 hash 一致|
|A-D5-LGB r0/f0|146.5 秒|3.11 GiB|61,386 预测最大绝对差 6.94e-18；模型文本逐字节一致|
|GSAFE-LGB r1|15.6 秒|1.34 GiB|6,787 预测完全一致|

A 的极小差异 **consistent with rounding（与浮点舍入相容）**，不是仅凭差异大小证明原因。H1 没有重跑全部重复；本次修订的检查也不构成完整三模型重训或跨平台验证。〔建议〕长任务另设资源监视；输入构建仅一次，训练需保留 CACHE 与 RUN 的完整绑定。
