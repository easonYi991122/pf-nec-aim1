# 环境安装与资源

安装术语：Python 虚拟环境（venv）；`cu128` 是冻结 torch 安装包对应的英伟达并行计算平台（CUDA）构建标记，实际版本字符串保持原样。历史抽查的 `r0/f0` 指重复 0／外折 0，`r1` 指重复 1；它们属于各自模型的重复编号规则。

已有运行环境：Mac Apple M4 芯片、16 十进制吉字节（GB）、Python 3.9.6；Linux Python 3.12.3。本包最终模型使用中央处理器（CPU）。以下从空的 Python 虚拟环境（venv） 开始，假定相应 Python 已安装且 shell 位于仓库根；将占位路径换为有权限的私有目录。这些是供获授权伙伴使用的安装步骤，本次修订未联网安装、未验收全新环境或 Linux。

## Mac

建议：匹配 Python 3.9.6，使用实际包版本快照 `env/requirements-mac.txt`；快照包含历史工具包，不是最小依赖锁。Mac 在轻量梯度提升机（LightGBM）前导入 torch，加载 OpenMP 数值运行库（libomp）；入口已这样处理。

以下示例中的名称：安装示例中的私有环境目录变量（`PF_ENV`）；解释器模块搜索路径环境变量（`PYTHONPATH`）；禁止生成字节码缓存的解释器环境变量（`PYTHONDONTWRITEBYTECODE`）；授权只读数据根目录环境变量（`PF_DATA_ROOT`）；授权私有缓存根目录环境变量（`PF_CACHE_ROOT`）；授权私有运行输出目录环境变量（`PF_RUN_DIR`）.

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

建议：匹配 Python 3.12.3。原记录的 torch 为 2.8.0+cu128；先从对应官方 wheel 源安装，再安装其余锁定直接依赖。CPU wheel 替代或版本升级应建立新环境并重新核对数值，不在本手册中默认为等价。Linux 文件只锁定直接依赖，不是完整传递依赖快照。

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

入口将数值线程设为 2，单进程常驻内存（RSS）上限为 8 二进制吉字节（GiB）；顺序运行。资源保护使用 Unix 的 resource 模块，Windows 原生入口尚未适配。资源不足时保留已完成记录并停止该配置，一起调整后续预算；复现保留原重复、菜单和模型配置。

首次代码交接与等价性抽查记录（H1）的实测（内部 `h1/report.md` 和等价性回执，不随包分发；关键数字已摘入）：

|步骤|墙钟|峰值 RSS|检查范围|
|---|---|---|---|
|从 clean 和 20 表构建|156.9 秒|2.47 GiB|13 张输出表逐值、类型与帧 hash 一致|
|日级参照（A-D5-LGB） r0/f0|146.5 秒|3.11 GiB|61,386 预测最大绝对差 6.94e-18；模型文本逐字节一致|
|编码增强的轻量梯度提升树参照（GSAFE-LGB） r1|15.6 秒|1.34 GiB|6,787 预测完全一致|

A-D5-LGB 的极小差异 **consistent with rounding（与浮点舍入相容）**，不是仅凭差异大小证明原因。H1 没有重跑全部重复；本次修订的检查也不构成完整三模型重训或跨平台验证。建议：长任务另设资源监视；输入构建仅一次，训练需保留授权私有缓存根目录的简写（CACHE）与私有运行输出目录的简写（RUN）的完整绑定。
