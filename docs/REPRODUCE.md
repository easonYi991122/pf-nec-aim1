# 一起复现：我们使用的步骤

以下使用 Python 虚拟环境（venv）；`r0/f0` 表示重复 0／外折 0。目录简称 CACHE 与 RUN 分别指授权私有缓存根和运行输出根；两者须成对保留。

以下是我们自己复现时用的步骤。可以先做无需患者数据的合成测试，熟悉接口后再一起核对真实数据授权、输入版本和共享评价合同。仓库维护者可以帮助定位已有配置；研究决定由大家共同讨论。

我们都需要遵守数据使用协议（DUA）、伦理审查委员会（IRB）要求及首席研究者（PI）的数据协议；患者数据、权重和预测留在授权私有目录，未发表代码的共享边界见 [EXCLUDED](EXCLUDED.md)。

## 先选要复现的合同

时间记号：术后日数（POD），手术当天从零计。

|合同|最终交接模型|主指标|重复／外层|
|---|---|---|---|
|本地清理表合同（PI72-CLEAN）|编码增强的轻量梯度提升树参照（GSAFE-LGB）（629 列）；本地清理表的全列可用时点屏蔽树模型参照（T8-D5SAFE-LGB）（604 列）|术后 POD≥0 受试者工作特征曲线下面积（AUROC）|本地合同的一起始重复编号（r1–5），每次固定住院外测|
|逐日滚动预测任务（Task A）按未来结局筛选的开发队列（formal） v2.6 / harness v3|日级参照（A-D5-LGB）（604 列，本轮参照）|总体 AUROC；术后为关键次指标|日级任务的零起始重复编号（r0–4），每次 5 折住院折外预测（OOF）|

## 配置环境与运行

先按 [ENVIRONMENT](ENVIRONMENT.md) 从空的 Python 虚拟环境（venv） 安装 Mac／Linux 环境，再在仓库根目录执行下列命令；`python` 必须指向该环境。完整 20 张原始表、clean 列序、队列流程与输出树见 [DATA_LAYOUT](DATA_LAYOUT.md)。只写授权私有根目录。

以下示例中的名称：解释器模块搜索路径环境变量（`PYTHONPATH`）；禁止生成字节码缓存的解释器环境变量（`PYTHONDONTWRITEBYTECODE`）；授权只读数据根目录环境变量（`PF_DATA_ROOT`）；授权私有缓存根目录环境变量（`PF_CACHE_ROOT`）；授权私有运行输出目录环境变量（`PF_RUN_DIR`）.

```sh
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export PF_DATA_ROOT=/authorized/pc4
export PF_CACHE_ROOT=/private/pf-nec-cache
export PF_RUN_DIR=/private/pf-nec-cache/runs
python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
python -m pf_nec.verify
python -m pf_nec.cli build-data
for r in 1 2 3 4 5; do
  python -m pf_nec.cli train --model GSAFE-LGB --repeat "$r"
  python -m pf_nec.cli train --model T8-D5SAFE-LGB --repeat "$r"
done
for r in 0 1 2 3 4; do
  for f in 0 1 2 3 4; do
    python -m pf_nec.cli train --model A-D5-LGB --repeat "$r" --fold "$f"
  done
done
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model T8-D5SAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --descriptive-ci
```

为便于比较，我们顺序运行，每进程 2 数值线程、常驻内存（RSS）≤8 二进制吉字节（GiB）；超限时保留记录并一起检查原因。缓存只构建一次；续跑须保持输入、版本、绝对路径和成对 CACHE/RUN，不能只移动 RUN。单臂描述性置信区间（CI）、两个参照的显式 family 配对示例、400 次 harness 与 2000 次分层 bootstrap 的差别、JavaScript 对象表示法（JSON）形状和容差见 [EVALUATION](EVALUATION.md)。

## 已核对的范围

首次代码交接时从输入重建的 13 张表一致；A-D5-LGB r0/f0 与 GSAFE-LGB r1 做过真实等价性抽查。A 极小预测差与舍入相容（consistent with rounding）；详见 [ENVIRONMENT](ENVIRONMENT.md)。这些抽查不代表完整重复、Linux 或 Windows 已重跑，也不构成独立验证。合成测试覆盖三条模型路径、因果时序不变性、评价和卡片接口。

这些上限让各次运行使用相同资源约定，也避免内存不足产生不完整结果。若预计超限，可以一起调整新研究的版本和预算；正在复现的合同保留原设置。固定菜单与比较族能减少看过结果后选择分析路径的偏差，既有拆分也不会因换种子而重新成为未触碰的留出集。

想接着做新工作，可看 [TEAM_TASKS](TEAM_TASKS.md) 和 [TEAM_USAGE](TEAM_USAGE.md)。
