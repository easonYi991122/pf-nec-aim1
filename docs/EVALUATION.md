# 评价合同与可执行示例

输出与区间记号：`PF_RUN_DIR` 是授权私有运行输出目录变量；`RUN_ROOT` 是配置解析后的输出根。描述性置信区间（CI）由单臂接口计算，不能替代完整比较族的同时区间。

冻结 `evaluate.py`、harness v3/v2 的数值逻辑不变。单臂入口 `pf_nec.inference.single_arm_intervals` 只组合原评价器的对齐、抽样与计分函数，直接计算单臂描述性区间，无需第二个比较臂。

时间记号：术后日数（POD），手术当天从零计。

|字段|含义／检查|
|---|---|
|repeat, row_key|每次重复的唯一行键，预测须与独立 targets 完整对应|
|stay, case|住院键及病例住院指示；case 不等于当天 y|
|y, pod|当天二元标签及原 POD；不得由预测表反推目标集|
|fold|逐日滚动预测任务（Task A）必须有 0–4 折；每次合并完整折外预测（OOF）后计分|
|probability|预测概率，必须有限且在 [0, 1]|

评价先核对完整行集；内连接会遗漏缺失行，不能代替完整性检查。重复使用的日行仍属同一住院；重复指标等权。平均精确率（AP）/预测概率与二元结局的均方误差（Brier）描述抽样保留日，不是部署校准。

## 单臂描述性区间

先按 [复现指南](REPRODUCE.md) 完成对应模型的所有训练，再从仓库根目录运行。`--descriptive-ci` 使用病例／对照分层的住院并集 bootstrap；可同时加 `--harness`，但两组区间应分别命名。

以下示例中的名称：编码增强的轻量梯度提升树参照（`GSAFE-LGB`）；日级参照（`A-D5-LGB`）.

```sh
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --descriptive-ci
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --harness
```

## 本地清理表评价合同（PI72-CLEAN）两个交接参照的配对评价

保存下段为仓库外的脚本，使用已配置的解释器模块搜索路径环境变量（`PYTHONPATH`）和解释器执行。它比较经过可用时点屏蔽的编码增强特征方案（G-safe）与按日可用时点屏蔽的登记特征库（D5-safe）。**显式 family 只有这个描述性对比**，不是历史完整晋级族；即使局部区间为正也不能宣称晋级。若要评价未来候选对两个参照，必须另行预声明全部对比和 family；不能以本例替代完整历史比较族。复用已接受私有输出时，可将装载行替换为读取独立 targets 和两个预测表的 `pd.read_parquet`；其字段及完整性要求相同，不能从预测表反造 targets。

以下示例中的名称：本地清理表的全列可用时点屏蔽树模型参照（`T8-D5SAFE-LGB`）；术后受试者工作特征曲线下面积字段（`AUROC_post`）.

```python
from pf_nec import cli, config, evaluate as ev, run
import pandas as pd

repeats = [1, 2, 3, 4, 5]
targets, g = cli.load_evaluation_inputs("GSAFE-LGB", repeats)
other_targets, d = cli.load_evaluation_inputs("T8-D5SAFE-LGB", repeats)
pd.testing.assert_frame_equal(targets, other_targets)
family = [{"arm": "GSAFE-LGB", "reference": "T8-D5SAFE-LGB", "metric": "AUROC_post"}]
result = ev.evaluate(targets, {"GSAFE-LGB": g, "T8-D5SAFE-LGB": d},
                     frame="PI72-CLEAN", stage="confirm", contrasts=family, family=family)
assert not result["family_complete"]
assert not result["comparisons"][0]["promotion"]["passes"]
run.write_json(config.writable(config.RUN_ROOT / "reference_pair.json"), result)
```

## 区间算法与输出

|入口|bootstrap|解释|
|---|---|---|
|Task A `--harness`|默认 400 次，住院非分层抽样|冻结 harness 的旧描述口径；不是日级任务历史确认、归因与收尾报告（R11）主表的区间算法|
|`--descriptive-ci` / `ev.evaluate`|2000 次尝试；病例／对照分层；同一住院跨重复、阶段、臂共用抽样次数|固定预测下的百分位描述区间；每项至少 1990 次有效；无效 draw 不重抽|
|显式 family 的同时区间|2000 次中至少 1990 次全族共同有效；最大绝对中心化误差|仅完整冻结比较族才能参与晋级判定；子族永不替代全族|

本地清理表历史确认报告（R9）/R11 主表使用后两行的口径；最小可检测差（MDD）=0.01、5 次完整重复及同时下界>0 的门不变。3 次诊断初筛（screen）从不晋级；5 次复用 screen，也不构成独立确认。

输出到 `PF_RUN_DIR/<model>/evaluation.json`；配对示例到 `PF_RUN_DIR/reference_pair.json`。完整单臂 JavaScript 对象表示法（JSON）另含 `per_repeat`（每次 metrics/counts/decomposition）、全部阶段指标和条件性标志。下面只展示应核对的**汇总字段投影**；其数值来自内部 `r9/report.md`，不随包分发，关键数字已摘入：

以下示例中的名称：总体受试者工作特征曲线下面积字段（`AUROC_total`）.

```json
{
  "model": "GSAFE-LGB",
  "frame": "PI72-CLEAN",
  "repeats": [1, 2, 3, 4, 5],
  "equal_repeat_mean": {"AUROC_post": 0.7329, "AUROC_total": 0.8095},
  "promotion_assessed": false,
  "descriptive_ci": {
    "attempted_draws": 2000,
    "arm_intervals": {
      "AUROC_post": {"estimate": 0.7329, "valid_draws": 2000,
                     "undefined_draws": 0, "percentile_ci95": [0.7020, 0.7622]}
    }
  }
}
```

核对要求：计数、键、重复集合必须精确一致。比较本页 4 位小数与 JSON 时，绝对容差为 0.00005；5 位小数为 0.000005，这只是显示舍入容差。相同输入／版本重复计算应确定性一致；不同平台显示容差不能代替对训练变化的检查，须另核对模型、参数、停止轮数与预测。`null` 是无定义或支持不足，不是 0；不删掉该重复后求均值。配对 JSON 中须有 `comparisons`、`arm_intervals`、`simultaneous_family`、`family_complete=false`、`attempted_draws` 和有效抽样数。

## 阶段分解

PI72-CLEAN 术前均阴性，因此 `AUROC_total = f_pre × cross_stage_AUC + (1−f_pre) × AUROC_post`，其中 f_pre 分母是全部阴性日。R9 的 f_pre 为 0.270–0.301；跨阶段曲线下面积（AUC） **实测**臂均值为 0.999918–1.000000，不能只从全阴性标签推出来。术前阶段内受试者工作特征曲线下面积（AUROC）为 null。

Task A 有术前阳性。R11 的 A-D5-LGB 分解如下（5 次均值）；四项加权贡献还原总体 0.71751。通用 `cross_stage_positive_vs_preop_negative_auc` 字段比较所有阳性与术前阴性，包含术前阳性；不是纯术后阳性那一格。

|正日阶段／负日阶段|权重|AUC|
|---|---|---|
|前／前|0.02653|0.54227|
|前／后|0.07666|0.14612|
|后／前|0.23059|0.96123|
|后／后|0.66622|0.70588|

来源：`r9/report.md`、`r11/report.md`（内部报告，不随包分发；关键数字已摘入）。分解误差检查用完整精度，不能用已舍入的表要求逐位恒等。
