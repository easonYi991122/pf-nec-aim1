# 受限表格与缺失机制建模任务（T-2） / 早期术后顺序信息任务（T-3）：合成接口开工示例

紧缩字段：`fit_pool_sha256` 是拟合训练池身份的 安全散列算法的文件完整性摘要（SHA-256）；`gestagewks` 是孕周；`RUN_ROOT` 是配置解析后的输出根；`validation_auc` 是内层验证受试者工作特征曲线下面积（AUROC）；文件及完整性哈希清单（MANIFEST）用于核对交付文件身份。

如果你想先熟悉接口，可以从下面的合成例子开始；这是我们自己检查单次拟合时使用的步骤。请在仓库根目录使用已配置的解释器。示例只验证单次 fit/apply，不执行完整新任务。源码入口：[带日行身份、预测字段及可用性元数据的输入对象（FeatureRows）](../src/pf_nec/features.py)、[绑定训练池身份与种子的拟合上下文对象（FitContext）](../src/pf_nec/selectors.py)；完整因果性／隔离测试：[test_team.py](../tests/test_team.py)、[test_team_sequence.py](../tests/test_team_sequence.py)、[test_team_m1.py](../tests/test_team_m1.py)。

以下示例中的名称：解释器模块搜索路径环境变量（`PYTHONPATH`）；禁止生成字节码缓存的解释器环境变量（`PYTHONDONTWRITEBYTECODE`）；授权私有运行输出目录环境变量（`PF_RUN_DIR`）.

```sh
PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_team.py tests/test_team_sequence.py tests/test_team_m1.py -p no:cacheprovider --basetemp="$PF_RUN_DIR/team-tests"
```

时间记号：术后日数（POD），手术当天从零计。

|对象／阶段|如何使用与隔离|
|---|---|
|FeatureRows|keys 为住院／日期／精确日行身份，values 为已按历史日门控的预测字段；available、gates、admission 为元数据。只通过 build_features 或受审适配器创建，不混入结局。标签另放以 row 标识符（ID）索引的 Series，不能以住院键或数组位置替代。|
|FitContext / child|FitContext 记录 frame、repeat、fold、inner、pool 并绑定种子。child 指当前内层训练子集或外层重拟合训练池；在当前训练池内按树模型增益排序选列（GAIN）和标准化只能拟合这个训练池。|
|inner|内层按住院分开 train／validation；validation_labels 和 validation_score_mask 都按精确日行 ID 对齐。遮罩决定内层评分行，不把外测标签传进来；T-3 的 POD0–2 适配仍未启用。|
|refit|内层选择完成后，在整个外层训练池重新取 GAIN、拟合标准化和模型。使用已选 epoch；不得接收外测标签或把 held-out 行用于早停。|
|apply|只接收已选列的输入与行身份，不接收标签；历史神经模型选择程序组（M1）还要求从 refit package 取 preprocessing 与 fit_pool_sha256，不能在 apply 行重拟合。预测顺序对应 package 内 row_key。|

下面对序列门控循环单元（GRU）与 M1 带缺失衰减处理的门控循环单元（GRU-D）各跑一次合成链。以 `PYTHONPATH=src:tests` 执行该 Python 段；`sample`、`extract` 来自随包测试夹具。固定的 2 个 epoch 只为 smoke，不是内层选择结果；临时产物写入 PF_RUN_DIR 后清理。

以下示例中的名称：轻量梯度提升机（`LightGBM`）；本地清理表评价合同（`PI72-CLEAN`）；带缺失衰减的窗口序列学习器配置键（`GRUD-WINDOW`）；训练池内增益排序的前五十列（`GAIN50`）.

```python
import tempfile
from pathlib import Path
import torch  # Import before LightGBM on Mac.
import pandas as pd
from pf_nec import config, selectors
from explore.team import inputs, sequence, sequence_m1
from test_gsafe import sample
from test_team import extract

config.initialize()
train, meta = sample(120)  # Entirely synthetic FeatureRows and metadata.
train.values.loc[:, "gestagewks"] = 30 + 6 * meta.case.to_numpy()
labels = pd.Series(meta.y.to_numpy(), index=meta.row_key)
held, _ = sample(12, 100)
context = selectors.FitContext(frame="PI72-CLEAN", repeat=1, pool="outer")
for runner, learner in ((sequence, "GRU"), (sequence_m1, "GRUD-WINDOW")):
    with tempfile.TemporaryDirectory(dir=config.RUN_ROOT, prefix="team-smoke-") as folder:
        out = Path(folder)
        rows, selected = inputs.gain50(train, labels, context)
        held_rows = inputs.selected_rows(held, selected)
        common = dict(frame="PI72-CLEAN", repeat=1, recipe="GAIN50",
                      window=7, learner=learner, synthetic=True)
        fit_cfg = runner.make_config(**common, stage="refit", chosen_epochs=2)
        packed = runner.build_fit_package(out/"fit.zip", rows, labels, fit_cfg)
        fit_dir = extract(packed["path"], out/"fit")
        receipt = runner.fit_package(fit_dir, out/"models", device="cpu", synthetic_smoke=True)
        state = {} if runner is sequence else dict(
            preprocessing=receipt["preprocessing"], fit_pool_sha256=receipt["config"]["fit_pool_sha256"])
        apply_cfg = runner.make_config(**common, stage="apply")
        packed = runner.build_apply_package(out/"apply.zip", held_rows, apply_cfg, **state)
        apply_dir = extract(packed["path"], out/"apply")
        probability = runner.apply_package(apply_dir, out/"models", out/"predictions.npz", device="cpu")
        assert probability.shape == (len(held_rows.keys),)
```

## 配置决策表

|项目|已给定|可以一起确定的配置|
|---|---|---|
|T-2 菜单与内层汇总|GAIN50-三日日历窗口（w3）/七日日历窗口（w7）；summarize_inner 对 3 折的 validation_auc 等权平均，并取 3×3 个 best_epoch 的整数中位数供 refit|新程序的选窗排序、并列容差／平局规则须写入卡的配置附录；未冻结前不得真实选窗，不默认继承历史完整 M1 菜单|
|T-2 参照|随包本地清理表的全列可用时点屏蔽树模型参照（T8-D5SAFE-LGB）、编码增强的轻量梯度提升树参照（GSAFE-LGB）及冻结配置可定位于最终 spec|新比较驱动须绑定相同行／标签和共同记录的固定配置，合成例子不能据以判断临床适用性|
|T-3 输入与对照|训练池内增益选列加七日日历窗口的配置（GAIN50-w7），时间卷积网络（TCN）／GRU 对无序多层感知机（MLP）、窗口树和仅时钟树|POD0–2 行集／评分／回执适配、使用滞后展开窗口的树模型（R1-LGB）与仅时钟轻量梯度提升机学习器代号（LGB）的精确新任务配置、完整联合比较族须另冻；不能宣称现接口已执行 T-3|
|扩展位置|在 src/explore/team 旁新增独立身份的驱动／适配器及合成测试|改现有 learner、基础 runner 或 team spec，邀请伙伴共同复核，再同步版本／绑定哈希、重建 MANIFEST 并重测，保证结果可比；不改最终模型 spec 或 harness|

共享参数的小型表格模型集成（TabM）对应 sequence_m1 的表格共享参数模型的滞后展开配置键（TABM-R1），调用链和确切依赖检查见 test_team_m1.py；其库版本不可替换。完整历史重建特征库名称（D5）神经路径、任务总调度、POD0–2 新评价和告警驱动仍 not shipped。研究问题与运行安排共同讨论，大家使用获授权主机；详细门与预算见 [TEAM_TASKS](TEAM_TASKS.md)。

## 探索规格的门值范围

[探索规格](../src/explore/team/spec.json)中的 false 门值及字段名保持交接版本原样。原构建器固定写入这些值，随包测试只检查不自动授权真实运行；没有定义逐运行状态初始化或更新的机制，因此本包不能认证它们是“每个新运行的默认值”。这些静态门值不能替代[当前状态](STATUS_AND_NEXT.md)。每次真实运行仍须事先记录任务卡要求的审查、配置冻结、输入时点核查和数据授权；研究附录冻结不等于临床阈值获认可。
