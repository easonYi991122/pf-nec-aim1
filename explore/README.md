# 探索性分析与脚手架

这里用术后日数（POD）表示从手术当天零起算的日数，接口字段 `pod`／`pods` 沿用该定义。置信区间（CI）、数值缺失标记（NaN）与曲线下面积（AUC，此处指受试者工作特征曲线下面积）分别用于不确定性、缺失与区分度输出。`RiskSurgVIS` 是血管活性—正性肌力评分（VIS）原表。授权私有运行目录变量为 `PF_RUN_DIR`；仅背景／时钟特征的单日模型记为 B-only-w1。

归因字段：`rank_ci_support_ok` 表示满足排名区间的支持量要求；`nan_rows` 是数值缺失行数；`mean_auc_loss` 是置换后的平均区分度损失；`ci95` 为 95% 区间。输入结构见本页“卡片与汇总证据 schema”；这些是模型依赖证据，不是治疗效应。

`src/explore/` 与最终模型共享冻结接口，但其解释、诊断和因果草案不是性能候选“晋级／未晋级”的同一种判断。未来分析须另定方案；结构搜索的暂停与重启条件见 [STATUS_AND_NEXT](../docs/STATUS_AND_NEXT.md)。

特征切片记号：背景与时钟（B）、累计历史（C）、当日状态（S）。

|入口|支持范围|需要另做的工作|
|---|---|---|
|历史归因模块（T3） `explore.t3.interpret` / `t3_fast`|分组 Shapley 加性解释（SHAP）、时间归因、轨迹统计；`gsafe_t3.packets(engine, output)` 从保存模型核对固定预测并产生内存 packet|逐日滚动预测任务（Task A）先组装完整各折、各重复；lead 需要独立核对的 event_day，packet 里的 day 不能替代事件日；仅保存汇总|
|历史切片、窗口、序列与学习曲线诊断程序（I7） `explore.i7.diagnostics`|8 背景变量、B+C / B+S / B+C+S 切片，窗口 1 或 7；learning_curve_stays、learning_slopes 保留方法|原未随包提供的来源模型特征家族（PI-29）曲线不能重跑；新模型曲线另冻结；B/C/S 见 GLOSSARY|
|第一研究目标中的目标试验模拟部分（Aim 1b） `stage12` / `stage34`|测量、资格、给定条件下两种治疗策略都有实际支持（positivity）、卡片证据合并、试验与有向无环图（DAG）草案、讨论排序|准备度、时间零、动作含义、未测混杂与效应协议待首席研究者（PI）确认；没有效应估计|
|team `explore.team.inputs` / `sequence` / `sequence_m1`|按日可用时点屏蔽的登记特征库（D5-safe）内训练池内选出的前五十列（GAIN50）；新身份的单次拟合及无标签应用|完整驱动、早期术后顺序信息任务（T-3）行集／回执、联合族和告警适配待冻结；见 [TEAM_USAGE](../docs/TEAM_USAGE.md)|


I7 可执行入口：`python -m explore.i7.diagnostics --repeat 1 --arm B-only-w1`，需要先 build-data。结果回执在 `PF_RUN_DIR/explore-i7/<context>/`。完整 3 次诊断初筛（screen）须分别运行；该命令本身只训练并保存预测，不生成已接受报告或晋级结论。评价可用 `pf_nec.inference.single_arm_intervals`，targets 必须从 slice_provider 的独立 test 元数据取得。

历史完整序列／历史神经模型选择程序组（M1）菜单及调度不随包；`src/explore/team` 已提供新身份的 GAIN50 单次 fit/apply 接口，包含时间卷积网络（TCN）／门控循环单元（GRU）、无序对照及 M1 学习器。完整新任务驱动仍未提供，边界和合成使用见 [TEAM_TASKS](../docs/TEAM_TASKS.md)。主线暂停和冻结卡下受限表格与缺失机制建模任务（T-2）/T-3 研发遵循 [STATUS_AND_NEXT](../docs/STATUS_AND_NEXT.md)；不以重要性重开搜索。

## Aim 1b 合成演示

以下两个命令行接口（CLI）仅输出合成汇总。

```sh
python -m explore.aim1b.stage12 --synthetic
python -m explore.aim1b.stage34 --synthetic
```

下段可直接在已配置解释器中运行，完整调用卡片、试验、DAG 与排序入口。标题和协议字段都是**合成占位值**；空模型证据表示未提供，不表示重要性为零。真实卡片必须由获准方案提供。

以下示例中的名称：因果讨论卡的固定排序键（`RANK_ORDER`）；历史行动候选的汇总证据字段（`T6_candidates`）.

```python
from explore.aim1b import stage34 as s
cards = [dict(id=cid, title="Synthetic " + cid, rating="maybe",
              time_zero="Synthetic decision", eligibility="Synthetic eligibility",
              rescue="Synthetic rescue", additional_confounds="Synthetic confounders")
         for cid in s.RANK_ORDER]
summary = {"repeats": [1, 2, 3, 4, 5], "T6_candidates": [], "models": {}}
enriched = s.stage3_cards(cards, summary, [], [], [])
diagnostics = s.run_diagnostics(s.synthetic_tables(120), pods=(3,), graces=(1,))
ranked = s.rank_cards(enriched, diagnostics["diagnostics"])
trials = [s.trial_skeleton(c) for c in enriched if c["stage4_included"]]
dags = [s.draft_dag(c) for c in enriched if c["stage4_included"]]
assert len(ranked) == 13 and len(trials) == len(dags) == 9
```

## 卡片与汇总证据 schema

|输入|必须提供的结构|
|---|---|
|cards|13 个不同 id，覆盖 RANK_ORDER；每卡包含 title、rating、time_zero、eligibility、rescue、additional_confounds。stage3 保留额外原字段|
|summary|repeats 必须为 [1,2,3,4,5]；models 字典与 T6_candidates 列表；空列表只表示没有证据|
|models 每项|overall.post.top20；可选 lag_profile；pod 和 lead 的各层含 top20、rank_ci_support_ok|
|top20 / 历史行动候选线索卡编号（T6）贡献|feature、mean_abs、normalized_share、selection_frequency、repeat_rank_range、top10_frequency、top20_frequency、per_repeat、included_in_any_model；未汇总项保留 null|
|T6_candidates 与 t6_csv|candidate_id 一一对应；features、contribution_and_stability、按术后日数及事件提前量分层的归因字段（POD_lead_profile）、trajectory_support 在逗号分隔值文件（CSV）中为 JavaScript 对象表示法（JSON）字符串，须与 summary 逐值一致|
|trajectories|feature、alignment、day；eligible_rows、nonmissing、unknown_codes、nan_rows、unavailable_rows、adjacent_calendar_pairs、available_pairs、missingness_switches；只合并测量分母|
|permutation|model、group、features（JSON 字符串）、mean_auc_loss、uncertainty（含 ci95 的 JSON 字符串）、status|
|diagnostics|run_diagnostics 返回对象的 diagnostics 列表；rank_cards 只读，不按结局效应优化排序|

只把聚合证据传给 stage3_cards；不要传患者行。trial_skeleton 和 draft_dag 只对 stage4_included 卡调用（9 个问题），rank_cards 接受完整 13 卡。原历史本地清理表历史归因报告（R10）合并报告不随包分发；同 schema 的独立安全模型汇总可使用这些函数，缺失归因不补零。

严重度代理风险的讨论字段（`S_RISK`）与因果讨论排序的理由字段（`RANK_REASONS`）保留原中文字符串；含义摘要：胸骨关闭因终点可测、已测重叠较好而优先讨论，外周动脉线因残余不平衡仅作条件性备选，脐动脉线保留，通气及手术室拔管后置，药物当前只支持快照对照；所有候选均有高或很高的严重度代理风险，包括未测准备度、指征和治疗响应，因此排序不确立可交换性或治疗效应。

## 授权 raw 表的适配

时间记号：术后日数（POD），手术当天从零计。

建议：先完成合成测试，再在私有环境中运行下面的测量／重叠示例。它读取原始表，经过同一日期解析，展示固定 POD3、1 日宽限；不是正式效应分析。它会拟合治疗代理倾向模型，**不训练坏死性小肠结肠炎（NEC）结局模型**。外层资源监视与暂停规则同 AGENTS。

```python
from pf_nec import config, contract, run
from explore.aim1b import stage12 as a1, stage34 as s
config.require_data()
config.initialize()
contract.reset_threads()
tables = {name: a1.read_table(name)[0] for name in
          ("IndexSurgHosp", "MechVent", "Sternum", "ArterialLine", "RiskSurgVIS", "NEC")}
tables["PreopRiskFactor"] = s.read_plain_table("PreopRiskFactor")[0]
result = s.run_diagnostics(tables, pods=(3,), graces=(1,))
run.write_json(config.writable(config.RUN_ROOT / "aim1b_diagnostics.json"), result)
```
