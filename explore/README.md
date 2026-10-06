# 探索性分析与脚手架

〔事实〕`src/explore/` 与最终模型共享冻结接口，但其解释、诊断和因果草案不是性能候选“晋级／未晋级”的同一种判断。未来分析须另定方案；结构搜索的暂停与重启条件见 [STATUS_AND_NEXT](../docs/STATUS_AND_NEXT.md)。

|入口|支持范围|需要另做的工作|
|---|---|---|
|T3 `explore.t3.interpret` / `t3_fast`|分组 SHAP、时间归因、轨迹统计；`gsafe_t3.packets(engine, output)` 从保存模型核对固定预测并产生内存 packet|Task A 先组装完整各折、各重复；lead 需要独立核对的 event_day，packet 里的 day 不能替代事件日；仅保存汇总|
|I7 `explore.i7.diagnostics`|8 背景变量、B+C / B+S / B+C+S 切片，窗口 1 或 7；learning_curve_stays、learning_slopes 保留方法|原 PI-29 曲线不能重跑；新模型曲线另冻结；B/C/S 见 GLOSSARY|
|Aim 1b `stage12` / `stage34`|测量、资格、positivity、卡片证据合并、试验与 DAG 草案、讨论排序|准备度、时间零、动作含义、未测混杂与效应协议待 PI 确认；没有效应估计|

I7 可执行入口：`python -m explore.i7.diagnostics --repeat 1 --arm B-only-w1`，需要先 build-data。结果回执在 `PF_RUN_DIR/explore-i7/<context>/`。完整 3 次 screen 须分别运行；该命令本身只训练并保存预测，不生成已接受报告或晋级结论。评价可用 `pf_nec.inference.single_arm_intervals`，targets 必须从 slice_provider 的独立 test 元数据取得。

TCN／GRU、无序对照和 M1/GPU 训练代码不提供；已试结果见 RESULTS。〔建议〕结构类搜索暂停；仅在用途讨论改变主指标，或新数据／信息源与预先冻结的假设足以支持新研究时，才考虑重启。解释性重要性不能成为重启依据；条件同 [STATUS_AND_NEXT](../docs/STATUS_AND_NEXT.md)。

## Aim 1b 合成演示

〔事实〕以下两个 CLI 仅输出合成汇总。

```sh
python -m explore.aim1b.stage12 --synthetic
python -m explore.aim1b.stage34 --synthetic
```

下段可直接在已配置解释器中运行，完整调用卡片、试验、DAG 与排序入口。标题和协议字段都是**合成占位值**；空模型证据表示未提供，不表示重要性为零。真实卡片必须由获准方案提供。

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
|top20 / T6 贡献|feature、mean_abs、normalized_share、selection_frequency、repeat_rank_range、top10_frequency、top20_frequency、per_repeat、included_in_any_model；未汇总项保留 null|
|T6_candidates 与 t6_csv|candidate_id 一一对应；features、contribution_and_stability、POD_lead_profile、trajectory_support 在 CSV 中为 JSON 字符串，须与 summary 逐值一致|
|trajectories|feature、alignment、day；eligible_rows、nonmissing、unknown_codes、nan_rows、unavailable_rows、adjacent_calendar_pairs、available_pairs、missingness_switches；只合并测量分母|
|permutation|model、group、features（JSON 字符串）、mean_auc_loss、uncertainty（含 ci95 的 JSON 字符串）、status|
|diagnostics|run_diagnostics 返回对象的 diagnostics 列表；rank_cards 只读，不按结局效应优化排序|

只把聚合证据传给 stage3_cards；不要传患者行。trial_skeleton 和 draft_dag 只对 stage4_included 卡调用（9 个问题），rank_cards 接受完整 13 卡。原历史 R10 合并报告不随包分发；同 schema 的独立安全模型汇总可使用这些函数，缺失归因不补零。

〔事实〕`S_RISK` 与 `RANK_REASONS` 保留原中文字符串；含义摘要：胸骨关闭因终点可测、已测重叠较好而优先讨论，外周动脉线因残余不平衡仅作条件性备选，脐动脉线保留，通气及手术室拔管后置，药物当前只支持快照对照；所有候选均有高或很高的严重度代理风险，包括未测准备度、指征和治疗响应，因此排序不确立可交换性或治疗效应。

## 授权 raw 表的适配

〔建议〕先完成合成测试，再在私有环境中运行下面的测量／重叠示例。它读取原始表，经过同一日期解析，展示固定 POD3、1 日宽限；不是正式效应分析。它会拟合治疗代理倾向模型，**不训练 NEC 结局模型**。外层资源监视与暂停规则同 AGENTS。

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
