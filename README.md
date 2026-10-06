# Project Passion Fruit：NEC 研究代码交接

English: [README.en.md](README.en.md)。双语约定：`docs/<NAME>.md` 为中文，`docs/en/<NAME>.md` 为对应英文；根目录 README／AGENTS 与 explore/README 的英文版使用 `.en.md`。两种语言承载相同事实和数字，没有优先语言；构建测试检查数字 token 一致。

〔事实〕本包包含最终模型代码、冻结规则、合成测试、汇总结果和探索性脚手架。它不含数据、权重或预测。使用者须已经获同一 PC4 数据的 DUA／IRB 授权。这里是回顾性开发集内部评价材料，尚无独立或外部验证。缩写先查 [GLOSSARY](docs/GLOSSARY.md)。

|合同|最终交接模型|主指标|重复／外层|
|---|---|---|---|
|PI72-CLEAN 本地 clean-table 合同|GSAFE-LGB（629 列）；T8-D5SAFE-LGB（604 列）|术后 POD≥0 AUROC|r1–5，每次固定住院外测|
|Task A formal v2.6 / harness v3|A-D5-LGB（604 列，本轮参照）|总体 AUROC；术后为关键次指标|r0–4，每次 5 折住院 OOF|

合同的行、标签和验证方案不同，AUROC 不跨合同比较。PI72-CLEAN 是本地重建，不是论文 0.79 的最终计分设计；该设计尚未核实。A-D5-LGB 不替换历史冠军；Task B 本轮未重训。

## 我们停在哪里

〔推断〕模型用途尚未定义：谁在何时看到风险、随后改变什么？〔建议〕优先做 A1 用途对应评价、A2 简单术时评分与日级模型的正式比较、A3 留出整个中心的验证。结构类搜索暂停；用途改变主指标或新信息支持预先冻结的新问题后再考虑重启。详见 [STATUS_AND_NEXT](docs/STATUS_AND_NEXT.md)。

〔建议〕Aim 1b 优先讨论胸骨关闭时机，外周动脉线拔除为条件性备选；待 PI 回答 7 问后才冻结。Aim 2 本轮无新工作；B2 阶段混合评价的方法学线索保留。

〔事实〕请分开读三个集合：**最终模型**见上表；**未达到预设晋级门槛的候选**（T8、T4、M1、Task A 程序）见 [RESULTS](docs/RESULTS.md) 的效应量和区间；**探索性分析与脚手架**（T3、I7、Aim 1b）见 [explore](explore/README.md)。重要性和改动输入后的分数不是治疗效应。

〔事实〕日阳性率必须按合同和阶段解释：Task A 约 0.35%；PI72-CLEAN 外测总体约 1.0%、术后约 1.4–1.5%。PI72-CLEAN（本地 clean-table）的术前全阴性只给出分解恒等式；跨阶段 AUC≈1 是该合同的实测结果，不是标签构成的数学必然，也不归属于论文设计。Task A 的 A-D5-LGB 对应字段约为 0.92，但比较的是全部阳性（含术前阳性）与术前阴性，不能当成纯术后阳性那一格；四格定义见 [EVALUATION](docs/EVALUATION.md)，不得据此跨合同比较性能。来源：`r9/report.md`、`r11/report.md`（内部报告，不随包分发；关键数字见 RESULTS）。

## 从安装到复现

〔建议〕先按 [ENVIRONMENT](docs/ENVIRONMENT.md) 从空 venv 安装 Mac／Linux 环境，再在仓库根目录执行下列命令；`python` 必须指向该环境。完整 20 张原始表、clean 列序、队列流程与输出树见 [DATA_LAYOUT](docs/DATA_LAYOUT.md)。只写授权私有根目录。

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

顺序运行，每进程 2 数值线程、RSS≤8 GiB；超限停止，不改配置。缓存只构建一次；续跑须保持输入、版本、绝对路径和成对 CACHE/RUN，不能只移动 RUN。单臂描述性 CI、两个参照的显式 family 配对示例、400 次 harness 与 2000 次分层 bootstrap 的差别、JSON 形状和容差见 [EVALUATION](docs/EVALUATION.md)。

## 核查范围与阅读顺序

〔事实〕历史 H1 从输入重建的 13 张表一致；A-D5-LGB r0/f0 与 GSAFE-LGB r1 做过真实等价性抽查。A 极小预测差与舍入相容（consistent with rounding）；详见 ENVIRONMENT。这些抽查不代表完整重复、Linux 或 Windows 已重跑，也不构成独立验证。合成测试覆盖三条模型路径、因果时序不变性、评价和卡片接口。

先读 [TASKS](docs/TASKS.md) 和 [RESULTS](docs/RESULTS.md)，再读 STATUS_AND_NEXT。共享排除范围见 [EXCLUDED](docs/EXCLUDED.md)，书目和内部来源解释见 [REFERENCES](docs/REFERENCES.md)。MANIFEST 的 source_path 仅为溯源，不是运行依赖。人员与编码 agent 都遵守 [AGENTS](AGENTS.md)／[AGENTS.en](AGENTS.en.md)；CLAUDE 指向同一规则。
