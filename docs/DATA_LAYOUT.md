# 输入、队列流程、输出与续跑

〔事实〕`PF_DATA_ROOT` 指向同一授权只读输入版本。必须有下列 20 张 raw 表和 1 张 clean 表；不把额外 CSV 放入 raw 目录。保持原文件名和字段名大小写、编码、日期文本、缺失标记及数值精度；不自行把日期或住院键转成整数。raw 读取器使用 UTF-8 或回退 cp1252。

```text
PF_DATA_ROOT/
  NEC Cleaned Data.csv
  Raw CSV Files/
    AllOperations.csv
    Arrest.csv
    ArterialLine.csv
    CABSI.csv
    Catheterizations.csv
    Complications.csv
    DSSI.csv
    ECMO.csv
    IndexSurgHosp.csv
    IntracardLine.csv
    MechVent.csv
    NEC.csv
    PreopRiskFactor.csv
    Procedures.csv
    RiskSurgVIS.csv
    Sternum.csv
    Stroke.csv
    Surgdiag.csv
    Therapies.csv
    UTI.csv
```

## 列序和输入识别

〔事实〕clean **必须保留原始列序**。`legacy/rebuild_v26.py` 按 `cols[25:58]` 与 `cols[109:124]` 取列；不能按字母重排或删掉看似无用的列。关键连接字段包括 raw 的 `hospitalizationidNEW`、`cardsurgdtSHIFT`，clean 的 `hospitalizationidnew`、`cardsurgdtshift_index`、`day`、`outcome_3d`、`necbelldtshift`。完整 header 与代码映射从本地 clean 重建；临床字典及数据派生 crosswalk 不随包分发。

`python -m pf_nec.cli build-data` 在全新缓存执行；它记录每个输入 SHA-256、核对重建列名／顺序与人群，并在完成后重新核对输入哈希。保留私有 `core/manifest.json`，用它比较输入版本；文件名相同不保证数据版本相同。源码定位：`src/pf_nec/data.py:build_frames`、`src/pf_nec/legacy/rebuild_v26.py:Rebuilder`。

## 队列流程（人数是住院数，不是独立患儿数）

〔事实〕流程存在两个分支，PI72-CLEAN 不能从 formal 内连接取得。

|步骤／分支|已接受计数|规则与来源|
|---|---|---|
|raw 主表|11,938 住院，61 中心|完整源队列；`a2/report.md`|
|clean 原表|320,143 行，11,931 住院|原样输入；`01_review/数据全貌与分组说明.md`；H1 等价性回执|
|raw → backing cache|462,721 行；v3 划分表 11,907 住院|重建后按首次 NEC／出院前过滤；保留 formal 及辅助集；H1 等价性回执、harness v3|
|backing → Task A formal|301,468 日，11,674 住院，359 病例，1,066 阳性日|clean-linked a/b/none、仍在险、原 POD≤31；`r11/report.md`|
|完整 raw 日历 + clean → PI72-CLEAN|318,992 有标签日，11,931 住院，356 病例／阳性日|核对 clean 日期和行键、去缺失标签；不与 formal 相交；`ROUND_REDO_summary_zh.md`|
|PI72-CLEAN 每重复抽样|训练 1,116 住院；外测 279 住院|训练病例／对照 285/831，外测 71/208；`ROUND_REDO_summary_zh.md`、`r9/report.md`|

以上来源均为内部报告／回执，不随包分发，关键数字已摘入。backing 包含辅助行，因此 11,907 不是 formal 的 11,674；两条分支也不是逐级相减的单一漏斗。详细组别见 [GLOSSARY](GLOSSARY.md)。

〔事实〕排除规则与“各原因排除多少住院”须分开：clean 比 raw 少 7 次住院；原盘点给出其组成：NEC 表无记录 4、原晚期组（POD≥31）2、NEC 日期早于入院 1。该旧分组不等同于本包的 formal 分组；组成也不说明为何从 clean 中移除，已接受清表概况未记载实际剔除原因，**本包来源未记录原因（reason not documented in shipped sources）**，不能推定全部因 NEC 或缺失值被排除。backing 的随包 `legacy/build_v26.py:assemble_rows` 先剔除首次 NEC 当日及以后、出院当日及以后的行，再仅保留 clean-linked 的 formal 或辅助集合；未关联 clean、err 及不满足任一集合条件的行被丢弃。`harness_v3.make_splits` 只对保留下来的住院生成划分。因此 11,907 来自 backing，不是简单从 clean 的 11,931 直接删除某个已知组；这两个住院数之间逐原因计数**未在本包来源记录（reason not documented in shipped sources）**。backing 到 formal 再排除 c/d 辅助人群及 formal 窗口以外的行；PI72-CLEAN 另按 clean 保留日与非缺失标签构建。来源：`01_review/数据全貌与分组说明.md`（内部报告，不随包分发；关键计数已摘入）及上述随包代码；未为解释差额新增患者级推导。

## 私有输出树

〔事实〕以下是目录模式，N 表示重复，F 表示折。两个根目录都不得位于原始数据目录中；实际路径由 `config.py` 解析。G-safe 权重和预测位于 CACHE，不能只保留 RUN。

```text
PF_CACHE_ROOT/
  reconstruction_mappings.json
  v26_rows.parquet
  splits_v3.parquet
  core/
    manifest.json
    rows.parquet, stays.parquet, source_metadata.parquet
    D5.parquet, S.parquet, raw_codes.parquet
    split_01.parquet ... split_05.parquet
  gsafe/rN/gsafe/
    predictions.parquet
    outer/model.txt, outer/result.json, outer/encoding_full.json
    innerN/ ...
  tmp/, mpl/
PF_RUN_DIR/
  versions.json
  GSAFE-LGB/rN-f-1.json
  T8-D5SAFE-LGB/rN-f-1.json
  A-D5-LGB/rN-fF.json
  <model>/<context>/fit_ledger.json, jobs/ ...
  <model>/evaluation.json
  reference_pair.json
```

〔建议〕输入和软件版本不变、CACHE 与 RUN 成对且仍在原绝对路径时，可重发原训练命令续跑。保留 models、encoding、predictions、jobs 回执和 fit_ledger；检查点验证训练绑定与输出存在性，不是任意缓存修复器。不要将旧 G-safe CACHE 与空 RUN 拼接；旧缓存会返回检查点，而新 ledger 无法代表原拟合历史。已有 `core/manifest.json` 时构建器拒绝重建；中断的 `.partial` 文件不构成成功缓存，也没有批次续建保证，需新的 CACHE。

〔建议〕迁移机器／路径或升级包时建立新成对 CACHE/RUN，重新运行并核对；旧绝对路径不能自动重定位。不要清理 CACHE 后仅保留 RUN，也不要跳过版本冻结文件。输入映射有歧义或人群变化时停止，不通过删行或改键“修复”。实际运行时先验收环境和合成测试。
