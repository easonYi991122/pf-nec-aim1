# 输入、队列流程、输出与续跑

文件用途：`IndexSurgHosp` 为索引手术住院主表；`IntracardLine` 为心内导管记录；`MechVent` 为机械通气记录；`PreopRiskFactor` 为术前风险因素；`RiskSurgVIS` 为血管活性—正性肌力评分（VIS）记录；`Surgdiag` 为手术诊断。名称保持授权原表拼写，临床定义与记录时点以授权字典为准。

连接字段：`hospitalizationidNEW`（raw）与 `hospitalizationidnew`（clean）是住院键；构建器按原键核对，不能视作跨住院患者标识。`cardsurgdtSHIFT` 是源手术日期文本，`cardsurgdtshift_index` 是 clean 索引手术的日期序数，`necbelldtshift` 是 clean 首次事件日期序数。随包 `legacy/rebuild_v26.py` 和 `data.py` 给出解析、索引手术匹配和日期对齐；SHIFT/NEW 字样不证明真实日期、移位量或新的连接关系，须核对授权字典及私有映射。

组别记号：a＝术后第 4–31 天首发；b＝手术当天至术后第 3 天首发；c＝术前首发；d＝术后第 31 天以后首发；none＝无已知首发日期，不等于证实无病；err＝事件日早于入院的错误日期组。当前开发队列采用 a/b/none，详见[术语表](GLOSSARY.md)。

授权只读数据根目录环境变量（`PF_DATA_ROOT`）指向同一授权只读输入版本。必须有下列 20 张 raw 表和 1 张 clean 表；不把额外逗号分隔值文件（CSV）放入 raw 目录。保持原文件名和字段名大小写、编码、日期文本、缺失标记及数值精度；不自行把日期或住院键转成整数。raw 读取器使用 Unicode 文本的可变长度编码（UTF-8）或回退 Windows-1252 文本编码（cp1252）。

以下示例中的名称：坏死性小肠结肠炎（`NEC`）；导管相关血流感染表（`CABSI`）；深部手术部位感染表（`DSSI`）；体外膜肺氧合（`ECMO`）；尿路感染表（`UTI`）.

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

clean **必须保留原始列序**。`legacy/rebuild_v26.py` 按 `cols[25:58]` 与 `cols[109:124]` 取列；不能按字母重排或删掉看似无用的列。关键连接字段包括 raw 的 `hospitalizationidNEW`、`cardsurgdtSHIFT`，clean 的 `hospitalizationidnew`、`cardsurgdtshift_index`、`day`、本地清理表的单阳性日标签（`outcome_3d`）、`necbelldtshift`。完整 header 与代码映射从本地 clean 重建；临床字典及数据派生 crosswalk 不随包分发。

`python -m pf_nec.cli build-data` 在全新缓存执行；它记录每个输入安全散列算法的文件完整性摘要（SHA-256）、核对重建列名／顺序与人群，并在完成后重新核对输入哈希。保留私有 `core/manifest.json`，用它比较输入版本；文件名相同不保证数据版本相同。源码定位：`src/pf_nec/data.py:build_frames`、`src/pf_nec/legacy/rebuild_v26.py:Rebuilder`。

## 队列流程（人数是住院数，不是独立患儿数）

流程存在两个分支，本地清理表评价合同（PI72-CLEAN）不能从按未来结局筛选的开发队列（formal）内连接取得。

时间记号：术后日数（POD），手术当天从零计。

|步骤／分支|已接受计数|规则与来源|
|---|---|---|
|raw 主表|11,938 住院，61 中心|完整源队列；`a2/report.md`|
|clean 原表|320,143 行，11,931 住院|原样输入；`01_review/数据全貌与分组说明.md`；首次代码交接与等价性抽查记录（H1）等价性回执|
|raw → backing cache|462,721 行；v3 划分表 11,907 住院|重建后按首次 NEC／出院前过滤；保留 formal 及辅助集；H1 等价性回执、harness v3|
|backing → 逐日滚动预测任务（Task A） formal|301,468 日，11,674 住院，359 病例，1,066 阳性日|clean-linked a/b/none、仍在险、原 POD≤31；`r11/report.md`|
|完整 raw 日历 + clean → PI72-CLEAN|318,992 有标签日，11,931 住院，356 病例／阳性日|核对 clean 日期和行键、去缺失标签；不与 formal 相交；`ROUND_REDO_summary_zh.md`|
|PI72-CLEAN 每重复抽样|训练 1,116 住院；外测 279 住院|训练病例／对照 285/831，外测 71/208；`ROUND_REDO_summary_zh.md`、`r9/report.md`|

以上来源均为内部报告／回执，不随包分发，关键数字已摘入。backing 包含辅助行，因此 11,907 不是 formal 的 11,674；两条分支也不是逐级相减的单一漏斗。详细组别见 [GLOSSARY](GLOSSARY.md)。

排除规则与“各原因排除多少住院”须分开：clean 比 raw 少 7 次住院；原盘点给出其组成：NEC 表无记录 4、原晚期组（POD≥31）2、NEC 日期早于入院 1。该旧分组不等同于本包的 formal 分组；组成也不说明为何从 clean 中移除，已接受清表概况未记载实际剔除原因，**本包来源未记录原因（reason not documented in shipped sources）**，不能推定全部因 NEC 或缺失值被排除。backing 的随包 `legacy/build_v26.py:assemble_rows` 先剔除首次 NEC 当日及以后、出院当日及以后的行，再仅保留 clean-linked 的 formal 或辅助集合；未关联 clean、err 及不满足任一集合条件的行被丢弃。`harness_v3.make_splits` 只对保留下来的住院生成划分。因此 11,907 来自 backing，不是简单从 clean 的 11,931 直接删除某个已知组；这两个住院数之间逐原因计数**未在本包来源记录（reason not documented in shipped sources）**。backing 到 formal 再排除 c/d 辅助人群及 formal 窗口以外的行；PI72-CLEAN 另按 clean 保留日与非缺失标签构建。来源：`01_review/数据全貌与分组说明.md`（内部报告，不随包分发；关键计数已摘入）及上述随包代码；未为解释差额新增患者级推导。

## 私有输出树

以下是目录模式，N 表示重复，F 表示折。两个根目录都不得位于原始数据目录中；实际路径由 `config.py` 解析。经过可用时点屏蔽的编码增强特征方案（G-safe）权重和预测位于授权私有缓存根目录的简写（CACHE），不能只保留私有运行输出目录的简写（RUN）。

以下示例中的名称：授权私有缓存根目录环境变量（`PF_CACHE_ROOT`）；历史重建特征库名称（`D5`）；授权私有运行输出目录环境变量（`PF_RUN_DIR`）；编码增强的轻量梯度提升树参照（`GSAFE-LGB`）；本地清理表的全列可用时点屏蔽树模型参照（`T8-D5SAFE-LGB`）；日级参照（`A-D5-LGB`）.

缓存名称：`splits_v3` 是冻结住院划分表；`S.parquet` 是 `data.py:build_frames` 经 `line_S.build_features` 构建的支持状态缓存，不是历史切片的当日状态 S。

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

建议：输入和软件版本不变、CACHE 与 RUN 成对且仍在原绝对路径时，可重发原训练命令续跑。保留 models、encoding、predictions、jobs 回执和 fit_ledger；检查点验证训练绑定与输出存在性，不是任意缓存修复器。不要将旧 G-safe CACHE 与空 RUN 拼接；旧缓存会返回检查点，而新 ledger 无法代表原拟合历史。已有 `core/manifest.json` 时构建器拒绝重建；中断的 `.partial` 文件不构成成功缓存，也没有批次续建保证，需新的 CACHE。

建议：迁移机器／路径或升级包时建立新成对 CACHE/RUN，重新运行并核对；旧绝对路径不能自动重定位。不要清理 CACHE 后仅保留 RUN，也不要跳过版本冻结文件。发现映射歧义或人群变化时停止并记录差异，一起核对来源；删行或改键会改变输入身份。实际运行时先验收环境和合成测试。
