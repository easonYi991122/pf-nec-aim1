# 信号接口与链接治理任务（T-6）：接口规格与未决表

合成血氧字段 `spo2_daily_mean` 对应脉搏血氧饱和度（SpO2）；具体测量定义、单位及质量规则须随合成字段合同共同固定，不能只凭血氧名称混同不同测量。

本页为无患者的合成接口提案，不是现有评分器的完整输入清单。首阶段可立即整理 schema 和未决项；评分器身份、目标、字段映射及生成约定共同确定前，完整评分重算保持前置条件未齐，保留原因并报告可完成部分（BLOCKED）。下表是合成字段名称、单位和可见时点约定，不复制真实记录、字典释义或权重。

时间记号：术后日数（POD），手术当天从零计。

|字段名|单位／类型|截至预测时点实际可见（as-of）时点与用途|
|---|---|---|
|synthetic_subject_key|合成字符串，无量纲|仅用于行身份与划分，不作预测输入；不得换成真实标识|
|score_at|带时区时间戳|当前评分日末；固定 cutoff，不由未来事件推算|
|gestational_age_weeks|周，数值|出生记录在 cutoff 前已录入；仅胎龄、诊断、手术类型与术后日数的锚点对照（anchor-only）输入|
|diagnosis_code|合成类别编码，无量纲|首次可见且不晚于 cutoff 的诊断；anchor-only 输入|
|surgery_type|合成类别编码，无量纲|手术日末且术时信息已记录后才可见，术前屏蔽；anchor-only 输入|
|pod|日，整数|按截至 cutoff 已知的索引手术计算；POD0 为手术日，术前缺失；anchor-only 输入|
|heart_rate_daily_mean|beats/min，数值|仅汇总当天且 availability_time 不晚于 cutoff 的合成心率观测；信号臂候选，不属于 anchor-only|
|spo2_daily_mean|%，数值|仅汇总当天且 availability_time 不晚于 cutoff 的合成血氧观测；信号臂候选，不属于 anchor-only|
|`registry_inputs.<field>.value`|由 required_fields 声明类型|登记评分器完整预测输入的逐字段映射；字段名、单位及可见门尚待冻结，不能以锚点代替缺失字段|
|`registry_inputs.<field>.unit`|规范单位字符串|必须匹配已冻结输入定义；未知或不兼容即 invalid，不自动猜换算|
|`registry_inputs.<field>.timestamp` / `registry_inputs.<field>.availability_time`|带时区时间戳|分别为测量／事件发生和录入可见时间；两者均不得晚于 cutoff|
|quality / missing_reason|合成质量标签／缺失原因|仅使用 cutoff 前已知状态；迟到与缺测不能默认正常值|
|scorer_id / scorer_version / target_id / required_fields|接口元数据|调用前冻结的身份、目标和必需字段规格；不从验证结局选取|

调用约定：`score_registry(request) -> {synthetic_subject_key, score_at, scorer_id, scorer_version, target_id, score, valid, missing_inputs}`。输入是上述元数据和预测字段，不含标签。有效时 score 是 0–1 的概率；缺必需字段、单位／可见时点非法或身份不匹配时 valid=false、score=null，并列明原因，不以零分或锚点评分静默替代。原型应验证相同对齐输入重算得到相同分数，并验证未来／迟到观测不改变过去评分。

目标关系：T-6 的未来 30 日模拟标签只是接口测试约定，与本地清理表评价合同（PI72-CLEAN）的本地清理表的单阳性日标签（outcome_3d）、逐日滚动预测任务（Task A）的随后三日首次结局标签（y3）和寻找治疗改进空间的研究用途（U2）的真实随后三十日首次院内结局标签（yB30）均不等同。必须另定合成评分器并按模拟目标拟合；若未来应用现有评分器，则保留其原目标身份，只作为基线／先验输入，现有评分器仍预测其原终点，改用新终点需要另定并拟合对应评分器。生成足够随访才能给模拟标签；未知观察不记阴性。

|一起确定的事项／所需材料|现在可以从哪里开始|
|---|---|
|评分器身份、目标、required_fields 及各字段定义／单位／as-of 映射|只建逐字段表；未匹配项标 unavailable，不宣称可重算|
|住院、事件、死亡／出院、缺测生成过程及合成随机种子|先列边界测试，不能按模拟性能挑生成参数|
|完整／缺测／迟到 3 个场景的缺测率、到达延迟和样本量|仅写配置键与未决项，不猜数字；仍受原中央处理器（CPU） 4 小时、图形处理器（GPU） 0 小时、真实结局 fit 0 的限额|
|仅锚点、评分单独、仅信号、信号加评分的统一行集及训练／验证隔离|用 synthetic_subject_key 定义隔离测试；仅锚点严格限上表标明的字段|
|数据持有人、诚实中介、伦理审查委员会（IRB）／数据使用协议（DUA）答复及验证污染审计|完成治理问题清单，不联系机构或推定获准链接|

本页只定义接口，不实现评分器。完整任务与零结果交付见 [TEAM_TASKS](TEAM_TASKS.md)，证据与方法边界见 [AIM2_LINKAGE_NOTE](AIM2_LINKAGE_NOTE.md)。
