# 人员与编码 agent 的交接规则

English: [AGENTS.en.md](AGENTS.en.md)。先读 README、TASKS 与 STATUS_AND_NEXT；双语目录规则见 README。

- 原始数据只读，只用已有 DUA／IRB 授权的数据。患者级表、缓存、模型和预测不得进入 Git；仅写 PF_CACHE_ROOT 或 PF_RUN_DIR。
- 外部路径只通过 `src/pf_nec/config.py` 的环境变量解析。真实运行显式设置 PF_DATA_ROOT；路径默认值不代表授权。
- 不改冻结 spec、特征规则、种子、风险集或评价器以追求一致或更高指标。新研究另立版本，不复用已使用的锁定集声称独立验证。
- 先跑合成测试。Task A 每次先合并 5 折 OOF，再对重复等权；PI 同时报总体与术后。不得跨合同比较 AUROC。
- 每个预测日只用其结束前已知的信息；历史日先门控再生成窗口。G 编码只在对应训练池内交叉拟合。
- 每进程 2 数值线程、RSS 上限 8 GiB；顺序运行大任务，超限停止并汇报，不静默减少菜单或重复。
- Mac 先导入 torch，再导入 LightGBM。环境差异如实记录；迁移／续跑约束见 DATA_LAYOUT。
- `src/pf_nec/` 是最终模型与冻结评价；`src/explore/` 是探索性分析与脚手架。不要把解释工具称为未晋级性能程序。
- 输出使用〔事实〕／〔推断〕／〔建议〕或 [fact]/[inference]/[suggestion] 区分证据。重要性和改动输入后的分数不是治疗效应；没有完成的独立／外部验证不得声称完成。
- 新增／修改的文档同时维护中英版及相同数字。MANIFEST 的 source_path 仅为来源身份，不是可打开的包内依赖。

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pf_nec.verify
```
