# 机器与 agent 交接规则

本规则适用于任意操作系统和编码 agent。先阅读 README 与 docs；根据自己的环境选择执行方式。

- 原始数据只读。仅使用已经获 DUA／IRB 授权的数据。患者级表、缓存、模型、预测不得进入 Git；只写 `PF_CACHE_ROOT` 或 `PF_RUN_DIR`。
- 所有外部路径由 `src/pf_nec/config.py` 的三个环境变量解析。默认数据路径不代表授权；真实运行必须设置 `PF_DATA_ROOT`。
- 不修改冻结 spec、特征规则、随机种子、风险集或评价函数来追求一致或提高指标。拟变更须另立版本并说明影响。
- 先运行合成测试。Task A 完整评价必须覆盖五个外折；重复内先合并 OOF，再对重复等权。PI 合同报告术后及总体。不得跨合同相减 AUROC。
- 预测日只使用当日结束前已知信息；历史值先按各历史日屏蔽，再生成窗口。G 编码在每个训练池内重新交叉拟合。
- 单进程2数值线程；Mac 总 RSS 上限8 GiB。不要并行启动多个大任务。超限停止并报告，不静默缩小菜单、训练池或重复数。
- Mac 在导入 LightGBM 前先导入 torch；代码入口已经这样处理。依赖版本见 env，平台差异应记录。

`src/pf_nec/` 是最终模型、数据构建与冻结评价；`src/explore/` 是未晋级方法；`tests/` 只用合成数据。完整命令见 README。检查入口：

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp="$PF_CACHE_ROOT/test-temp"
PYTHONPATH=src python -m pf_nec.verify
```

汇报必须区分实测、推断与未完成事项。不要把解释性 SHAP 当治疗效应，不把同一开发集再次评分称为独立验证。
