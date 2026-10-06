# 数据与私有输出布局

设置 `PF_DATA_ROOT` 指向已经获准访问的只读数据目录，要求：

```text
PF_DATA_ROOT/
  NEC Cleaned Data.csv
  Raw CSV Files/
    IndexSurgHosp.csv
    NEC.csv
    PreopRiskFactor.csv
    ... 其余原始表，共20表
```

保留原文件名、字段名、日期文本与编码；程序读取UTF-8或既定cp1252，不改输入。实际原始表清单由私有构建回执记录hash。运行时无需读取PI训练模型或原作者源代码。Aim 1b的额外字典语义核验仍需使用者在授权范围内查阅原数据字典；字典不在代码包中。

`PF_CACHE_ROOT` 存放新建的v2.6行／v3划分、PI合同缓存、局部重建映射和临时文件；`PF_RUN_DIR` 存放模型、OOF、拟合回执与汇总。二者可位于仓库外，由同一config解析，不能位于数据目录内。默认分别是仓库内被忽略的`.private`及其`runs`子目录；真实运行仍建议显式设置私有位置。

全新机器执行 `python -m pf_nec.cli build-data`，不需要原工作区缓存。该步骤先重建完整日历，再分别生成PI与formal行。formal保留冻结缓存的键表示，并在模型入口验证原始键可无损恢复；不擅自把精度不同的住院键合并。

Git默认忽略数据、缓存、模型及预测类型；忽略规则不是共享授权。提交前仍需检查暂存清单，公开包完整性由MANIFEST记录；本地临时产物不得加入交付树。
