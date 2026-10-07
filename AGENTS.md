# 人员与人工智能（AI）编码助手的共同约定

[English](AGENTS.en.md)。可以先读 [README](README.md)、[任务定义](docs/TASKS.md)和[当前状态](docs/STATUS_AND_NEXT.md)。

仓库维护者维护本包并记录共享评价合同，方便大家比较结果和复核修改。

- 我们都需要遵守数据使用协议（DUA）、伦理审查委员会（IRB）要求及 首席研究者（PI） 的数据协议。原始数据只读；患者级表、缓存、模型权重和预测不得进入 Git，只写授权私有缓存根目录（PF_CACHE_ROOT）或运行目录（PF_RUN_DIR）。未发表知识产权的共享边界见 [EXCLUDED](docs/EXCLUDED.md)。
- 在已登记主机以外使用患者级数据，仍须按上述协议先取得仓库维护者和 PI 的确认；任务卡和默认路径不构成传输授权。这项义务适用于我们所有人。
- 外部路径统一通过 `src/pf_nec/config.py` 的环境变量解析。真实运行显式设置授权数据根目录（PF_DATA_ROOT），让输入身份可追溯；默认路径不代表授权。
- 为保持可比，复现保留固定模型规格、特征规则、种子、风险集和评价器。新想法可以共同讨论并另立版本；既有留出集已用于开发，换种子不能恢复独立验证。
- 先跑合成测试可尽早发现接口和时序问题。日级任务（Task A）每次先合并 5 折折外预测（OOF），再对重复等权；本地清理表合同（PI72-CLEAN）同时报告总体和术后指标。受试者工作特征曲线下面积（AUROC）只在相同合同、风险集及验证方案内比较。
- 每个预测日只用其结束前已知的信息，历史日先门控再组窗口；增强编码（G）只在对应训练池内交叉拟合，避免外测信息进入训练。
- 每进程 2 数值线程、常驻内存（RSS）上限 8 二进制吉字节（GiB），大任务顺序运行。达到上限时保留记录、一起检查原因；静默减少菜单或重复会破坏预定比较。
- Mac 先导入 torch 再导入轻量梯度提升机（LightGBM），以正确加载数值运行库。环境差异和续跑要求见 [DATA_LAYOUT](docs/DATA_LAYOUT.md)。
- `src/pf_nec/` 提供最终模型与固定评价，`src/explore/` 提供探索性分析与脚手架。重要性和改动输入后的分数不能证明治疗效应，研究报告清楚区分已测事实、解释和建议。
- 文档中英文同步、数字一致，缩写先说明原意。文件清单（MANIFEST）的 `source_path` 字段记录原项目的来源位置，不是包内运行路径；修改后一起复核版本、哈希、合成测试与文档。

以下是我们自己检查代码时使用的入口：

以下示例中的名称：解释器模块搜索路径环境变量（`PYTHONPATH`）；禁止生成字节码缓存的解释器环境变量（`PYTHONDONTWRITEBYTECODE`）.

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp="$PF_RUN_DIR/test-temp"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pf_nec.verify
```
