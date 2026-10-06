# Project Passion Fruit：NEC 最终模型代码交接

English: [README.en.md](README.en.md)

本仓库只含研究代码、冻结规则、合成测试和汇总结果。使用者须已经获准访问同一 PC4 数据；仓库不提供数据、训练权重或预测。这是开发集交叉验证的复现材料，尚无独立外部验证。

|合同|保留模型|主指标|重复／外层|
|---|---|---|---|
|PI72-CLEAN 本地 clean-table 合同|GSAFE-LGB（G-safe，629 列）；T8-D5SAFE-LGB（单日参照，604 列）|POD≥0 AUROC|r1–5，每次一个固定住院外测划分|
|Task A formal v2.6／harness v3|A-D5-LGB，单日604列|总 AUROC；术后为关键次指标|r0–4，每次5折住院 OOF|

两合同的行、标签和验证方案不同，AUROC 不跨合同比较。PI72-CLEAN 不代表恢复了论文最终评分设计。Task B 本轮未重训。

从仓库根目录执行，`python` 指向已按 `env/` 配置的解释器：

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
python -m pf_nec.cli evaluate --model GSAFE-LGB --repeats 1 2 3 4 5
python -m pf_nec.cli evaluate --model T8-D5SAFE-LGB --repeats 1 2 3 4 5
python -m pf_nec.cli evaluate --model A-D5-LGB --repeats 0 1 2 3 4 --harness
```

命令按顺序运行，数值线程限制为2；资源不足时停止，不减少重复或改变配置。缓存只建立一次，重建需指定新缓存目录。已完成训练可按同一命令续跑。输出均留在私有路径，提交前检查 Git 暂存文件。

运行时间与本次实测边界见 [环境](docs/ENVIRONMENT.md)；数据结构见 [数据布局](docs/DATA_LAYOUT.md)；指标与四格分解见 [评价合同](docs/EVALUATION.md)。T3 解释、I7 切片／学习曲线方法、Aim 1b 脚手架见 [探索入口](explore/README.md)，均未晋级。所有依赖切断及其复现影响见 [排除清单](docs/EXCLUDED.md)。
