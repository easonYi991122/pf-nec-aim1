# 未晋级方法入口

代码位于 `src/explore/`，与最终模型包使用同一冻结数据／评价接口。这里的方法不是推荐替换最终模型的结论。

- T3：`explore.t3.interpret` 与 `t3_fast` 保留原计算；`gsafe_t3.packets(engine, output)`验证固定预测可由保存模型重现，生成仅供私有内存聚合的SHAP packets。完整Task A解释须组装五折；SHAP是预测贡献，不是干预方向。
- I7：`python -m explore.i7.diagnostics --repeat 1 --arm B-only-w1`运行8背景变量模型。其它固定切片为B+C、B+S、B+C+S，窗口1或7日。`learning_curve_stays`保留原分层hash嵌套训练子集；`learning_slopes`保留原曲线统计接口。原曲线的模型家族未导出，不能声称复现其训练结果。若另做D5曲线须另立方案。
- Aim 1b：`python -m explore.aim1b.stage12 --synthetic` 和 `python -m explore.aim1b.stage34 --synthetic`展示聚合测量／positivity诊断。原函数可接受本地授权表；治疗可测性、资格、时间零点、混杂与可比性仍需PI／临床确认。它们不估计NEC治疗效应。

序列TCN／GRU、无顺序对照、M1及GPU尝试均未晋级，本包不提供其代码。其保留价值是后续样本量、优化与时间信息问题的研究线索，不能把既有负结果解释为方法永远无效。
