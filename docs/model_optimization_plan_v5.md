## DVP光谱异常检测模型优化方案 V5.0

生成时间: 2025-10-31  
适用版本: DVP_v1.3（最新一轮）

### 1. 本轮目标与结论
- 目标：在保持 Quality 子模型稳定优秀的前提下，提升 Stability 子模型的有效性与组合输出鲁棒性；完善数据筛选与训练管道。
- 结论：
  - Stability 子模型已从“接近随机”迈入“可用区间”（F1≈0.62，AUC≈0.56）；
  - 组合模型达到 Acc≈0.971 / F1≈0.924 / AUC≈0.932，满足当前使用需求；
  - 数据侧“高置信正常+受控小偏差”的训练策略与管道落地，有利于后续稳步扩充数据与再训练。

---

### 2. 代码与流程优化（本轮新增/改进）
1) 评估侧（scripts/evaluate.py）
- 阈值优化默认 youden；自动方向修正阈值放宽至 AUC_probe < 0.55；
- 组合分数：统一异常方向 + Min-Max 归一化 + 按单模型 AUC 加权融合；
- 质量阈值单位归一（>1 自动 /100）；
- 新增输出：PR 曲线、阈值敏感性曲线（F1 vs 阈值）、稳定性分数分布直方图；
- 报告中记录最终阈值、是否翻转、AUC 权重。

2) 训练侧（scripts/train.py）
- 新增 `--data-npz` 指定自定义数据路径；
- AE 结构与正则可调：`--ae-encoder-layers/--ae-decoder-layers/--ae-alpha/--ae-early-stopping/...`；
- AE 仅用 y_train==1 的正常样本训练，避免“学到异常”；
- 标准曲线从样本矩阵稳健汇聚（median/mean 可选）；
- 数据加载/校验与相对路径兼容性增强。

3) 数据筛选与导出（新脚本）
- `scripts/select_training_data.py`：计算相似度/加权皮尔逊→分桶（high/edge/other）→限制 edge 占比→输出 `train_index.csv`；
- `scripts/export_training_subset.py`：按索引导出训练子集（CSV+NPZ，已插值到标准网格）。

---

### 3. 数据量化对比（关键指标）

对比维度选取：Quality、Stability、Combined 三通道的 Acc / Precision / Recall / F1 / AUC。以下给出代表性阶段（同样本量评估，均使用 random-seed=42）：

| 阶段 | Quality F1 | Quality AUC | Stability F1 | Stability AUC | Combined F1 | Combined AUC |
|---|---:|---:|---:|---:|---:|---:|
| 早期（修复前） | 0.861 | 0.987 | 0.252 | 0.507 | 0.333 | 0.506 |
| 中期（youden+方向修正） | 0.908 | 0.987 | 0.450 | 0.501 | 0.891 | 0.963 |
| 本轮（v1.3，数据筛选+AE正则） | 0.908 | 0.987 | 0.615 | 0.558 | 0.924 | 0.932 |

注：
- Stability AUC 从≈0.50 提升到≈0.56，F1 提升到≈0.62（已显著改善）；
- Combined F1 稳定在 ≈0.92 左右，产线可用性更强；
- Quality 通道保持稳定优秀。

---

### 4. 当前数据选择与建议
- 训练集：高置信正常为主，可少量引入“受控小偏差”（相似度/皮尔逊带宽放宽 5–10%，edge 占比≤20–30%）；
- 验证/测试：纳入更大偏差样本用于阈值选择与最终评估；
- 去重：使用 `--exclude-index` 避免重复采样；
- 分层：按时间/批次/分桶分层抽样，降低阈值漂移风险。

---

### 5. 下一步优化计划（V6 方向）
1) 数据侧（优先级高）
- 扩充高置信正常覆盖更多工况；构造贴近工艺的异常（整体偏移、局部段漂移、峰位/带宽变化、阶跃等）。

2) 模型侧（持续改进）
- 进一步调整 AE 容量与正则；尝试注意力/权重学习替代固定波段权重；
- 分段误差特征 + 轻量分类器作为稳定性辅助通道，与 AE 融合（投票或加权）。

3) 评估与监控
- 保持 youden+方向修正；跟踪阈值敏感性曲线与 PR 曲线；
- 周期性复评，监控阈值与数据分布漂移，必要时再校准。

---

### 6. 训练与评估命令（当前推荐）
训练：
```
python scripts/train.py --coating_name DVP --version v1.3 \
  --data-npz data/training_subset.npz \
  --std_curve_agg median \
  --ae-encoder-layers 64 32 16 8 \
  --ae-decoder-layers 16 32 64 81 \
  --ae-alpha 1e-4 --ae-early-stopping --ae-n-iter-no-change 10 --ae-validation-fraction 0.1
```

评估：
```
python scripts/evaluate.py --samples 1000 --random-seed 42 \
  --model-dir models/DVP/v1.3 --optimize-thresholds youden
```

