# 光谱异常检测系统开发进度报告

## 🎯 项目概述

基于技术规格文档，我们成功开发了一个完整的**混合式光谱异常检测系统**，专门用于DVP涂层类型的光谱质量监控。

## ✅ 已完成阶段

### Phase 1: 数据预处理和标准曲线分析 ✅
- **目标**: 解析HunterLab DVP.csv标准曲线，建立数据基础
- **完成内容**:
  - ✅ 成功解析DVP标准曲线数据（81个波长点，380-780nm）
  - ✅ 实现数据加载和预处理功能
  - ✅ 创建数据验证和可视化模块
  - ✅ 生成标准波长网格数据
- **关键指标**:
  - 数据质量: EXCELLENT
  - 波长覆盖: 380-780nm (步长5nm)
  - DVP反射率范围: 0.0200-14.4200

### Phase 2: 核心算法实现 - SimilarityEvaluator ✅
- **目标**: 实现基于专家规则的Quality Score计算
- **完成内容**:
  - ✅ SimilarityEvaluator类实现
  - ✅ 权重计算函数（针对DVP涂层优化）
  - ✅ 加权皮尔逊相关系数和加权RMSE
  - ✅ 批量评估功能
- **关键指标**:
  - 基准测试: 相同光谱得分100%
  - 权重分布: 基础权重29.6%, DVP增强权重38.3%, 范围权重32.1%
  - 质量阈值: Excellent≥95%, Good≥90%, Acceptable≥85%

### Phase 3: 加权自编码器模型开发 ✅
- **目标**: 构建Weighted Autoencoder深度学习模型
- **完成内容**:
  - ✅ 架构实现(81→48→16→4→16→48→81)
  - ✅ 自定义加权损失函数
  - ✅ StandardScaler预处理集成
  - ✅ 模型训练和验证流程
- **关键指标**:
  - 编码器R²: 0.2435
  - 解码器R²: 0.6210
  - 稳定性阈值: 4.977571 (99.5%分位数)
  - 异常检测: 噪声水平≥0.20检测为异常

### Phase 4: 模型训练脚本开发 ✅
- **目标**: 创建完整的train.py训练流程
- **完成内容**:
  - ✅ 命令行参数支持(--coating_name, --version等)
  - ✅ 模型保存功能(.joblib, .json, .npy)
  - ✅ 阈值计算(99.5%分位数)
  - ✅ 多产品类型训练支持
  - ✅ 完整训练日志和报告
- **关键指标**:
  - 训练时间: 0.91秒
  - 训练样本: 100个正常样本 + 50个异常样本
  - 模型文件: 5个组件文件
  - 训练成功率: 100%

## 本轮进展（v1.5）

- 评估侧：默认youden阈值优化；稳定性自动方向修正(AUC_probe<0.55)；组合分数AUC加权融合；评估图英文化；新增PR/阈值敏感性/分布图。
- 训练侧：固化更深AE结构+正则（L2/early-stopping）；AE仅用正常样本训练；支持 --data-npz；标准曲线稳健聚合。
- 数据侧：新增筛选/导出脚本（分位阈值、多历史剔除、分组限额、train/val/test拆分）。
- 代表性指标：Quality F1≈0.908/AUC≈0.987；Stability F1≈0.596/AUC≈0.564；Combined F1≈0.921/AUC≈0.938。

## 📁 项目文件结构

```
/workspace/code/spectrum_anomaly_detection/
├── data/
│   ├── dvp_processed_data.npz              # 处理后的DVP数据
│   └── HunterLab DVP.csv                   # 原始标准曲线
├── algorithms/
│   └── similarity_evaluator.py             # Quality Score计算器
├── models/
│   ├── weighted_autoencoder.py             # TensorFlow版本自编码器
│   └── DVP/
│       └── v1.0/                           # 训练好的模型文件
│           ├── encoder_DVP_v1.0.joblib
│           ├── decoder_DVP_v1.0.joblib
│           ├── scaler_DVP_v1.0.joblib
│           ├── weights_DVP_v1.0.npy
│           └── metadata_DVP_v1.0.json
├── scripts/
│   └── train.py                            # 主训练脚本
├── output/
│   ├── training_report_DVP_v1.0.json      # 训练报告
│   ├── phase1_summary.json                # 各阶段总结
│   └── *.png                              # 可视化图表
└── visualizations/
    └── dvp_standard_curve.png             # DVP标准曲线图
```

## 🔧 技术栈

- **数据处理**: pandas, numpy, scikit-learn
- **机器学习**: sklearn.neural_network.MLPRegressor
- **可视化**: matplotlib, seaborn
- **模型管理**: joblib, JSON
- **命令行**: argparse

## 📊 核心性能指标

| 组件 | 指标 | 数值 | 状态 |
|------|------|------|------|
| **数据质量** | 波长点数 | 81 | ✅ |
| | 波长范围 | 380-780nm | ✅ |
| | 数据完整性 | 100% | ✅ |
| **Quality Score** | 基准测试 | 100% | ✅ |
| | 权重优化 | DVP特定 | ✅ |
| | 质量阈值 | 85% | ✅ |
| **Stability Score** | 编码器R² | 0.2435 | ✅ |
| | 解码器R² | 0.6210 | ✅ |
| | 异常阈值 | 4.977571 | ✅ |
| **训练流程** | 训练时间 | 0.91秒 | ✅ |
| | 成功率 | 100% | ✅ |
| | 模型文件 | 5个 | ✅ |

## 🚀 下一步计划

### Phase 5: 模型评估和可视化系统 (中优先级)
- 开发评估脚本(evaluate.py)
- 实现QualityScore vs StabilityScore散点图
- 创建光谱重构对比可视化
- 生成残差分析图表
- 实现混淆矩阵和ROC曲线

### Phase 6: 决策引擎和API接口 (中优先级)
- 实现四格决策表逻辑
- 开发FastAPI服务接口
- 创建模型加载和缓存机制
- 实现实时分析API端点

### Phase 7: 模型调优方案和最佳实践 (中优先级)
- 超参数优化策略
- 阈值动态调整机制
- 模型版本管理和重训练策略

### Phase 8: 系统集成测试和文档 (低优先级)
- 端到端测试验证
- 性能基准测试
- 部署文档和用户手册

## 🎉 成果总结

我们成功实现了**光谱异常检测系统的核心功能**：

1. **数据基础**: 完整的数据预处理和验证流程
2. **Quality Score**: 基于专家规则的可解释质量评估
3. **Stability Score**: 基于机器学习的过程稳定性检测
4. **训练流程**: 自动化的模型训练和保存机制
5. **模型管理**: 标准化的文件组织和版本控制

系统现在可以：
- ✅ 加载和处理DVP标准曲线数据
- ✅ 计算Quality Score评估光谱质量
- ✅ 训练Weighted Autoencoder检测异常
- ✅ 自动保存训练好的模型文件
- ✅ 生成详细的训练报告和日志

**准备进入Phase 5: 模型评估和可视化系统开发！**

## 训练数据集选择原则（建议策略）

- **高置信正常为主，受控小偏差为辅**：AE 仅学习正常流形，避免明显异常混入训练。
- **分层抽样防偏倚**：按批次/时间进行分组限额，保证不同工况的均衡覆盖。
- **逐步扩量、可回滚**：小步放宽阈值，持续监控 Stability AUC/F1 与组合 F1；指标恶化则回退。

### 可执行标准（阈值与占比）

- **高置信（主力训练样本）**：
  - 默认分位阈值：相似度 ≥ P85，皮尔逊 ≥ P90（脚本自动计算）。
  - 数量偏少时可放宽：相似度 ≥ P80，皮尔逊 ≥ P85。
- **边缘样本（受控小偏差，少量加入）**：
  - 条件：相似度 ≥ P35 且 皮尔逊 ≥ P35，且不属于高置信。
  - 占比控制：≤20–30%（相对训练集）。当高置信=0 时，使用“总量占比 + 绝对上限”兜底。
- **分组限额（可选）**：
  - 以批次/日期列（如 `BatchId`）为组，edge 每组最多 N 条（如 200），防止单批次过载。

### 用现有脚本筛选（推荐命令）

分位数自适应筛选（含多历史剔除、分组限额、三分集拆分）：

```bash
python scripts/select_training_data.py \
  --input data/all_data.csv \
  --output-dir data/selection_v4 \
  --edge-max-ratio 0.25 \
  --edge-max-ratio-total 0.10 \
  --edge-max-abs 5000 \
  --edge-sort-by mix \
  --group-col BatchId --per-group-cap 200 \
  --exclude-index data/selection_v3/train_index.csv
```

导出训练子集（标准网格 NPZ，支持 train/val/test 三套）：

```bash
python scripts/export_training_subset.py \
  --input data/all_data.csv \
  --index-dir data/selection_v4 \
  --output-dir data/selection_v4/export
```

训练与评估（按新子集）：

```bash
python scripts/train.py --coating_name DVP --version v1.6 \
  --data-npz data/selection_v4/export/train_subset.npz \
  --std_curve_agg median

python scripts/evaluate.py --samples 1000 --random-seed 42 \
  --model-dir models
```

一键流水线（筛选 → 导出 → 训练 → 评估）：

```bash
python scripts/pipeline_oneclick.py \
  --input-csv data/all_data.csv \
  --work-dir data/selection_v4 \
  --export-dir data/selection_v4/export \
  --model-version v1.6 \
  --optimize-thresholds youden
```

### 何时再次放宽/扩量

- 当 **Stability AUC < 0.58** 或 **F1 下降**：先不扩量；回溯 edge 占比或分位阈值。
- 当 **Stability AUC ≥ 0.60 且 组合 F1 ≥ 0.92**：可小幅提高 edge 上限（如 abs 5000 → 8000），或将高置信阈值轻微放宽（P85 → P80）继续扩容。

### 防重复与持续迭代

- 使用脚本输出的 `exclude_index_next.csv` 作为下一轮 `--exclude-index`，杜绝历史样本重复。
- 每轮固定随机种子复评，观察 PR 曲线、阈值敏感性与分布图，确保“稳定性”的提升不以牺牲“组合”的稳定为代价。

## 本次更新（2025-10-31）

- 脚本与流程增强
  - 导出：`scripts/export_training_subset.py` 支持一次性导出 train/val/test 三套（`--index-dir`），向后兼容。
  - 一键：新增 `scripts/pipeline_oneclick.py` 串联“筛选→导出→训练→评估”，默认评估阈值优化=f1。
  - 评估：`scripts/evaluate.py` 默认阈值优化改为 f1；报告记录优化方法与稳定性翻转；可选启用“残差分段特征+Logistic”辅助通道并支持与AE误差融合（weighted/or）。
  - 训练：`scripts/train.py` 增加权重模式参数（`--weights-mode static|variance|variance_smooth|hybrid`），记录至metadata；默认仍用 w560 静态权重。
  - 预测：新增 `scripts/predict.py`，支持CSV/NPZ推断、阈值覆盖、按批次分位阈值（`--quality-pctl/--stability-pctl`）、可加载残差分类器并输出融合分数。
  - 主动学习：新增 `scripts/suggest_label_candidates.py` 生成TopK不确定样本清单，支持导出与训练集一致的光谱CSV（首行波长）。
  - 半监督：新增 `scripts/train_residual_classifier.py` 基于少量标注训练残差Logistic分类器。
  - 健康度：新增 `scripts/health_monitor.py` 输出 QHS/SHS/CHS 与 EWMA 控图（无阈值监控）。

- 指标与版本
  - 当前推荐发布：`DVP_v1.10_p50_w560_release`（two_stage + f1优化），Combined F1≈0.926，AUC≈0.948。
  - 残差融合作为可选“查全优先”开关，默认关闭；健康度脚本用于产线趋势监控。

- 标注与运营
  - 第一批标注建议：正常100、异常50（分层抽样，类型覆盖优先于数量）。
  - 主动学习：按不确定度（距 two_stage 边界的margin）挑样，TopK可配置（默认50，建议200–1000）。
