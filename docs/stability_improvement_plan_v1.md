# Stability Score 判别力提升方案 v1

**创建日期**: 2025-11-01  
**目标**: 提升Stability Score的AUC从0.45→0.60+，F1从0.44→0.55+  
**当前版本**: DVP_v1.10_p50_w560_release

---

## 1. 问题诊断

### 1.1 核心问题

通过系统诊断分析（`analyze_stability_issue.py`），识别出以下核心问题：

| 问题编号 | 问题描述 | 严重性 | 当前状态 |
|---------|---------|-------|----------|
| **问题1** | 训练样本严重不足：仅使用200个样本（0.75%可用数据） | 🔴 高 | 实际有26666个可用样本未充分利用 |
| **问题2** | Stability AUC=0.451，接近随机猜测水平 | 🔴 高 | 模型几乎没有判别力 |
| **问题3** | 阈值分离度仅1.31σ，远低于3σ标准 | 🟡 中 | 阈值与正常分布严重重叠 |

### 1.2 对比分析

| 指标 | Quality Score | Stability Score | 差距 |
|-----|--------------|----------------|------|
| AUC | 0.9867 | 0.4506 | **0.536** |
| F1 | 0.9085 | 0.4413 | **0.467** |
| 召回率 | 1.0000 | 0.3481 | **0.652** |
| 精确率 | 0.8323 | 0.6026 | 0.230 |

**结论**: Quality Score表现优秀，说明数据和标签质量良好，Stability Score的低性能主要由**训练策略**而非数据本身导致。

---

## 2. 改进方案

### 2.1 优先级排序

| 优先级 | 改进措施 | 预期提升 | 实施难度 |
|-------|---------|---------|---------|
| **P0** | 扩大训练样本：200→2000+ | AUC +0.10~0.15 | 低 |
| **P0** | 降低edge样本占比：20%→10-15% | AUC +0.05~0.08 | 低 |
| **P1** | 优化权重设计：试验variance_smooth模式 | AUC +0.02~0.05 | 中 |
| **P1** | 调整阈值方法：P99.9→P99.5或MAD | F1 +0.05~0.08 | 低 |
| **P2** | 增加AE正则化：alpha=2e-4→3e-4 | 防止过拟合 | 低 |
| **P3** | 架构升级：AE→VAE | AUC +0.05~0.10 | 高 |

### 2.2 具体实施步骤

#### 步骤1：扩大训练样本 + 优化数据筛选 (P0)

**目标**: 训练样本从200增加到2000+，edge占比从20%降至10-15%

**操作命令**:
```bash
# 重新筛选训练数据（调整分位数阈值和edge占比）
python scripts/select_training_data.py \
    --input data/all_data.csv \
    --output-dir data/selection_v11 \
    --high-sim-pctl 80 \
    --high-pear-pctl 85 \
    --edge-sim-pctl 45 \
    --edge-pear-pctl 50 \
    --edge-max-ratio 0.15 \
    --split-train 0.70 \
    --split-val 0.15 \
    --split-test 0.15

# 导出训练子集
python scripts/export_training_subset.py \
    --input data/all_data.csv \
    --index-dir data/selection_v11 \
    --output data/training_subset_v11.npz
```

**验证点**:
- 训练集样本数≥2000
- edge样本占比≤15%
- high样本占比≥80%

#### 步骤2：试验不同权重模式 (P1)

**目标**: 对比static vs variance_smooth模式的性能差异

**操作命令**:
```bash
# 训练variance_smooth模式模型
python scripts/train.py \
    --coating DVP \
    --version v1.12_variance_smooth \
    --data-npz data/training_subset_v11.npz \
    --weights-mode variance_smooth \
    --alpha 0.0002

# 训练static模式模型（作为对照）
python scripts/train.py \
    --coating DVP \
    --version v1.12_static \
    --data-npz data/training_subset_v11.npz \
    --weights-mode static \
    --alpha 0.0002
```

#### 步骤3：增加正则化强度 (P2)

**目标**: 防止过拟合，提高泛化能力

**操作命令**:
```bash
# 训练更高正则化的模型
python scripts/train.py \
    --coating DVP \
    --version v1.13_alpha3e4 \
    --data-npz data/training_subset_v11.npz \
    --weights-mode variance_smooth \
    --alpha 0.0003
```

#### 步骤4：评估与对比 (必选)

**操作命令**:
```bash
# 评估v1.12_variance_smooth
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_variance_smooth \
    --output-dir evaluation/runs/v1.12_variance_smooth \
    --optimize-thresholds f1

# 评估v1.12_static
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_static \
    --output-dir evaluation/runs/v1.12_static \
    --optimize-thresholds f1

# 评估v1.13_alpha3e4
python scripts/evaluate.py \
    --model-dir models/DVP/v1.13_alpha3e4 \
    --output-dir evaluation/runs/v1.13_alpha3e4 \
    --optimize-thresholds f1
```

---

## 3. 预期效果

### 3.1 量化目标

| 指标 | 基线（v1.10） | 目标（v1.12+） | 提升幅度 |
|-----|-------------|---------------|---------|
| Stability AUC | 0.451 | **0.60+** | +0.15 |
| Stability F1 | 0.441 | **0.55+** | +0.11 |
| Stability 召回率 | 0.348 | **0.50+** | +0.15 |
| Combined F1 | 0.926 | **≥0.90** | 保持 |
| 阈值分离度 | 1.31σ | **≥2.5σ** | +1.2σ |

### 3.2 里程碑定义

| 里程碑 | 条件 | 状态 |
|-------|------|------|
| **M1: 可用** | Stability AUC≥0.55, Combined F1≥0.90 | 待验证 |
| **M2: 良好** | Stability AUC≥0.60, Combined F1≥0.92 | 待验证 |
| **M3: 优秀** | Stability AUC≥0.65, Combined F1≥0.93 | 待验证 |

---

## 4. 风险与缓解

### 4.1 潜在风险

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| Combined F1下降 | 中 | 高 | 保持edge样本≥10%，不要完全剔除 |
| 训练时间过长 | 低 | 中 | 使用early_stopping，限制max_iter |
| 过拟合 | 中 | 中 | 增加正则化alpha，监控train/val loss差距 |
| 数据分布偏移 | 低 | 高 | 保留val/test集验证泛化能力 |

### 4.2 回退策略

如果v1.12+版本的Combined F1<0.90，则：
1. 回退到v1.10_p50_w560_release作为生产版本
2. 调整edge_max_ratio从15%→18%重新训练
3. 考虑使用残差分类器辅助通道（`--use-residual-clf`）

---

## 5. 执行计划

### 5.1 时间线

| 阶段 | 任务 | 预计时长 |
|-----|------|---------|
| **第1天** | 数据筛选v11 + 导出训练集 | 1小时 |
| **第2天** | 训练v1.12 (static + variance_smooth) | 2小时 |
| **第3天** | 训练v1.13 (alpha=3e-4) + 评估所有模型 | 2小时 |
| **第4天** | 对比分析 + 选择最优版本 + 发布 | 1小时 |

**总预计时长**: 6小时

### 5.2 成功标准

- ✅ Stability AUC≥0.60
- ✅ Stability F1≥0.55
- ✅ Combined F1≥0.90
- ✅ 阈值分离度≥2.5σ
- ✅ 评估报告完整（ROC/PR/阈值敏感性/分布图）

---

## 6. 后续优化方向

如果v1.12+达到目标（Stability AUC≥0.60），可考虑：

1. **半监督学习路径**: 
   - 标注第一批样本（正常100+异常50）
   - 训练残差分类器并融合
   - 主动学习迭代标注

2. **架构升级路径**:
   - 实现VAE（Variational Autoencoder）
   - 捕获潜在分布，提升判别力
   - 需重构代码（PyTorch/TensorFlow）

3. **健康度监控部署**:
   - 固化health_monitor参数
   - 生产环境部署CHS监控
   - 趋势告警与可视化

---

**文档状态**: ✅ 待执行  
**负责人**: AI Agent  
**审核人**: 用户  
**更新日期**: 2025-11-01

