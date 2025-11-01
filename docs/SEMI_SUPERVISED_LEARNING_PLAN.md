# DVP光谱异常检测 - 半监督学习实施计划

**日期**: 2025-11-01  
**目标**: 通过半监督学习将Combined F1从0.747提升至0.80+  
**当前最佳**: Quality-First策略 (F1=0.747, Precision=0.826, Recall=0.682)  

---

## 📊 现状分析

### 当前模型性能限制

| 策略 | F1 | Precision | Recall | 主要问题 |
|-----|----|-----------| -------|---------|
| Quality-First | 0.747 | 0.826 | 0.682 | 漏检率32% |
| Weighted (F1) | 0.579 | 0.678 | 0.505 | 漏检率49.5% |
| OR (two_stage) | 0.545 | 0.382 | 0.950 | 误报率62% |

**根本原因**：
- 无监督方法依赖预设规则和统计模式
- 无法学习复杂的异常边界
- Quality和Stability各有盲区但无法精准互补

**突破路径**：
- 引入少量标注样本（100-200个）
- 训练监督分类器学习真实异常模式
- 融合三个通道：Quality + Stability + Residual Classifier

---

## 🎯 实施路线图

### 阶段1: 候选样本生成（已有基础）

**目标**: 选择最具信息量的样本进行标注

**方法**: 主动学习 - 选择"最不确定"的样本
- 靠近决策边界的样本最有价值
- 基于Quality和Stability的margin（到阈值的距离）
- 优先标注uncertainty最小的样本

**执行命令**:
```bash
# 从测试数据中生成候选样本
python scripts/suggest_label_candidates.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-npz data/export_v11/test_subset.npz \
    --top-k 200 \
    --output-csv output/label_candidates_v1.12.csv \
    --export-spectra-csv output/label_candidates_spectra_v1.12.csv
```

**输出**:
- `label_candidates_v1.12.csv`: 包含Quality/Stability分数、uncertainty的候选列表
- `label_candidates_spectra_v1.12.csv`: 对应的光谱数据（便于可视化标注）

### 阶段2: 人工标注

**标注规模**: 150-200个样本
- **正常样本**: 100个（占67%）
- **异常样本**: 50-100个（占33%）

**标注策略**:
1. **Top 50高不确定样本**: 必须标注（决策边界）
2. **随机50正常样本**: 增加正常样本多样性
3. **随机50异常样本**: 覆盖不同类型异常
4. **额外50边缘样本**: 根据初步标注结果补充

**标注规则**:
```
label = 0  # 正常
  - Quality Score > 0.85 且无明显形状异常
  - 光谱曲线平滑连续
  - 与标准曲线相似度高

label = 1  # 异常
  - 存在以下任一情况：
    * 峰值位置偏移 > 5nm
    * 峰值强度异常（过高/过低 > 15%）
    * 曲线出现毛刺、断裂
    * 整体形状与标准曲线差异大
    * 存在噪声干扰
```

**标注工具**:
- Excel打开`label_candidates_spectra_v1.12.csv`
- 添加`label`列（0或1）
- 可选：添加`comment`列记录异常原因

**质量控制**:
- 双人标注：30%样本由两人独立标注，检验一致性
- 一致性目标：Cohen's Kappa ≥ 0.80
- 如果不一致，讨论后重新标注

### 阶段3: 训练残差分类器

**特征工程**: 
残差分类器使用"分段残差特征"，而非原始光谱：
```python
# 对每个样本计算残差
residual = original_spectrum - reconstructed_spectrum

# 提取4个波段的统计特征
bands = [(400,480), (480,560), (560,680), (680,780)]
for each band:
    - mean_abs: 平均绝对残差
    - rms: 均方根误差
    - max_abs: 最大绝对残差
# 总特征维度: 4波段 × 3特征 = 12维
```

**训练命令**:
```bash
# 训练残差分类器
python scripts/train_residual_classifier.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-csv output/label_candidates_v1.12_labeled.csv \
    --output-dir models/DVP/v1.12_varsm_21k \
    --test-split 0.2
```

**输出**:
- `residual_clf_DVP_v1.12_varsm_21k.joblib`: 训练好的残差分类器
- `residual_clf_training_report_v1.12.json`: 训练报告
  - 训练集AUC/F1
  - 测试集AUC/F1
  - 特征重要性

**预期性能** (基于残差特征):
- Residual Classifier AUC: 0.75-0.85
- Residual Classifier F1: 0.70-0.80

### 阶段4: 三通道融合评估

**融合策略**: 加权组合三个分数

```python
# 三个异常分数
quality_anomaly_score    # 0-1，越高越异常
stability_anomaly_score  # 0-1，越高越异常
residual_anomaly_score   # 0-1，越高越异常（残差分类器输出概率）

# 根据各自的AUC加权
α = AUC_quality / (AUC_quality + AUC_stability + AUC_residual)
β = AUC_stability / (AUC_quality + AUC_stability + AUC_residual)
γ = AUC_residual / (AUC_quality + AUC_stability + AUC_residual)

combined_score = α * quality + β * stability + γ * residual
```

**评估命令**:
```bash
# 融合评估
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --combine-strategy weighted \
    --use-residual-clf \
    --residual-fuse-mode weighted \
    --test-data-npz data/export_v11/test_subset.npz \
    --optimize-thresholds f1 \
    --samples 1000
```

**预期结果**:
- Combined F1: **0.80-0.85** (当前0.747)
- Combined Precision: 0.75-0.85
- Combined Recall: 0.80-0.90
- Combined AUC: 0.85-0.90

---

## 📅 实施时间表

| 阶段 | 任务 | 预计时间 | 负责人 |
|-----|------|---------|--------|
| **阶段1** | 生成候选样本 | 10分钟 | AI助手（自动） |
| **阶段2** | 人工标注150-200样本 | **4-6小时** | 领域专家 |
| **阶段3** | 训练残差分类器 | 10分钟 | AI助手（自动） |
| **阶段4** | 融合评估 | 10分钟 | AI助手（自动） |
| **总计** | - | **1工作日** | - |

**关键路径**: 人工标注（4-6小时）

---

## 💰 成本效益分析

### 投入成本
- **人力成本**: 1名专家 × 4-6小时 = 0.5-0.75人天
- **计算成本**: 可忽略不计（训练10分钟）
- **机会成本**: 延迟1工作日部署

### 预期收益
- **性能提升**: F1从0.747提升至0.80-0.85 (+7-14%)
- **误报率降低**: 预计从17%降至10-15%
- **漏检率降低**: 预计从32%降至15-20%
- **长期价值**: 建立标注样本库，支持后续持续改进

### ROI评估
- **短期ROI** (3个月):
  - 减少误报：假设每天100次检测，误报率从17%降至12%
  - 节省人工复检时间：5次/天 × 5分钟 × 60工作日 = 25小时
  - 换算：25小时 > 6小时标注时间 → **ROI = 4.2x**

- **长期ROI** (1年):
  - 持续学习：标注样本可重复使用
  - 模型迭代：支持后续版本快速优化
  - 知识积累：形成异常样本库和判定标准

---

## 🚀 快速启动指南

### Step 1: 生成候选样本（立即执行）

```bash
cd /mnt/c/Users/24523/Desktop/film_color/train_project/code/spectrum_anomaly_detection

# 激活环境
# conda activate dvp_py310  # 或者你的环境名

# 生成候选样本（选择最不确定的200个）
python scripts/suggest_label_candidates.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-npz data/export_v11/test_subset.npz \
    --top-k 200 \
    --output-csv output/label_candidates_v1.12.csv \
    --export-spectra-csv output/label_candidates_spectra_v1.12.csv

echo "✅ 候选样本已生成！"
echo "📂 输出文件："
echo "   - output/label_candidates_v1.12.csv"
echo "   - output/label_candidates_spectra_v1.12.csv"
echo ""
echo "📋 下一步：人工标注（添加label列：0=正常，1=异常）"
```

### Step 2: 标注指南（交给领域专家）

**标注文件**: `output/label_candidates_spectra_v1.12.csv`

**操作步骤**:
1. 用Excel打开CSV文件
2. 在最后一列添加`label`列
3. 查看每行的光谱数据（400-780nm的反射率值）
4. 判断是否异常：
   - `label = 0`: 正常
   - `label = 1`: 异常
5. 保存为：`output/label_candidates_v1.12_labeled.csv`

**标注技巧**:
- 参考`quality_score`和`stability_score`列作为辅助
- 如果两个分数都接近阈值（uncertainty小），说明这是边界样本
- 重点关注峰值位置、峰值强度、曲线平滑性
- 可以用Python/Excel绘制光谱曲线辅助判断

### Step 3: 训练残差分类器（标注完成后）

```bash
# 训练残差分类器
python scripts/train_residual_classifier.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-csv output/label_candidates_v1.12_labeled.csv \
    --output-dir models/DVP/v1.12_varsm_21k \
    --test-split 0.2

echo "✅ 残差分类器训练完成！"
echo "📂 输出文件："
echo "   - models/DVP/v1.12_varsm_21k/residual_clf_DVP_v1.12_varsm_21k.joblib"
echo "   - models/DVP/v1.12_varsm_21k/residual_clf_training_report_v1.12.json"
```

### Step 4: 融合评估（验证效果）

```bash
# 三通道融合评估
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --combine-strategy weighted \
    --use-residual-clf \
    --residual-fuse-mode weighted \
    --test-data-npz data/export_v11/test_subset.npz \
    --optimize-thresholds f1 \
    --samples 1000

echo "✅ 融合评估完成！"
echo "📂 查看结果："
echo "   - evaluation/performance_metrics.json"
echo "   - evaluation/evaluation_report.md"
```

---

## 📊 预期性能对比

### 融合前 vs 融合后

| 指标 | 当前最佳<br>(Quality-First) | 预期<br>(三通道融合) | 提升 |
|-----|------------------------|------------------|------|
| **Combined F1** | 0.747 | **0.80-0.85** | **+7-14%** |
| **Precision** | 0.826 | 0.75-0.85 | 持平或略降 |
| **Recall** | 0.682 | **0.80-0.90** | **+17-32%** |
| **误报率** | 17.4% | **10-15%** | **↓40%** |
| **漏检率** | 31.8% | **10-20%** | **↓50%** |

### 关键改进点

1. **召回率大幅提升**:
   - 残差分类器学习真实异常模式
   - 弥补Quality和Stability的盲区
   - 预计召回率从0.682提升至0.80-0.90

2. **误报率进一步降低**:
   - 监督学习提供更精准的异常边界
   - 减少基于规则的误判
   - 预计误报率从17%降至10-15%

3. **整体F1突破0.80**:
   - 平衡precision和recall
   - 达到生产部署的理想水平

---

## 🛠️ 备选方案

### 方案A: 轻量级标注（100个样本）

如果标注资源有限：
- 只标注Top 100最不确定样本
- 预期F1提升至0.78-0.82（略低于200样本）
- 时间缩短至2-3小时

### 方案B: 渐进式标注

分批标注，逐步改进：
1. **第一批**: 标注50个样本 → 训练 → 评估
2. **第二批**: 根据第一批结果，标注50个补充样本
3. **第三批**: 继续补充至150-200个

优势：可以更早看到初步效果，及时调整策略

### 方案C: 众包标注

如果有多名专家：
- 每人标注50-100个样本
- 重叠标注30%样本用于一致性检验
- 并行标注，缩短总时间至1-2小时

---

## ⚠️ 风险与缓解

### 风险1: 标注质量不佳

**影响**: 残差分类器性能差，无法提升F1

**概率**: 中等

**缓解措施**:
- 提供详细标注指南
- 双人标注30%样本验证一致性
- 标注前培训：展示典型正常/异常案例

### 风险2: 标注样本代表性不足

**影响**: 模型泛化能力差，测试集效果好但生产环境差

**概率**: 低

**缓解措施**:
- 使用主动学习选择最具信息量的样本
- 包含边界样本、典型样本、边缘样本
- 覆盖不同类型异常（峰值偏移、强度异常、噪声等）

### 风险3: 过拟合标注数据

**影响**: 训练集性能高但测试集性能差

**概率**: 中等

**缓解措施**:
- 20%数据作为测试集
- 使用正则化（Logistic回归的C参数）
- 监控训练/测试性能差距

### 风险4: 时间成本超预期

**影响**: 延迟部署时间

**概率**: 高（标注通常比预期慢）

**缓解措施**:
- 采用方案A（轻量级标注100个样本）
- 或方案B（渐进式标注，先快速迭代）
- 多人并行标注

---

## 📈 成功指标

### 必达指标（P0）
- ✅ Combined F1 ≥ 0.80
- ✅ Combined Recall ≥ 0.75
- ✅ Residual Classifier AUC ≥ 0.75

### 期望指标（P1）
- 🎯 Combined F1 ≥ 0.85
- 🎯 Combined Precision ≥ 0.80
- 🎯 Combined Recall ≥ 0.85
- 🎯 误报率 ≤ 15%
- 🎯 漏检率 ≤ 20%

### 优秀指标（P2）
- 🏆 Combined F1 ≥ 0.90
- 🏆 Combined AUC ≥ 0.90

---

## 📞 支持与帮助

### 技术支持
- 如遇脚本错误，检查：
  1. Python环境是否激活
  2. 模型文件路径是否正确
  3. 输入数据格式是否符合要求

### 标注支持
- 提供标注培训材料：`docs/LABELING_GUIDE.md`（待创建）
- 提供典型案例库：正常样本 × 10，异常样本 × 10
- 提供可视化工具：光谱曲线绘制脚本

---

**准备就绪，开始执行！** 🚀

第一步：运行候选样本生成命令（见上方Quick Start）

