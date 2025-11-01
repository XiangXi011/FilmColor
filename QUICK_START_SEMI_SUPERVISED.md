# 半监督学习快速启动

**当前环境**: 已激活 `(dvp_py310)` ✅  
**当前目录**: `/mnt/c/Users/24523/Desktop/film_color/train_project/code/spectrum_anomaly_detection` ✅

---

## 🚀 立即执行

### Step 1: 生成候选样本（10分钟）

```bash
python scripts/suggest_label_candidates.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-npz data/export_v11/test_subset.npz \
    --top-k 200 \
    --output-csv output/label_candidates_v1.12.csv \
    --export-spectra-csv output/label_candidates_spectra_v1.12.csv
```

**输出文件**:
- `output/label_candidates_v1.12.csv` - 候选样本列表（含Quality/Stability分数、uncertainty）
- `output/label_candidates_spectra_v1.12.csv` - 光谱数据（用于标注）

---

### Step 2: 人工标注（4-6小时）

**标注文件**: `output/label_candidates_spectra_v1.12.csv`

**操作**:
1. 用Excel打开CSV文件
2. 添加`label`列（建议在第4列，便于查看）
3. 逐行标注：
   - `label = 0`: 正常
   - `label = 1`: 异常
4. 保存为：`output/label_candidates_v1.12_labeled.csv`

**标注指南**: 详见 `docs/LABELING_GUIDE.md`

**快速判断**:
- 峰值位置偏移 > 5nm → 异常
- 峰值强度 < 85% 或 > 100% → 异常  
- 曲线有毛刺、噪声 → 异常
- 整体形状异常 → 异常

---

### Step 3: 训练残差分类器（10分钟）

**前提**: 完成Step 2标注

```bash
python scripts/train_residual_classifier.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --input-csv output/label_candidates_v1.12_labeled.csv \
    --output-dir models/DVP/v1.12_varsm_21k \
    --test-split 0.2
```

**输出文件**:
- `models/DVP/v1.12_varsm_21k/residual_clf_DVP_v1.12_varsm_21k.joblib`
- `models/DVP/v1.12_varsm_21k/residual_clf_training_report_v1.12.json`

---

### Step 4: 融合评估（10分钟）

```bash
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --combine-strategy weighted \
    --use-residual-clf \
    --residual-fuse-mode weighted \
    --test-data-npz data/export_v11/test_subset.npz \
    --optimize-thresholds f1 \
    --samples 1000
```

**查看结果**:
```bash
cat evaluation/performance_metrics.json
```

---

## 📊 预期效果

| 指标 | 当前 | 预期 | 提升 |
|-----|------|------|------|
| **Combined F1** | 0.747 | **0.80-0.85** | **+7-14%** |
| **Precision** | 0.826 | 0.75-0.85 | 持平 |
| **Recall** | 0.682 | **0.80-0.90** | **+17-32%** |

---

## ⚡ 最小可行方案（2小时）

如果时间有限，可以：
1. 只标注Top 50最不确定样本
2. 训练→评估
3. 如效果不佳再补充标注

---

## 📦 相关文档

- **实施计划**: `docs/SEMI_SUPERVISED_LEARNING_PLAN.md` (完整方案，14页)
- **标注指南**: `docs/LABELING_GUIDE.md` (详细标准和案例)
- **执行脚本**: `scripts/run_semi_supervised_pipeline.sh` (Linux/Mac一键脚本)

---

**开始执行Step 1！** 🚀

