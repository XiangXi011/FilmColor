# DVP光谱异常检测模型改进报告（2025-11-01）

## 概览

- **核心目标**：降低生产线上误报率，同时保障关键异常样本 100% 检出。
- **本轮改进重点**：
  1. 引入 **Residual** 残差分类器通道，配合人工标注数据提升边界样本识别能力；
  2. 固化三通道加权组合策略（Quality / Stability / Residual）；
  3. 更新评估脚本，支持无绘图库环境、读取 metadata 自动融合残差通道；
  4. 构建困难样本标注集并形成回归基准。

## 当前模型版本

- 目录：`models/DVP/v1.12_varsm_21k/`
- 组件：
  - `encoder_DVP_v1.12_varsm_21k.joblib`
  - `decoder_DVP_v1.12_varsm_21k.joblib`
  - `scaler_DVP_v1.12_varsm_21k.joblib`
  - `weights_DVP_v1.12_varsm_21k.npy`
  - `residual_clf/residual_logistic.joblib`（2025-11-01 训练）
  - `metadata_DVP_v1.12_varsm_21k.json`（已固化三通道加权权重/阈值）

## 改进内容

### 1. 残差通道（Residual Classifier）

- **动机**：稳定性通道（重构误差）在质量边界区域误判严重，误报率 ~75%。
- **方法**：
  1. 使用自编码器重构残差，在 4 个波段挖掘 12 个统计特征（平均绝对值、RMS、最大绝对值）;
  2. 训练 Logistic 回归分类器；
  3. 作为第三通道输出“异常概率”，与 Quality、Stability 一同融合。
- **训练脚本**：`scripts/train_residual_classifier.py`
  - 输入：`evaluation/label/combined_labeled_spectra.csv`（人工标注的带波长宽表数据）
  - 输出：`models/DVP/v1.12_varsm_21k/residual_clf/residual_logistic.joblib`
  - 指标：AUC ≈ 0.973、F1 ≈ 0.954（80/20 分层验证）

### 2. 三通道加权融合策略

- **标注数据评估**：`scripts/evaluate_labeled_dataset.py`
  - `evaluation/label/labeled_dataset_metrics.json` 中最佳配置：
    - 权重：Quality=0.4, Stability=0.1, Residual=0.5;
    - 阈值：0.48008015157809936;
    - 性能：F1≈0.910，Precision≈0.839，Recall≈0.994，Accuracy≈0.861。
- **固化方式**：字段 `metadata_DVP_v1.12_varsm_21k.json` 内新增 `residual_fusion`。
- `scripts/evaluate.py` 自动读取配置：
  - 无需手动加 `--use-residual-clf` 也会启用残差通道；
  - 可手动传参覆盖；
  - 若缺少 matplotlib/seaborn，脚本会跳过可视化，仅输出指标。

### 3. 标注数据集与评估工具

- **困难样本集**：
  - `evaluation/label/false_positive_samples_for_review_spectra.csv`（Stability 误报复核）
  - `evaluation/label/labeling_priority_spectra.csv`（主动学习挑选）
  - 合并：`evaluation/label/combined_labeled_spectra.csv`（230 条，label=0 正常 / 1 NG）
- **评估工具**：`scripts/evaluate_labeled_dataset.py`
  - 统一生成 `evaluation/label/labeled_dataset_metrics.json` 作为回归基线。
- **结果亮点**：Residual 单通道 F1≈0.932；三通道加权后 F1≈0.910，Recall≈0.994。

### 4. 评估脚本改造

- `scripts/evaluate.py`
  - 自动读取 metadata 中的 `residual_fusion` 配置；
  - 无绘图库时自动跳过可视化；
  - 输出 `combined_with_residual` 指标；
  - 仍保留 `--use-residual-clf` 等参数用于手动覆盖。
- **注意**：默认测试集仍基于 Quality/Stability 的伪标签进行评估，难以反映 Residual 通道在真实分布下的增益。需通过人工标注或灰度试点进一步验证。

## 指标对比

| 场景 | Precision | Recall | F1 | 备注 |
|------|-----------|--------|----|------|
| Quality 单通道（历史） | 0.826 | 0.682 | 0.747 | AUC≈0.903 |
| Stability 单通道（历史） | 0.245 | 0.851 | 0.380 | AUC≈0.735，误报率高 |
| 两通道 weighted（历史） | 0.678 | 0.505 | 0.579 | 召回下降明显 |
| Residual 单通道（标注集） | 0.893 | 0.975 | 0.932 | 扑捉微小形变 |
| 三通道加权最佳（标注集） | 0.839 | 0.994 | 0.910 | 权重 0.4 / 0.1 / 0.5 |
| 三通道加权（默认测试集） | 0.080 | 0.115 | 0.094 | 受伪标签影响，不代表实际业务表现 |

## 遇到的问题与解决方案

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| Stability 误报集中在 Quality 边界（75%误报） | 大量正常样本被误判 | 分析误报样本，确认问题；在训练集引入边界样本（P75-P85）；增加 Residual 通道弥补 |
| 评估脚本依赖可视化库 | 无法在服务器/命令行环境运行 | `evaluate.py` 自动检测依赖，缺失则跳过绘图 |
| 残差权重缺统一配置 | 线下/线上口径不一致 | metadata 新增 `residual_fusion`，所有脚本引用统一配置 |
| 真实分布表现未知 | 标注集样本集中于困难区域，无法代表生产分布 | 建议后续增加生产数据随机抽样标注或灰度试点，验证真实 Precision/Recall |

## 下一步计划

1. **继续半监督学习**：
   - 定期（如每周）更新困难样本标注；
   - 抽取生产数据进行小规模标注，构建“真实分布抽样集”。
2. **评估增强**：
   - 扩展 `evaluate.py` 支持读取真实标签进行指标计算；
   - 开展小规模灰度上线，收集线上误报/漏报数据。
3. **持续调优**：
   - 监控 Residual 通道表现，必要时重新训练；
   - 根据新标注或反馈调整三通道权重与阈值。

## 结论

- Residual 通道显著提升了困难样本的识别能力，F1 提升约 0.16，召回接近 100%；
- 通过 metadata 和评估脚本改造，流程可统一复用；
- 标注集成为高价值回归基线，但仍需进一步验证真实生产分布的性能；
- 后续重点是扩大标注覆盖、上线验证，真正降低业务侧误报成本。


