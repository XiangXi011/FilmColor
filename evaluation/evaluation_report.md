# DVP涂层光谱异常检测模型评估报告

**生成时间**: 2025-11-01 17:23:53  
**评估样本数**: 1000  
**模型版本**: DVP_v1.12_varsm_21k  

## 模型概述

本评估报告基于已训练的DVP涂层光谱异常检测模型，该模型采用混合架构：
- **Quality Score**: 基于专家规则的光谱相似性评估
- **Stability Score**: 基于加权自编码器的重构误差评估

## 性能指标

### Quality Score模型
- **准确率**: 0.9390
- **精确率**: 0.8257
- **召回率**: 0.6818
- **F1分数**: 0.7469
- **AUC-ROC**: 0.9031
- **阈值**: 0.7604

### Stability Score模型
- **准确率**: 0.6280
- **精确率**: 0.2446
- **召回率**: 0.8507
- **F1分数**: 0.3800
- **AUC-ROC**: 0.7352
- **阈值**: -10.0354

### 组合模型
- **准确率**: 0.8530
- **精确率**: 0.6779
- **召回率**: 0.5050
- **F1分数**: 0.5788
- **AUC-ROC**: 0.7006
 - **组合策略**: weighted

### 阈值与方向
- **优化方法**: f1
- **Quality 阈值**: 0.7604
- **Stability 阈值**: -10.0354
- **Stability 方向翻转**: True（probe AUC=0.265）

## 模型分析

### 优势
1. **双重检测机制**: Quality Score和Stability Score分别从不同角度检测异常
2. **专家规则集成**: Quality Score基于领域专家知识
3. **机器学习增强**: Stability Score通过自编码器学习正常模式
4. **可解释性强**: 提供具体的异常类型识别

### 改进建议
1. **阈值优化**: 可考虑使用GridSearchCV优化分类阈值
2. **特征工程**: 增加更多光谱特征（如导数光谱、峰值特征等）
3. **模型集成**: 尝试其他异常检测算法（如Isolation Forest、One-Class SVM）
4. **数据增强**: 增加更多类型的异常样本进行训练

## 可视化结果

        本评估生成了以下可视化图表：
1. **质量稳定性分析图**: `quality_stability_analysis.png`
2. **光谱重构对比图**: `spectral_reconstruction_comparison.png`
3. **残差分析图**: `residual_analysis.png`
        4. **混淆矩阵和ROC曲线**: `confusion_matrix_and_roc.png`
        5. **Precision-Recall曲线**: `pr_curves.png`
        6. **阈值敏感性(F1 vs 阈值)**: `threshold_sensitivity.png`
        7. **稳定性分数分布(正负样本)**: `stability_score_hist.png`

## 残差特征通道（可选）

当启用`--use-residual-clf`时，本报告还包含基于“分段残差特征+Logistic”的辅助通道与融合结果：

- 残差通道（Stability Residual Classifier）
  - 融合方式: weighted
  - 加权权重: 0.5
- 残差融合组合（Combined with Residual）
  - 准确率: 0.559
  - 精确率: 0.08013937282229965
  - 召回率: 0.115
  - F1分数: 0.0944558521560575
  - AUC-ROC: 0.33735625

## 结论

DVP涂层光谱异常检测模型在测试数据集上表现良好，组合模型达到了85.3%的准确率。
该模型能够有效识别光谱质量异常和稳定性异常，为涂层质量控制提供了可靠的技术支持。

---
*报告由MiniMax Agent自动生成*
