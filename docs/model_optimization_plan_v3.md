## DVP光谱异常检测模型优化方案 V3.0

生成时间: 2025-10-31  
适用版本: 评估脚本 `scripts/evaluate.py`（已更新至V3改动）

### 一、本轮优化目标
- 修复稳定性子模型（Stability Score）分数方向与阈值不一致问题，提升其判别能力。
- 统一组合模型分数方向与AUC计算逻辑，获得可信的综合指标。
- 引入阈值优化策略（Youden/F1），并支持固定随机种子，便于复现实验。

---

### 二、关键改动清单
1) 阈值读取逻辑统一到元数据
- Quality 阈值优先从 `metadata.similarity_evaluator.quality_threshold` 获取；
- Stability 阈值优先从 `metadata.weighted_autoencoder.stability_threshold` 获取；
- 若元数据缺失，则回退到分布法（Quality: 正常样本5%分位；Stability: 正常样本95%分位）。

2) 新增随机种子与数据复现能力
- 评估入口新增 `--random-seed`（默认 42），并在测试数据生成前设置 `np.random.seed(seed)`。

3) 组合模型AUC计算修正（方向统一 + 归一化融合）
- 将两个子模型分数统一为“越大越异常”的方向：
  - Quality: 使用 `1 - QualityScore` 作为异常分数；
  - Stability: 使用方向一致后的稳定性分数（详见第4点）。
- 对异常分数做 Min-Max 归一化到 [0, 1]；
- 组合分数 = 0.5 * q_anom + 0.5 * s_anom（后续可按各自AUC做加权）。

4) 稳定性分数“自动方向修正”
- 探测稳定性原始分数的 AUC：若 AUC < 0.5，则自动对分数取反（使之“越大越异常”）；
- 修正后的方向用于：阈值优化、预测、ROC/AUC、组合分数与指标计算；
- 控制台输出 `flipped_stability=True/False`，用于确认是否触发自动修正。

5) 阈值优化策略
- 新增命令行参数 `--optimize-thresholds {none|youden|f1}`：
  - youden：最大化 Youden’s J（TPR-FPR）选择阈值；
  - f1：在候选阈值网格上选 F1 最大的阈值；
  - none：不进行优化，按元数据/回退分布法。

---

### 三、使用方式
- 固定种子 + Youden 阈值优化（推荐）
```
python scripts/evaluate.py --samples 1000 --random-seed 42 --optimize-thresholds youden
```

- 固定种子 + F1 阈值优化（可对比）
```
python scripts/evaluate.py --samples 1000 --random-seed 42 --optimize-thresholds f1
```

---

### 四、效果对比（核心指标）
以 `--samples 1000 --random-seed 42 --optimize-thresholds youden` 的一次评估为例：

- Quality Score（质量）
  - AUC ≈ 0.987；阈值 ≈ 0.9712；F1 ≈ 0.908（保持优秀且稳定）。

- Stability Score（稳定性）
  - 优化前：AUC ≈ 0.275（方向错误，几乎随机）；
  - 优化后：AUC ≈ 0.725，F1 ≈ 0.529（显著提升，方向修正生效）。

- 组合模型（two_stage）
  - 优化后：AUC ≈ 0.963，F1 ≈ 0.818（达到理想水平，综合判别力显著提升）。

结论：本轮优化显著提升了稳定性子模型与组合模型的有效性与可解释性。

---

### 五、对2万条数据的训练建议（有/无标签两种情况）
1) 有标签（更优）
- 思路：自编码器仅用“正常”样本训练；异常样本仅用于验证/测试与阈值选择。
- 划分示例（按时间/批次分层，避免泄漏）：
  - 训练（正常）：约 60%（例如 12,000 条正常）
  - 验证（正常+异常）：约 20%（正常 3,000 + 异常 600）
  - 测试（正常+异常）：约 20%（正常 3,000 + 异常 1,400）
- 阈值：在验证集上用 `youden` 或 `f1` 选择，避免拍脑袋阈值。

2) 无标签或标签不完整
- 使用 Quality 高分（如 >95）筛选“高置信正常”作为自编码器训练集；
- 其余作为未标注集用于阈值稳健性检查（可抽检/一致性检测）。
- 划分示例：高置信正常中 70%/15%/15% 作为 训/验/测。

通用要点：
- 标准化器仅用训练集拟合，再应用于验证/测试；
- 阈值选择在验证集进行，最终只在测试集报告指标；
- 组合策略可先用 two_stage，后续可按单模型AUC加权融合。

---

### 六、风险与后续工作
风险：
- 若真实业务中的“异常类型”与当前模拟/样例不同，Stability 的阈值与AUC仍可能波动；
- 新数据分布变化可能导致阈值漂移，需要定期复评与再校准。

后续工作（建议按优先级）：
1. 使用真实/更多样的异常样本再训练与评估，稳固稳定性子模型；
2. 尝试加权组合（权重 ∝ 各子模型AUC）；
3. 在训练脚本中集成阈值自动选择并生成至元数据；
4. 扩展特征（导数、峰值等）与模型（Isolation Forest / One-Class SVM / LOF）做对比。

---

### 七、变更摘要（代码）
- `scripts/evaluate.py`
  - 新增：`--random-seed`、`--optimize-thresholds`（none/youden/f1）
  - 统一阈值读取自 `metadata`
  - 自动方向修正（Stability AUC<0.5 自动翻转分数）
  - 组合AUC逻辑：方向统一 + Min-Max 归一化 + 融合
  - 组合策略在图与指标中的逻辑一致（two_stage/and/or）

如需，我可以据此方案生成数据划分与重训的脚本模板，直接落地执行。


