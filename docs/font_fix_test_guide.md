# 中文字体修复测试指南

## 修复内容

已修复评估图片标题中文字体乱码问题。修改了以下文件：

1. **scripts/evaluate.py** - 评估脚本的字体设置函数
2. **utils/data_validator.py** - 数据验证器的字体设置函数  
3. **algorithms/decision_engine.py** - 决策引擎的可视化方法

## 修复说明

### 问题原因

在Windows系统上，matplotlib默认无法找到合适的中文字体，导致中文标题显示为乱码（方块字符）。

### 解决方案

修改了 `setup_matplotlib_for_plotting()` 函数，根据操作系统自动选择合适的字体：

- **Windows**: 优先使用 "Microsoft YaHei"（微软雅黑）、"SimHei"（黑体）
- **macOS**: 使用 "PingFang SC"、"Hiragino Sans GB"
- **Linux**: 使用 "Noto Sans CJK SC"、"WenQuanYi Zen Hei"

## 测试步骤

### 1. 确保依赖已安装

```bash
pip install matplotlib seaborn numpy scikit-learn joblib pandas
```

### 2. 运行评估脚本

```bash
python scripts/evaluate.py --samples 100 --model-dir models
```

### 3. 检查生成的图片

运行完成后，检查 `evaluation/` 目录下的图片文件：

- `quality_stability_analysis.png` - 质量稳定性分析图
- `spectral_reconstruction_comparison.png` - 光谱重构对比图
- `residual_analysis.png` - 残差分析图
- `confusion_matrix_and_roc.png` - 混淆矩阵和ROC曲线

### 4. 验证字体显示

打开生成的图片，检查以下内容：

✅ **正常显示**：所有中文标题和标签都清晰可见，没有方块字符
- 图表标题：如 "DVP涂层光谱异常检测评估结果"
- 坐标轴标签：如 "Quality Score"、"稳定性评分"、"波长 (nm)"
- 图例：如 "正常 (Normal)"、"质量异常 (Quality)"
- 文本标签：如 "假阳性率"、"真阳性率"

❌ **仍然乱码**：如果还有方块字符，可能是系统缺少字体，需要手动安装

## 快速测试脚本

如果只想快速测试字体设置，可以运行：

```bash
python test_font_fix.py
```

这会生成一个简单的测试图片 `evaluation/font_test.png`，用于验证字体设置是否正确。

## 手动验证字体

如果生成的图片仍有问题，可以检查系统可用的字体：

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查看所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
print("可用字体数量:", len(fonts))

# 查找中文字体
chinese_fonts = [f for f in fonts if any(keyword in f for keyword in ['YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'Ming', 'Song'])]
print("中文字体:", chinese_fonts)
```

## 故障排查

### 问题1：仍然显示乱码

**可能原因**：
- 系统没有安装中文字体
- matplotlib字体缓存需要更新

**解决方法**：
1. Windows系统通常自带微软雅黑，如果缺少可以手动安装
2. 清除matplotlib字体缓存：
   ```python
   import matplotlib.font_manager
   matplotlib.font_manager._rebuild()
   ```

### 问题2：找不到字体文件

**解决方法**：
1. 确认系统已安装中文字体
2. 手动指定字体路径：
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
   ```

## 修改的文件列表

1. `scripts/evaluate.py` - 第37-64行
2. `utils/data_validator.py` - 第18-47行
3. `algorithms/decision_engine.py` - 第360-373行

所有修改都向后兼容，不会影响现有功能。

