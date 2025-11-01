#!/usr/bin/env python3
"""
Quality Score与Stability Score相关性验证实验

目的：
1. 计算Quality Score和Stability Score的相关系数
2. 判断两个分数是否提供独立信息
3. 决定是否保留Stability通道

决策标准：
- 如果 |r| > 0.7 → 高度相关，考虑移除Stability
- 如果 |r| < 0.4 → 独立性好，保留并继续优化

Author: AI Assistant
Date: 2025-11-01
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


class ScoreCorrelationValidator:
    """Score相关性验证器"""
    
    def __init__(self, model_dir: str, test_data_npz: str = None):
        """
        初始化验证器
        
        Args:
            model_dir: 模型目录
            test_data_npz: 测试数据NPZ路径
        """
        self.model_dir = Path(model_dir)
        self.test_data_npz = test_data_npz
        self.output_dir = project_root / "evaluation" / "correlation_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.data_loader = SpectrumDataLoader()
        self.similarity_evaluator = SimilarityEvaluator()
        
        # 加载模型
        self.models = {}
        self._load_models()
        
        print(f"✅ 相关性验证器初始化完成")
        print(f"📁 模型目录: {self.model_dir}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def _load_models(self):
        """加载模型文件"""
        try:
            # 查找模型文件
            def _find_one(patterns):
                for pattern in patterns:
                    matches = sorted(self.model_dir.glob(pattern))
                    if matches:
                        return matches[0]
                return None
            
            encoder_path = _find_one([
                "*encoder*DVP*v*.joblib", "*encoder*DVP*.joblib", "*encoder*.joblib"
            ])
            decoder_path = _find_one([
                "*decoder*DVP*v*.joblib", "*decoder*DVP*.joblib", "*decoder*.joblib"
            ])
            scaler_path = _find_one([
                "*scaler*DVP*v*.joblib", "*scaler*DVP*.joblib", "*scaler*.joblib"
            ])
            
            if not (encoder_path and decoder_path and scaler_path):
                raise FileNotFoundError("未找到所需的模型文件")
            
            self.models['encoder'] = joblib.load(encoder_path)
            self.models['decoder'] = joblib.load(decoder_path)
            self.models['scaler'] = joblib.load(scaler_path)
            
            # 加载权重
            weights_file = _find_one([
                "weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"
            ])
            if weights_file and Path(weights_file).exists():
                self.models['weights'] = np.load(weights_file)
            else:
                # 创建默认权重
                wavelengths, _ = self.data_loader.load_dvp_standard_curve()
                weights = np.ones(len(wavelengths))
                peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                weights[peak_mask] *= 1.5
                self.models['weights'] = weights
                print("⚠️  使用默认DVP权重向量")
            
            print("✅ 模型文件加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载测试数据
        
        Returns:
            Tuple[spectra, wavelengths]
        """
        if self.test_data_npz and Path(self.test_data_npz).exists():
            print(f"📂 加载真实测试数据: {self.test_data_npz}")
            data = np.load(self.test_data_npz)
            spectra = data['dvp_values']
            wavelengths = data['wavelengths']
            print(f"✅ 加载 {len(spectra)} 个测试样本")
            return spectra, wavelengths
        else:
            # 回退到标准曲线
            print("⚠️ 未指定测试数据，使用DVP标准曲线")
            wavelengths, standard = self.data_loader.load_dvp_standard_curve()
            # 生成一些变体用于测试
            spectra = []
            for i in range(1000):
                noise = np.random.normal(0, 0.01, len(standard))
                spectra.append(standard + noise)
            return np.array(spectra), wavelengths
    
    def calculate_scores(self, spectra: np.ndarray, wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算Quality Score和Stability Score
        
        Args:
            spectra: 光谱数据 [N, W]
            wavelengths: 波长数组 [W]
            
        Returns:
            Tuple[quality_scores, stability_scores]
        """
        print("🔄 计算Quality Score和Stability Score...")
        
        # 加载标准曲线
        _, standard_spectrum = self.data_loader.load_dvp_standard_curve()
        
        # 计算Quality Score
        quality_scores = []
        for spectrum in spectra:
            result = self.similarity_evaluator.evaluate(
                spectrum, standard_spectrum, wavelengths, coating_name="DVP"
            )
            quality_score = result['similarity_score']
            quality_scores.append(quality_score)
        
        quality_scores = np.array(quality_scores)
        
        # 计算Stability Score (重构误差)
        stability_scores = []
        for spectrum in spectra:
            # 数据预处理
            spectrum_scaled = self.models['scaler'].transform(spectrum.reshape(1, -1))
            
            # 通过编码器-解码器重构
            encoded = self.models['encoder'].predict(spectrum_scaled)
            decoded = self.models['decoder'].predict(encoded)
            
            # 计算重构误差
            spectrum_original = self.models['scaler'].inverse_transform(spectrum_scaled)
            reconstruction_error = np.mean(
                self.models['weights'] * (spectrum_original - decoded) ** 2
            )
            
            stability_scores.append(reconstruction_error)
        
        stability_scores = np.array(stability_scores)
        
        print(f"✅ Score计算完成:")
        print(f"   - Quality Score范围: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")
        print(f"   - Stability Score范围: [{stability_scores.min():.6f}, {stability_scores.max():.6f}]")
        
        return quality_scores, stability_scores
    
    def analyze_correlation(self, quality_scores: np.ndarray, stability_scores: np.ndarray) -> Dict:
        """
        分析相关性
        
        Args:
            quality_scores: Quality Score数组
            stability_scores: Stability Score数组
            
        Returns:
            分析结果字典
        """
        print("\n" + "="*60)
        print("📊 相关性分析")
        print("="*60)
        
        # 1. Pearson相关系数
        pearson_r, pearson_p = stats.pearsonr(quality_scores, stability_scores)
        print(f"\n1. Pearson相关系数:")
        print(f"   r = {pearson_r:.4f} (p-value = {pearson_p:.4e})")
        
        # 2. Spearman秩相关系数
        spearman_r, spearman_p = stats.spearmanr(quality_scores, stability_scores)
        print(f"\n2. Spearman秩相关系数:")
        print(f"   ρ = {spearman_r:.4f} (p-value = {spearman_p:.4e})")
        
        # 3. 判断相关性强度
        abs_pearson = abs(pearson_r)
        abs_spearman = abs(spearman_r)
        
        print(f"\n3. 相关性强度判断:")
        if abs_pearson > 0.7:
            correlation_strength = "强相关"
            recommendation = "❌ 建议移除Stability Score（高度相关，不提供额外信息）"
            decision = "remove_stability"
        elif abs_pearson > 0.4:
            correlation_strength = "中等相关"
            recommendation = "⚠️ Stability提供部分独立信息，但重叠较多，建议谨慎保留"
            decision = "cautious_keep"
        else:
            correlation_strength = "弱相关/独立"
            recommendation = "✅ 建议保留Stability Score（独立性好，提供额外信息）"
            decision = "keep_stability"
        
        print(f"   - Pearson |r| = {abs_pearson:.4f} → {correlation_strength}")
        print(f"   - Spearman |ρ| = {abs_spearman:.4f}")
        print(f"\n4. 决策建议:")
        print(f"   {recommendation}")
        
        # 4. 计算R²（决定系数）
        # Quality对Stability的解释程度
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(quality_scores.reshape(-1, 1), stability_scores)
        r2_q_to_s = lr.score(quality_scores.reshape(-1, 1), stability_scores)
        
        # Stability对Quality的解释程度
        lr.fit(stability_scores.reshape(-1, 1), quality_scores)
        r2_s_to_q = lr.score(stability_scores.reshape(-1, 1), quality_scores)
        
        print(f"\n5. R²（决定系数）分析:")
        print(f"   - Quality → Stability: R² = {r2_q_to_s:.4f} (Quality能解释Stability {r2_q_to_s*100:.1f}%的变异)")
        print(f"   - Stability → Quality: R² = {r2_s_to_q:.4f} (Stability能解释Quality {r2_s_to_q*100:.1f}%的变异)")
        
        # 6. 互信息（非线性相关性）
        from sklearn.feature_selection import mutual_info_regression
        mi_q_to_s = mutual_info_regression(
            quality_scores.reshape(-1, 1), stability_scores, random_state=42
        )[0]
        mi_s_to_q = mutual_info_regression(
            stability_scores.reshape(-1, 1), quality_scores, random_state=42
        )[0]
        
        print(f"\n6. 互信息（非线性相关性）:")
        print(f"   - MI(Quality → Stability) = {mi_q_to_s:.4f}")
        print(f"   - MI(Stability → Quality) = {mi_s_to_q:.4f}")
        
        results = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'correlation_strength': correlation_strength,
            'r2_quality_to_stability': float(r2_q_to_s),
            'r2_stability_to_quality': float(r2_s_to_q),
            'mi_quality_to_stability': float(mi_q_to_s),
            'mi_stability_to_quality': float(mi_s_to_q),
            'decision': decision,
            'recommendation': recommendation
        }
        
        return results
    
    def create_visualizations(self, quality_scores: np.ndarray, stability_scores: np.ndarray, results: Dict):
        """
        创建可视化图表
        
        Args:
            quality_scores: Quality Score数组
            stability_scores: Stability Score数组
            results: 分析结果字典
        """
        print("\n📊 创建可视化图表...")
        
        # 设置样式
        plt.style.use('seaborn-v0_8')
        import platform
        system = platform.system()
        if system == "Windows":
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quality Score vs Stability Score 相关性分析', 
                     fontsize=16, fontweight='bold')
        
        # 1. 散点图 + 回归线
        ax1.scatter(quality_scores, stability_scores, alpha=0.5, s=20)
        
        # 添加回归线
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(quality_scores.reshape(-1, 1), stability_scores)
        y_pred = lr.predict(quality_scores.reshape(-1, 1))
        ax1.plot(quality_scores, y_pred, 'r-', linewidth=2, 
                label=f'线性回归 (R²={results["r2_quality_to_stability"]:.3f})')
        
        ax1.set_xlabel('Quality Score', fontsize=12)
        ax1.set_ylabel('Stability Score', fontsize=12)
        ax1.set_title(f'散点图 (Pearson r={results["pearson_r"]:.3f})', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 六边形密度图
        hexbin = ax2.hexbin(quality_scores, stability_scores, gridsize=30, cmap='YlOrRd')
        ax2.set_xlabel('Quality Score', fontsize=12)
        ax2.set_ylabel('Stability Score', fontsize=12)
        ax2.set_title('密度分布图', fontsize=14)
        plt.colorbar(hexbin, ax=ax2, label='样本密度')
        
        # 3. Quality Score分布 (KDE)
        ax3.hist(quality_scores, bins=50, alpha=0.7, color='blue', density=True, label='直方图')
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(quality_scores)
        x_range = np.linspace(quality_scores.min(), quality_scores.max(), 200)
        ax3.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax3.set_xlabel('Quality Score', fontsize=12)
        ax3.set_ylabel('密度', fontsize=12)
        ax3.set_title('Quality Score分布', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Stability Score分布 (KDE)
        ax4.hist(stability_scores, bins=50, alpha=0.7, color='green', density=True, label='直方图')
        kde = gaussian_kde(stability_scores)
        x_range = np.linspace(stability_scores.min(), stability_scores.max(), 200)
        ax4.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax4.set_xlabel('Stability Score', fontsize=12)
        ax4.set_ylabel('密度', fontsize=12)
        ax4.set_title('Stability Score分布', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "correlation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存: {output_path}")
        
        # 创建第二张图：详细分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('相关性详细分析', fontsize=16, fontweight='bold')
        
        # 1. 残差图
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(quality_scores.reshape(-1, 1), stability_scores)
        y_pred = lr.predict(quality_scores.reshape(-1, 1))
        residuals = stability_scores - y_pred
        
        ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('预测的Stability Score', fontsize=12)
        ax1.set_ylabel('残差', fontsize=12)
        ax1.set_title('残差图（检验线性关系）', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q图（检验残差正态性）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('残差Q-Q图（正态性检验）', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. 相关性强度可视化
        metrics = ['Pearson |r|', 'Spearman |ρ|', 'R² (Q→S)', 'R² (S→Q)']
        values = [
            abs(results['pearson_r']),
            abs(results['spearman_r']),
            results['r2_quality_to_stability'],
            results['r2_stability_to_quality']
        ]
        colors = ['red' if v > 0.7 else 'orange' if v > 0.4 else 'green' for v in values]
        
        bars = ax3.barh(metrics, values, color=colors, alpha=0.7)
        ax3.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='强相关阈值(0.7)')
        ax3.axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='中等相关阈值(0.4)')
        ax3.set_xlabel('相关性强度', fontsize=12)
        ax3.set_title('相关性指标汇总', fontsize=14)
        ax3.legend()
        ax3.set_xlim(0, 1)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax3.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=10)
        
        # 4. 决策建议文本框
        ax4.axis('off')
        decision_text = f"""
        相关性验证结果
        ================
        
        Pearson相关系数: r = {results['pearson_r']:.4f}
        Spearman秩相关: ρ = {results['spearman_r']:.4f}
        
        相关性强度: {results['correlation_strength']}
        
        R² (Quality → Stability): {results['r2_quality_to_stability']:.4f}
        R² (Stability → Quality): {results['r2_stability_to_quality']:.4f}
        
        互信息:
        - MI(Q→S): {results['mi_quality_to_stability']:.4f}
        - MI(S→Q): {results['mi_stability_to_quality']:.4f}
        
        决策建议
        ========
        {results['recommendation']}
        
        下一步行动
        ==========
        """
        
        if results['decision'] == 'remove_stability':
            decision_text += """
        1. 简化为Quality-Only架构
        2. 直接投入半监督学习
        3. 训练监督分类器替代Stability
        """
        elif results['decision'] == 'keep_stability':
            decision_text += """
        1. 保留双通道架构
        2. 继续三通道融合方案
        3. 执行半监督学习提升性能
        """
        else:  # cautious_keep
            decision_text += """
        1. 暂时保留Stability
        2. 进行误报样本分析
        3. 根据分析结果决定最终方案
        """
        
        ax4.text(0.1, 0.5, decision_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "correlation_detailed_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 详细分析图表已保存: {output_path}")
    
    def generate_report(self, results: Dict, n_samples: int):
        """
        生成分析报告
        
        Args:
            results: 分析结果字典
            n_samples: 样本数量
        """
        print("\n📝 生成分析报告...")
        
        report = f"""# Quality Score与Stability Score相关性验证报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**样本数量**: {n_samples}  
**模型目录**: {self.model_dir}

## 1. 验证目的

本报告旨在验证Quality Score和Stability Score之间的相关性，以决定是否保留Stability通道。

**决策标准**：
- 如果 |r| > 0.7 → 高度相关，考虑移除Stability
- 如果 |r| < 0.4 → 独立性好，保留并继续优化

## 2. 相关性分析结果

### 2.1 线性相关性

**Pearson相关系数**: r = {results['pearson_r']:.4f} (p-value = {results['pearson_p']:.4e})

- |r| = {abs(results['pearson_r']):.4f}
- 相关性强度: **{results['correlation_strength']}**

### 2.2 秩相关性

**Spearman秩相关系数**: ρ = {results['spearman_r']:.4f} (p-value = {results['spearman_p']:.4e})

- |ρ| = {abs(results['spearman_r']):.4f}

### 2.3 决定系数（R²）

**Quality → Stability**: R² = {results['r2_quality_to_stability']:.4f}
- Quality能解释Stability {results['r2_quality_to_stability']*100:.1f}%的变异

**Stability → Quality**: R² = {results['r2_stability_to_quality']:.4f}
- Stability能解释Quality {results['r2_stability_to_quality']*100:.1f}%的变异

### 2.4 非线性相关性（互信息）

**MI(Quality → Stability)**: {results['mi_quality_to_stability']:.4f}  
**MI(Stability → Quality)**: {results['mi_stability_to_quality']:.4f}

## 3. 决策建议

{results['recommendation']}

**决策**: `{results['decision']}`

## 4. 下一步行动

"""
        
        if results['decision'] == 'remove_stability':
            report += """
### 场景A：移除Stability Score（高度相关）

**结论**: Stability不提供额外信息，与Quality高度冗余

**行动计划**:
1. **简化架构**: 从双通道简化为Quality-Only
2. **直接半监督学习**: 跳过Stability优化，投入标注工作
3. **训练监督分类器**: 基于标注样本训练分类器，替代Stability
4. **预期效果**: F1可能直接提升至0.80-0.85

**时间成本**: 1-2天（标注150-200样本 + 训练分类器）

**风险**: 
- 低风险：相关性验证已证明Stability冗余
- 简化架构反而可能提升可解释性
"""
        elif results['decision'] == 'keep_stability':
            report += """
### 场景B：保留Stability Score（独立性好）

**结论**: Stability提供独立信息，与Quality互补

**行动计划**:
1. **保留双通道架构**: Quality + Stability双通道检测
2. **执行误报分析**: 理解Stability 75%误报率原因
3. **半监督学习**: 训练三通道融合（Quality + Stability + Residual Classifier）
4. **预期效果**: F1提升至0.80-0.85

**时间成本**: 2-3天（误报分析4小时 + 标注1-2天）

**优势**:
- 双通道互补，覆盖更全面
- Stability能捕获Quality遗漏的异常类型
"""
        else:  # cautious_keep
            report += """
### 场景C：谨慎保留（中等相关）

**结论**: Stability提供部分独立信息，但与Quality有重叠

**行动计划**:
1. **优先执行误报分析**: 理解Stability 75%误报率原因（4小时）
2. **根据误报分析结果决策**:
   - 如果误报主要是边界样本（P75-P85） → 训练数据太纯净，考虑移除
   - 如果误报是真实异常模式 → 保留并优化
3. **根据分析结果选择方案**:
   - 方案A: 移除Stability → 简化架构 + 半监督学习
   - 方案B: 保留Stability → 三通道融合 + 半监督学习

**时间成本**: 4小时（误报分析） + 根据结果再决定后续投入

**推荐理由**: 
- 相关性处于模糊区间，需要更多证据
- 误报分析能提供关键决策依据
- 风险最小化策略
"""
        
        report += """

## 5. 技术细节

### 5.1 相关性指标解释

| 指标 | 值 | 含义 |
|-----|---|------|
| Pearson r | {:.4f} | 线性相关强度 (-1到1) |
| Spearman ρ | {:.4f} | 秩相关强度 (对异常值鲁棒) |
| R² (Q→S) | {:.4f} | Quality对Stability的解释能力 |
| R² (S→Q) | {:.4f} | Stability对Quality的解释能力 |
| MI (Q→S) | {:.4f} | 非线性相关性（信息增益） |
| MI (S→Q) | {:.4f} | 非线性相关性（信息增益） |

### 5.2 判断标准

- **强相关** (|r| > 0.7): 两个分数高度冗余，移除一个影响不大
- **中等相关** (0.4 < |r| < 0.7): 部分冗余，需要权衡
- **弱相关/独立** (|r| < 0.4): 独立信息，两者互补

### 5.3 可视化文件

1. `correlation_analysis.png` - 散点图、密度图、分布图
2. `correlation_detailed_analysis.png` - 残差图、Q-Q图、指标汇总

## 6. 结论

基于相关性分析结果，本报告建议采取 **{decision}** 策略。

---
*报告由相关性验证脚本自动生成*
""".format(
            results['pearson_r'],
            results['spearman_r'],
            results['r2_quality_to_stability'],
            results['r2_stability_to_quality'],
            results['mi_quality_to_stability'],
            results['mi_stability_to_quality'],
            decision=results['recommendation']
        )
        
        # 保存报告
        report_path = self.output_dir / "correlation_validation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON结果
        json_path = self.output_dir / "correlation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 分析报告已保存: {report_path}")
        print(f"📊 JSON结果已保存: {json_path}")
    
    def run_validation(self):
        """运行完整的相关性验证流程"""
        print("\n" + "="*60)
        print("🚀 开始相关性验证实验")
        print("="*60)
        
        try:
            # 1. 加载测试数据
            spectra, wavelengths = self.load_test_data()
            
            # 2. 计算分数
            quality_scores, stability_scores = self.calculate_scores(spectra, wavelengths)
            
            # 3. 分析相关性
            results = self.analyze_correlation(quality_scores, stability_scores)
            
            # 4. 创建可视化
            self.create_visualizations(quality_scores, stability_scores, results)
            
            # 5. 生成报告
            self.generate_report(results, len(spectra))
            
            print("\n" + "="*60)
            print("🎉 相关性验证完成!")
            print(f"📁 所有结果已保存到: {self.output_dir}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"❌ 验证过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality Score与Stability Score相关性验证')
    project_root = Path(__file__).parent.parent
    parser.add_argument('--model-dir', type=str,
                       default=str(project_root / "models" / "DVP" / "v1.12_varsm_21k"),
                       help='模型目录')
    parser.add_argument('--test-data-npz', type=str,
                       default=str(project_root / "data" / "export_v11" / "test_subset.npz"),
                       help='测试数据NPZ路径')
    
    args = parser.parse_args()
    
    # 创建验证器并运行
    validator = ScoreCorrelationValidator(
        model_dir=args.model_dir,
        test_data_npz=args.test_data_npz
    )
    results = validator.run_validation()
    
    # 输出关键结论
    print("\n" + "="*60)
    print("📊 关键结论")
    print("="*60)
    print(f"Pearson相关系数: r = {results['pearson_r']:.4f}")
    print(f"相关性强度: {results['correlation_strength']}")
    print(f"\n{results['recommendation']}")
    print("="*60)


if __name__ == "__main__":
    main()

