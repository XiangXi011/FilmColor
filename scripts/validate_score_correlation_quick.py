#!/usr/bin/env python3
"""
Quality Score与Stability Score相关性验证实验 - 快速版本（无可视化）

目的：
1. 快速计算Quality Score和Stability Score的相关系数
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
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def load_models(model_dir: Path) -> Dict:
    """加载模型文件"""
    models = {}
    
    def _find_one(patterns):
        for pattern in patterns:
            matches = sorted(model_dir.glob(pattern))
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
    
    models['encoder'] = joblib.load(encoder_path)
    models['decoder'] = joblib.load(decoder_path)
    models['scaler'] = joblib.load(scaler_path)
    
    # 加载权重
    weights_file = _find_one([
        "weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"
    ])
    if weights_file and Path(weights_file).exists():
        models['weights'] = np.load(weights_file)
    else:
        # 创建默认权重
        data_loader = SpectrumDataLoader()
        wavelengths, _ = data_loader.load_dvp_standard_curve()
        weights = np.ones(len(wavelengths))
        peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
        weights[peak_mask] *= 1.5
        models['weights'] = weights
        print("[警告] 使用默认DVP权重向量")
    
    print("[完成] 模型文件加载成功")
    return models


def load_test_data(test_data_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载测试数据"""
    if test_data_npz and Path(test_data_npz).exists():
        print(f"[加载] 加载真实测试数据: {test_data_npz}")
        data = np.load(test_data_npz)
        spectra = data['dvp_values']
        wavelengths = data['wavelengths']
        print(f"[完成] 加载 {len(spectra)} 个测试样本")
        return spectra, wavelengths
    else:
        raise FileNotFoundError(f"测试数据不存在: {test_data_npz}")


def calculate_scores(spectra: np.ndarray, wavelengths: np.ndarray, 
                    models: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """计算Quality Score和Stability Score"""
    print("[计算] 计算Quality Score和Stability Score...")
    
    data_loader = SpectrumDataLoader()
    similarity_evaluator = SimilarityEvaluator()
    
    # 加载标准曲线
    _, standard_spectrum = data_loader.load_dvp_standard_curve()
    
    # 计算Quality Score
    quality_scores = []
    for spectrum in spectra:
        result = similarity_evaluator.evaluate(
            spectrum, standard_spectrum, wavelengths, coating_name="DVP"
        )
        quality_score = result['similarity_score']
        quality_scores.append(quality_score)
    
    quality_scores = np.array(quality_scores)
    
    # 计算Stability Score (重构误差)
    stability_scores = []
    for spectrum in spectra:
        # 数据预处理
        spectrum_scaled = models['scaler'].transform(spectrum.reshape(1, -1))
        
        # 通过编码器-解码器重构
        encoded = models['encoder'].predict(spectrum_scaled)
        decoded = models['decoder'].predict(encoded)
        
        # 计算重构误差
        spectrum_original = models['scaler'].inverse_transform(spectrum_scaled)
        reconstruction_error = np.mean(
            models['weights'] * (spectrum_original - decoded) ** 2
        )
        
        stability_scores.append(reconstruction_error)
    
    stability_scores = np.array(stability_scores)
    
    print(f"[完成] Score计算完成:")
    print(f"   - Quality Score范围: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")
    print(f"   - Stability Score范围: [{stability_scores.min():.6f}, {stability_scores.max():.6f}]")
    
    return quality_scores, stability_scores


def analyze_correlation(quality_scores: np.ndarray, stability_scores: np.ndarray) -> Dict:
    """分析相关性"""
    print("\n" + "="*60)
    print("[分析] 相关性分析")
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
        recommendation = "[建议移除] Stability Score（高度相关，不提供额外信息）"
        decision = "remove_stability"
    elif abs_pearson > 0.4:
        correlation_strength = "中等相关"
        recommendation = "[谨慎保留] Stability提供部分独立信息，但重叠较多，建议谨慎保留"
        decision = "cautious_keep"
    else:
        correlation_strength = "弱相关/独立"
        recommendation = "[建议保留] Stability Score（独立性好，提供额外信息）"
        decision = "keep_stability"
    
    print(f"   - Pearson |r| = {abs_pearson:.4f} → {correlation_strength}")
    print(f"   - Spearman |ρ| = {abs_spearman:.4f}")
    print(f"\n4. 决策建议:")
    print(f"   {recommendation}")
    
    # 4. 计算R²（决定系数）
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(quality_scores.reshape(-1, 1), stability_scores)
    r2_q_to_s = lr.score(quality_scores.reshape(-1, 1), stability_scores)
    
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
        'recommendation': recommendation,
        'n_samples': len(quality_scores)
    }
    
    return results


def generate_report(results: Dict, output_dir: Path):
    """生成分析报告"""
    print("\n[报告] 生成分析报告...")
    
    report = f"""# Quality Score与Stability Score相关性验证报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**样本数量**: {results['n_samples']}  

## 1. 核心结果

### Pearson相关系数
- r = {results['pearson_r']:.4f} (p-value = {results['pearson_p']:.4e})
- |r| = {abs(results['pearson_r']):.4f}
- **相关性强度**: {results['correlation_strength']}

### Spearman秩相关系数
- ρ = {results['spearman_r']:.4f} (p-value = {results['spearman_p']:.4e})
- |ρ| = {abs(results['spearman_r']):.4f}

## 2. 决定系数（R²）

| 方向 | R² | 解释能力 |
|------|-----|----------|
| Quality → Stability | {results['r2_quality_to_stability']:.4f} | Quality能解释Stability {results['r2_quality_to_stability']*100:.1f}%的变异 |
| Stability → Quality | {results['r2_stability_to_quality']:.4f} | Stability能解释Quality {results['r2_stability_to_quality']*100:.1f}%的变异 |

## 3. 互信息（非线性相关性）

- MI(Quality → Stability): {results['mi_quality_to_stability']:.4f}
- MI(Stability → Quality): {results['mi_stability_to_quality']:.4f}

## 4. 决策建议

**{results['recommendation']}**

**决策代码**: `{results['decision']}`

## 5. 下一步行动

"""
    
    if results['decision'] == 'remove_stability':
        report += """
### 方案A：移除Stability Score

**原因**: 与Quality高度相关（|r| > 0.7），不提供额外信息

**行动**:
1. 简化为Quality-Only架构
2. 直接投入半监督学习
3. 训练监督分类器替代Stability
4. 预期F1提升至0.80-0.85

**时间**: 1-2天
"""
    elif results['decision'] == 'keep_stability':
        report += """
### 方案B：保留Stability Score

**原因**: 与Quality独立性好（|r| < 0.4），提供额外信息

**行动**:
1. 保留双通道架构
2. 执行误报分析（理解75%误报率）
3. 半监督学习（三通道融合）
4. 预期F1提升至0.80-0.85

**时间**: 2-3天
"""
    else:  # cautious_keep
        report += """
### 方案C：谨慎保留（需要更多证据）

**原因**: 中等相关（0.4 < |r| < 0.7），需要误报分析决策

**行动**:
1. **立即执行误报分析**（4小时，P0优先级）
2. 根据误报原因决定：
   - 若误报主要是边界样本 → 移除Stability
   - 若误报是真实异常模式 → 保留Stability
3. 选择对应方案A或B

**时间**: 4小时分析 + 1-2天执行
"""
    
    report += """

## 6. 判断标准

- **强相关** (|r| > 0.7): 高度冗余，建议移除
- **中等相关** (0.4 < |r| < 0.7): 需要权衡，建议误报分析
- **弱相关/独立** (|r| < 0.4): 独立信息，建议保留

---
*报告由相关性验证脚本自动生成*
"""
    
    # 保存报告
    report_path = output_dir / "correlation_validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存JSON结果
    json_path = output_dir / "correlation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[保存] 分析报告已保存: {report_path}")
    print(f"[保存] JSON结果已保存: {json_path}")


def main():
    """主函数"""
    import argparse
    
    # 设置输出编码为UTF-8（Windows兼容）
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Quality Score与Stability Score相关性验证（快速版）')
    parser.add_argument('--model-dir', type=str,
                       default=str(project_root / "models" / "DVP" / "v1.12_varsm_21k"),
                       help='模型目录')
    parser.add_argument('--test-data-npz', type=str,
                       default=str(project_root / "data" / "export_v11" / "test_subset.npz"),
                       help='测试数据NPZ路径')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("[开始] 相关性验证实验（快速版，无可视化）")
    print("="*60)
    
    try:
        # 创建输出目录
        output_dir = project_root / "evaluation" / "correlation_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载模型
        models = load_models(Path(args.model_dir))
        
        # 2. 加载测试数据
        spectra, wavelengths = load_test_data(args.test_data_npz)
        
        # 3. 计算分数
        quality_scores, stability_scores = calculate_scores(spectra, wavelengths, models)
        
        # 4. 分析相关性
        results = analyze_correlation(quality_scores, stability_scores)
        
        # 5. 生成报告
        generate_report(results, output_dir)
        
        print("\n" + "="*60)
        print("[完成] 相关性验证完成!")
        print(f"[输出] 结果已保存到: {output_dir}")
        print("="*60)
        
        # 输出关键结论
        print("\n" + "="*60)
        print("[结论] 关键结论")
        print("="*60)
        print(f"Pearson相关系数: r = {results['pearson_r']:.4f}")
        print(f"相关性强度: {results['correlation_strength']}")
        print(f"\n{results['recommendation']}")
        print("="*60)
        
    except Exception as e:
        print(f"[错误] 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

