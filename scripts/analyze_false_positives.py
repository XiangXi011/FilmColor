#!/usr/bin/env python3
"""
Stability Score误报样本分析

目的：
1. 提取Stability Score的误报样本（预测为异常但实际正常）
2. 可视化光谱曲线，人工审查
3. 分类误报原因：边界样本/标注错误/正常波动
4. 理解75%误报率的根本原因

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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


class FalsePositiveAnalyzer:
    """误报样本分析器"""
    
    def __init__(self, model_dir: str, test_data_npz: str):
        """初始化分析器"""
        self.model_dir = Path(model_dir)
        self.test_data_npz = test_data_npz
        self.output_dir = project_root / "evaluation" / "false_positive_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.data_loader = SpectrumDataLoader()
        self.similarity_evaluator = SimilarityEvaluator()
        
        # 加载模型
        self.models = {}
        self._load_models()
        
        print(f"[初始化] 误报样本分析器初始化完成")
        print(f"[目录] 模型目录: {self.model_dir}")
        print(f"[目录] 输出目录: {self.output_dir}")
    
    def _load_models(self):
        """加载模型文件"""
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
            wavelengths, _ = self.data_loader.load_dvp_standard_curve()
            weights = np.ones(len(wavelengths))
            peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
            weights[peak_mask] *= 1.5
            self.models['weights'] = weights
        
        # 加载元数据
        metadata_path = _find_one([
            "*metadata*DVP*v*.json", "*metadata*DVP*.json", "*metadata*.json"
        ])
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print("[完成] 模型文件加载成功")
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载测试数据
        
        Returns:
            Tuple[spectra, wavelengths, quality_labels]
        """
        print(f"[加载] 加载测试数据: {self.test_data_npz}")
        data = np.load(self.test_data_npz)
        spectra = data['dvp_values']
        wavelengths = data['wavelengths']
        
        # 计算Quality标签（基于相似度评估）
        # 注意：这里假设测试数据都是正常样本（来自真实生产数据）
        # 实际异常需要通过Quality Score来判断
        print(f"[完成] 加载 {len(spectra)} 个测试样本")
        
        return spectra, wavelengths, None
    
    def calculate_scores_and_labels(self, spectra: np.ndarray, wavelengths: np.ndarray) -> Tuple:
        """
        计算Quality Score和Stability Score，并生成标签
        
        Returns:
            Tuple[quality_scores, stability_scores, quality_labels, stability_labels]
        """
        print("[计算] 计算Quality Score和Stability Score...")
        
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
            spectrum_scaled = self.models['scaler'].transform(spectrum.reshape(1, -1))
            encoded = self.models['encoder'].predict(spectrum_scaled)
            decoded = self.models['decoder'].predict(encoded)
            spectrum_original = self.models['scaler'].inverse_transform(spectrum_scaled)
            reconstruction_error = np.mean(
                self.models['weights'] * (spectrum_original - decoded) ** 2
            )
            stability_scores.append(reconstruction_error)
        
        stability_scores = np.array(stability_scores)
        
        # 确定阈值并生成标签
        # Quality阈值：使用测试数据的P5分位数（保守估计，5%最低质量样本为异常）
        quality_threshold_percentile = 5
        quality_threshold = np.percentile(quality_scores, quality_threshold_percentile)
        print(f"[阈值] 使用统计阈值:")
        print(f"   - Quality: P{quality_threshold_percentile} = {quality_threshold:.4f}")
        
        # Stability阈值：高于此值为异常
        # 使用95%分位数（与测试数据分布一致）
        stability_threshold_percentile = 95
        stability_threshold = np.percentile(stability_scores, stability_threshold_percentile)
        print(f"   - Stability: P{stability_threshold_percentile} = {stability_threshold:.6f}")
        
        # 生成标签
        quality_labels = (quality_scores < quality_threshold).astype(int)
        stability_labels = (stability_scores > stability_threshold).astype(int)
        
        print(f"\n[完成] Score计算完成:")
        print(f"   - Quality Score范围: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")
        print(f"   - Stability Score范围: [{stability_scores.min():.6f}, {stability_scores.max():.6f}]")
        print(f"\n[标签] 异常判定结果:")
        print(f"   - Quality异常: {quality_labels.sum()} / {len(quality_labels)} ({quality_labels.mean()*100:.1f}%)")
        print(f"   - Stability异常: {stability_labels.sum()} / {len(stability_labels)} ({stability_labels.mean()*100:.1f}%)")
        
        return quality_scores, stability_scores, quality_labels, stability_labels, quality_threshold, stability_threshold
    
    def identify_false_positives(self, quality_labels: np.ndarray, stability_labels: np.ndarray) -> np.ndarray:
        """
        识别Stability误报样本
        
        误报定义：Stability预测为异常(1)，但Quality判断为正常(0)
        
        Returns:
            误报样本的索引数组
        """
        # Stability误报：Stability认为异常，但Quality认为正常
        false_positive_mask = (stability_labels == 1) & (quality_labels == 0)
        false_positive_indices = np.where(false_positive_mask)[0]
        
        # 真阳性：两者都认为异常
        true_positive_mask = (stability_labels == 1) & (quality_labels == 1)
        true_positive_indices = np.where(true_positive_mask)[0]
        
        # 假阴性（Stability漏报）：Quality认为异常，但Stability认为正常
        false_negative_mask = (stability_labels == 0) & (quality_labels == 1)
        false_negative_indices = np.where(false_negative_mask)[0]
        
        # 真阴性：两者都认为正常
        true_negative_mask = (stability_labels == 0) & (quality_labels == 0)
        true_negative_indices = np.where(true_negative_mask)[0]
        
        print(f"\n[分类] 样本分类统计:")
        print(f"   - 真阴性(TN): {len(true_negative_indices)} ({len(true_negative_indices)/len(quality_labels)*100:.1f}%)")
        print(f"   - 假阳性(FP-Stability误报): {len(false_positive_indices)} ({len(false_positive_indices)/len(quality_labels)*100:.1f}%)")
        print(f"   - 假阴性(FN-Stability漏报): {len(false_negative_indices)} ({len(false_negative_indices)/len(quality_labels)*100:.1f}%)")
        print(f"   - 真阳性(TP): {len(true_positive_indices)} ({len(true_positive_indices)/len(quality_labels)*100:.1f}%)")
        
        # 计算Stability的精确率和召回率（以Quality为ground truth）
        if len(true_positive_indices) + len(false_positive_indices) > 0:
            precision = len(true_positive_indices) / (len(true_positive_indices) + len(false_positive_indices))
        else:
            precision = 0.0
        
        if len(true_positive_indices) + len(false_negative_indices) > 0:
            recall = len(true_positive_indices) / (len(true_positive_indices) + len(false_negative_indices))
        else:
            recall = 0.0
        
        print(f"\n[性能] Stability相对于Quality:")
        print(f"   - 精确率: {precision:.3f} (误报率: {1-precision:.3f})")
        print(f"   - 召回率: {recall:.3f} (漏检率: {1-recall:.3f})")
        
        return false_positive_indices
    
    def extract_false_positive_samples(self, spectra: np.ndarray, wavelengths: np.ndarray,
                                      quality_scores: np.ndarray, stability_scores: np.ndarray,
                                      false_positive_indices: np.ndarray, top_k: int = 200) -> pd.DataFrame:
        """
        提取误报样本的详细信息
        
        Args:
            top_k: 提取前K个误报样本（按重构误差排序）
        """
        print(f"\n[提取] 提取Top {top_k}个误报样本...")
        
        # 按Stability Score（重构误差）降序排序
        sorted_indices = false_positive_indices[np.argsort(-stability_scores[false_positive_indices])]
        selected_indices = sorted_indices[:min(top_k, len(sorted_indices))]
        
        # 构建DataFrame
        samples = []
        for idx in selected_indices:
            sample = {
                'sample_idx': int(idx),
                'quality_score': float(quality_scores[idx]),
                'stability_score': float(stability_scores[idx]),
                'spectrum': spectra[idx].tolist()
            }
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        print(f"[完成] 提取了 {len(df)} 个误报样本")
        if len(df) > 0:
            print(f"   - Quality Score范围: [{df['quality_score'].min():.4f}, {df['quality_score'].max():.4f}]")
            print(f"   - Stability Score范围: [{df['stability_score'].min():.6f}, {df['stability_score'].max():.6f}]")
        else:
            print(f"   [警告] 没有误报样本！这可能说明：")
            print(f"     1. Quality阈值设置过于宽松（大部分样本被判为异常）")
            print(f"     2. Stability阈值设置过于严格（很少样本被判为异常）")
            print(f"     3. 两个分数高度一致")
        
        return df
    
    def analyze_false_positive_patterns(
        self,
        df: pd.DataFrame,
        wavelengths: np.ndarray,
        quality_threshold: float,
        stability_threshold: float,
        quality_decision_threshold: float = 0.80
    ) -> Dict:
        """
        分析误报样本的特征模式
        """
        if len(df) == 0:
            print(f"\n[跳过] 没有误报样本，跳过特征分析")
            return {}
        
        print(f"\n[分析] 分析误报样本特征模式...")
        
        spectra = np.array(df['spectrum'].tolist())
        quality_scores = df['quality_score'].values
        stability_scores = df['stability_score'].values
        
        # 1. Quality Score分布分析
        q_mean = quality_scores.mean()
        q_std = quality_scores.std()
        q_percentiles = np.percentile(quality_scores, [25, 50, 75])
        
        print(f"\n1. Quality Score分布:")
        print(f"   - 均值 ± 标准差: {q_mean:.4f} ± {q_std:.4f}")
        print(f"   - 分位数(P25/P50/P75): {q_percentiles[0]:.4f} / {q_percentiles[1]:.4f} / {q_percentiles[2]:.4f}")
        print(f"   - 质量阈值: {quality_threshold:.4f}")
        print(f"   - 生产判定阈值: {quality_decision_threshold:.4f}")

        # 1.1 Quality区间分类
        band_counts = {
            'below_quality_threshold': int(np.sum(quality_scores < quality_threshold)),
            'edge_band': int(np.sum((quality_scores >= quality_threshold) & (quality_scores < quality_decision_threshold))),
            'above_decision_threshold': int(np.sum(quality_scores >= quality_decision_threshold))
        }
        total = len(quality_scores) if len(quality_scores) > 0 else 1
        print("   - 区间分布:")
        print(f"     <质量阈值({quality_threshold:.4f}): {band_counts['below_quality_threshold']} ({band_counts['below_quality_threshold']/total*100:.1f}%)")
        print(f"     边界区[{quality_threshold:.4f}, {quality_decision_threshold:.4f}): {band_counts['edge_band']} ({band_counts['edge_band']/total*100:.1f}%)")
        print(f"     ≥生产阈值({quality_decision_threshold:.4f}): {band_counts['above_decision_threshold']} ({band_counts['above_decision_threshold']/total*100:.1f}%)")
        
        # 2. Stability Score分布分析
        s_mean = stability_scores.mean()
        s_std = stability_scores.std()
        s_percentiles = np.percentile(stability_scores, [25, 50, 75])
        
        print(f"\n2. Stability Score分布:")
        print(f"   - 均值 ± 标准差: {s_mean:.6f} ± {s_std:.6f}")
        print(f"   - 分位数(P25/P50/P75): {s_percentiles[0]:.6f} / {s_percentiles[1]:.6f} / {s_percentiles[2]:.6f}")
        print(f"   - 阈值: {stability_threshold:.6f}")
        
        # 3. 光谱特征分析
        mean_spectrum = spectra.mean(axis=0)
        std_spectrum = spectra.std(axis=0)
        
        # 加载标准曲线对比
        _, standard_spectrum = self.data_loader.load_dvp_standard_curve()
        
        # 计算与标准曲线的差异
        diff_from_standard = mean_spectrum - standard_spectrum
        max_diff_idx = np.argmax(np.abs(diff_from_standard))
        max_diff_wavelength = wavelengths[max_diff_idx]
        max_diff_value = diff_from_standard[max_diff_idx]
        
        print(f"\n3. 光谱特征分析:")
        print(f"   - 平均光谱范围: [{mean_spectrum.min():.4f}, {mean_spectrum.max():.4f}]")
        print(f"   - 光谱标准差范围: [{std_spectrum.min():.4f}, {std_spectrum.max():.4f}]")
        print(f"   - 与标准曲线最大差异: {max_diff_value:.4f} @ {max_diff_wavelength:.0f}nm")
        
        # 4. 分段分析（DVP关键波段）
        bands = [
            (380, 450, "UV-Blue"),
            (450, 500, "Blue-Green"),
            (500, 600, "Green-Yellow"),
            (600, 780, "Red-IR")
        ]
        
        print(f"\n4. 分段特征分析:")
        for low, high, name in bands:
            mask = (wavelengths >= low) & (wavelengths <= high)
            band_mean_diff = np.mean(np.abs(diff_from_standard[mask]))
            band_std = np.mean(std_spectrum[mask])
            print(f"   - {name} ({low}-{high}nm):")
            print(f"     平均差异: {band_mean_diff:.4f}, 标准差: {band_std:.4f}")
        
        results = {
            'quality_score': {
                'mean': float(q_mean),
                'std': float(q_std),
                'percentiles': {25: float(q_percentiles[0]), 50: float(q_percentiles[1]), 75: float(q_percentiles[2])}
            },
            'quality_threshold': float(quality_threshold),
            'quality_decision_threshold': float(quality_decision_threshold),
            'quality_bands': {
                'below_quality_threshold': {
                    'count': band_counts['below_quality_threshold'],
                    'ratio': band_counts['below_quality_threshold'] / total
                },
                'edge_band': {
                    'range': [float(quality_threshold), float(quality_decision_threshold)],
                    'count': band_counts['edge_band'],
                    'ratio': band_counts['edge_band'] / total
                },
                'above_decision_threshold': {
                    'count': band_counts['above_decision_threshold'],
                    'ratio': band_counts['above_decision_threshold'] / total
                }
            },
            'stability_score': {
                'mean': float(s_mean),
                'std': float(s_std),
                'percentiles': {25: float(s_percentiles[0]), 50: float(s_percentiles[1]), 75: float(s_percentiles[2])}
            },
            'stability_threshold': float(stability_threshold),
            'spectrum': {
                'max_diff_wavelength': float(max_diff_wavelength),
                'max_diff_value': float(max_diff_value)
            }
        }
        
        return results
    
    def generate_report(self, results: Dict, n_total: int, n_false_positive: int):
        """生成分析报告"""
        print(f"\n[报告] 生成误报分析报告...")
        
        false_positive_rate = n_false_positive / n_total if n_total > 0 else 0
        
        report = f"""# Stability Score误报样本分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**总样本数**: {n_total}  
**误报样本数**: {n_false_positive}  
**误报率**: {false_positive_rate*100:.1f}%

## 1. 分析目的

理解Stability Score误报样本的特征，回答以下问题：
1. 误报样本的Quality Score分布如何？是否集中在边界区域？
2. 误报样本的光谱特征有何特点？
3. 误报的根本原因是什么？

## 2. 误报样本Quality Score分布

| 统计量 | 值 |
|-------|-----|
| 均值 | {results['quality_score']['mean']:.4f} |
| 标准差 | {results['quality_score']['std']:.4f} |
| P25 | {results['quality_score']['percentiles'][25]:.4f} |
| P50 (中位数) | {results['quality_score']['percentiles'][50]:.4f} |
| P75 | {results['quality_score']['percentiles'][75]:.4f} |
| 质量阈值 (统计) | {results['quality_threshold']:.4f} |
| 生产判定阈值 | {results['quality_decision_threshold']:.4f} |

### 2.1 质量区间分布

| 区间 | 样本数 | 占误报比例 |
|------|--------|------------|
| < 质量阈值 | {results['quality_bands']['below_quality_threshold']['count']} | {results['quality_bands']['below_quality_threshold']['ratio']*100:.1f}% |
| [质量阈值, 生产阈值) | {results['quality_bands']['edge_band']['count']} | {results['quality_bands']['edge_band']['ratio']*100:.1f}% |
| ≥ 生产阈值 | {results['quality_bands']['above_decision_threshold']['count']} | {results['quality_bands']['above_decision_threshold']['ratio']*100:.1f}% |

**分析**：
- 如果均值接近阈值（~0.80），说明误报集中在边界样本
- 如果均值远高于阈值，说明误报是真正的正常样本

## 3. 误报样本Stability Score分布

| 统计量 | 值 |
|-------|-----|
| 均值 | {results['stability_score']['mean']:.6f} |
| 标准差 | {results['stability_score']['std']:.6f} |
| P25 | {results['stability_score']['percentiles'][25]:.6f} |
| P50 (中位数) | {results['stability_score']['percentiles'][50]:.6f} |
| P75 | {results['stability_score']['percentiles'][75]:.6f} |
| 阈值 (统计) | {results['stability_threshold']:.6f} |

## 4. 光谱特征分析

**与标准曲线最大差异**:
- 波长: {results['spectrum']['max_diff_wavelength']:.0f}nm
- 差异值: {results['spectrum']['max_diff_value']:.4f}

## 5. 误报原因假设

基于当前分析，可能的误报原因：

### 假设1：训练数据过于纯净（P85-P100）

**解释**: 
- 自编码器训练数据为P85-P100的高质量样本（83.5%高质量）
- 测试数据包含P0-P100的全部样本
- P0-P85的正常样本（虽然质量略低但仍合格）被自编码器视为"未见过的模式"
- 导致重构误差偏大，被误判为异常

**验证方法**: 
- 检查误报样本的Quality Score分布
- 如果集中在P75-P85区间，说明这是边界样本问题

### 假设2：Stability Score本质是"新颖性检测"

**解释**:
- Stability Score实际衡量的是"与训练数据的相似度"
- 不等于"真实的质量稳定性"
- 重构误差大只说明"与训练数据不同"，不说明"有质量问题"

**验证方法**:
- 比较误报样本与标准曲线的差异
- 如果差异很小，说明样本本身是正常的

### 假设3：Stability和Quality捕获不同的异常模式

**解释**:
- Quality关注"形状相似性"（峰值位置、曲线形态）
- Stability关注"统计一致性"（整体分布、噪声水平）
- 两者确实互补，但Stability更敏感（高召回率，低精确率）

**验证方法**:
- 人工审查误报样本，判断是否有潜在的稳定性问题
- 如果有部分样本确实有轻微异常，说明Stability提供了额外信息

## 6. 下一步行动

根据分析结果，建议：

1. **如果误报主要是边界样本（P75-P85）**:
   - 原因：训练数据太纯净
   - 建议：扩大训练数据范围（P75-P100或P70-P100）
   - 或者：接受当前精确率，利用Stability的高召回率特性

2. **如果误报是真正的正常样本**:
   - 原因：Stability Score本质是新颖性检测
   - 建议：调整Stability权重（降低影响）
   - 或者：通过半监督学习引入Residual Classifier

3. **人工标注验证**:
   - 从200个误报样本中随机抽取50个
   - 3名专家独立标注
   - 计算标注一致性（Cohen's Kappa）
   - 确定"真实误报"vs"标注错误"的比例

---
*报告由误报分析脚本自动生成*
"""
        
        # 保存报告
        report_path = self.output_dir / "false_positive_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON结果
        json_path = self.output_dir / "false_positive_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'n_total': n_total,
                'n_false_positive': n_false_positive,
                'false_positive_rate': false_positive_rate,
                **results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[保存] 分析报告已保存: {report_path}")
        print(f"[保存] JSON结果已保存: {json_path}")
    
    def export_false_positive_samples(
        self,
        df: pd.DataFrame,
        wavelengths: np.ndarray,
        review_sample_size: int = 30,
        random_state: int = 42
    ):
        """导出误报样本供人工审查"""
        # 导出全量Top-K
        csv_path = self.output_dir / "false_positive_samples.csv"
        df[['sample_idx', 'quality_score', 'stability_score']].to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        spectra = np.array(df['spectrum'].tolist())
        npz_path = self.output_dir / "false_positive_samples.npz"
        np.savez(
            npz_path,
            spectra=spectra,
            wavelengths=wavelengths,
            sample_indices=df['sample_idx'].values,
            quality_scores=df['quality_score'].values,
            stability_scores=df['stability_score'].values
        )
        
        # 生成人工复核采样集（默认30条，若不足则全量）
        if len(df) > 0:
            review_size = min(review_sample_size, len(df))
            review_df = df.sample(review_size, random_state=random_state)
            review_csv = self.output_dir / "false_positive_samples_review.csv"
            review_df[['sample_idx', 'quality_score', 'stability_score']].to_csv(review_csv, index=False, encoding='utf-8-sig')
            review_npz = self.output_dir / "false_positive_samples_review.npz"
            review_spectra = np.array(review_df['spectrum'].tolist())
            np.savez(
                review_npz,
                spectra=review_spectra,
                wavelengths=wavelengths,
                sample_indices=review_df['sample_idx'].values,
                quality_scores=review_df['quality_score'].values,
                stability_scores=review_df['stability_score'].values
            )
            print(f"[导出] 复核采样集: {review_size} 条")
            print(f"   - CSV: {review_csv}")
            print(f"   - NPZ: {review_npz}")
        
        print(f"[导出] 误报样本已导出:")
        print(f"   - CSV: {csv_path}")
        print(f"   - NPZ: {npz_path}")
    
    def run_analysis(self, top_k: int = 200):
        """运行完整的误报分析"""
        print("\n" + "="*60)
        print("[开始] Stability Score误报样本分析")
        print("="*60)
        
        try:
            # 1. 加载数据
            spectra, wavelengths, _ = self.load_test_data()
            
            # 2. 计算分数和标签
            quality_scores, stability_scores, quality_labels, stability_labels, \
                quality_threshold, stability_threshold = self.calculate_scores_and_labels(spectra, wavelengths)
            
            # 3. 识别误报样本
            false_positive_indices = self.identify_false_positives(quality_labels, stability_labels)
            
            # 4. 提取误报样本
            df = self.extract_false_positive_samples(
                spectra, wavelengths, quality_scores, stability_scores,
                false_positive_indices, top_k=top_k
            )
            
            # 5. 分析误报特征
            if len(df) > 0:
                results = self.analyze_false_positive_patterns(
                    df,
                    wavelengths,
                    quality_threshold=quality_threshold,
                    stability_threshold=stability_threshold
                )
                
                # 6. 生成报告
                self.generate_report(results, len(spectra), len(false_positive_indices))
                
                # 7. 导出样本供人工审查
                self.export_false_positive_samples(df, wavelengths)
            else:
                print(f"\n[结论] 没有误报样本！")
                print(f"   这说明Stability Score与Quality Score高度一致（或阈值设置导致）")
                print(f"   建议检查阈值设置是否合理")
            
            print("\n" + "="*60)
            print("[完成] 误报分析完成!")
            print(f"[输出] 结果已保存到: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"[错误] 分析过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    import argparse
    
    # 设置输出编码为UTF-8（Windows兼容）
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Stability Score误报样本分析')
    parser.add_argument('--model-dir', type=str,
                       default=str(project_root / "models" / "DVP" / "v1.12_varsm_21k"),
                       help='模型目录')
    parser.add_argument('--test-data-npz', type=str,
                       default=str(project_root / "data" / "export_v11" / "test_subset.npz"),
                       help='测试数据NPZ路径')
    parser.add_argument('--top-k', type=int, default=200,
                       help='提取前K个误报样本（默认200）')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = FalsePositiveAnalyzer(
        model_dir=args.model_dir,
        test_data_npz=args.test_data_npz
    )
    analyzer.run_analysis(top_k=args.top_k)


if __name__ == "__main__":
    main()

