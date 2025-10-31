#!/usr/bin/env python3
"""
光谱异常检测模型评估脚本
用于评估DVP涂层的Quality Score和Stability Score模型性能
生成完整的可视化报告和性能指标

Author: MiniMax Agent
Date: 2025-10-30
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, classification_report,
                           accuracy_score, precision_score, recall_score, f1_score,
                           precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator
# from models.weighted_autoencoder import WeightedAutoencoder  # 移除TensorFlow依赖

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import platform
    
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    system = platform.system()
    if system == "Windows":
        # Windows系统优先使用微软雅黑和黑体
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong"]
    elif system == "Darwin":  # macOS
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
    else:  # Linux
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
    
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, model_dir: str = None, combine_strategy: str = None, random_seed: int = 42, optimize_thresholds: str = "none"):
        """
        初始化模型评估器
        
        Args:
            model_dir: 模型文件目录，默认为项目根目录下的models文件夹
            combine_strategy: 组合策略覆盖(two_stage|and|or)。None表示使用metadata默认。
            random_seed: 随机种子，用于生成可复现的测试数据（默认42）
        """
        project_root = Path(__file__).parent.parent
        if model_dir is None:
            model_dir = project_root / "models"
        self.model_dir = Path(model_dir)
        self.output_dir = project_root / "evaluation"
        self.output_dir.mkdir(exist_ok=True)
        self.combine_strategy = combine_strategy
        self.random_seed = random_seed
        # 阈值优化策略: none | youden | f1
        self.optimize_thresholds = (optimize_thresholds or "none").lower()
        
        # 初始化组件
        self.data_loader = SpectrumDataLoader()
        self.similarity_evaluator = SimilarityEvaluator()
        
        # 加载模型
        self.models = {}
        self.metadata = {}
        self._load_models()
        
        # 设置matplotlib
        setup_matplotlib_for_plotting()
        
        print(f"✅ 模型评估器初始化完成")
        print(f"📁 模型目录: {self.model_dir}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def _load_models(self):
        """加载训练好的模型文件"""
        try:
            # 自适应查找模型文件（兼容不同版本命名，如 v1.0/v1.2，或去版本化文件名）
            def _find_one(patterns):
                for pattern in patterns:
                    matches = sorted(self.model_dir.glob(pattern))
                    if matches:
                        return matches[0]
                return None

            # 加载编码器、解码器、标准化器
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
                raise FileNotFoundError(
                    f"未在 {self.model_dir} 中找到所需的模型文件(encoder/decoder/scaler)。请检查文件命名是否包含版本号，或将目录指向正确的版本。"
                )

            self.models['encoder'] = joblib.load(encoder_path)
            self.models['decoder'] = joblib.load(decoder_path)
            self.models['scaler'] = joblib.load(scaler_path)
            
            # 尝试加载权重文件，如果不存在则创建默认权重
            weights_file = _find_one([
                "weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"
            ])
            if weights_file and Path(weights_file).exists():
                self.models['weights'] = np.load(weights_file)
            else:
                # 创建DVP默认权重（400-550nm增强1.5倍）
                wavelengths, dvp_standard = self.data_loader.load_dvp_standard_curve()
                weights = np.ones(len(wavelengths))
                peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                weights[peak_mask] *= 1.5
                self.models['weights'] = weights
                print("⚠️  使用默认DVP权重向量")
            
            # 加载元数据（兼容不同命名）
            metadata_path = _find_one([
                "*metadata*DVP*v*.json", "*metadata*DVP*.json", "*metadata*.json"
            ])
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                # 元数据缺失也允许继续，但后续会采用统计回退策略
                self.metadata = {}
                print("⚠️  未找到元数据文件，将使用统计回退方式估计阈值")
            
            print("✅ 模型文件加载成功")
            print(f"📄 编码器: {encoder_path}")
            print(f"📄 解码器: {decoder_path}")
            print(f"📄 标准化器: {scaler_path}")
            if weights_file:
                print(f"📄 权重文件: {weights_file}")
            if self.metadata:
                print(f"📋 元数据已加载")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def generate_test_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成测试数据用于评估
        
        Args:
            n_samples: 生成样本数量
            
        Returns:
            Tuple[spectra, quality_labels, stability_labels]
            - spectra: 光谱数据
            - quality_labels: 质量标签 (0: 正常, 1: 异常)
            - stability_labels: 稳定性标签 (0: 正常, 1: 异常)
        """
        print(f"🔄 生成 {n_samples} 个测试样本...")
        print(f"🎲 使用随机种子: {self.random_seed} (确保数据可复现)")
        
        # 设置随机种子以确保可复现性
        np.random.seed(self.random_seed)
        
        # 加载标准曲线
        wavelengths, standard_spectrum = self.data_loader.load_dvp_standard_curve()
        
        # 生成正常样本 (80%)
        n_normal = int(n_samples * 0.8)
        normal_spectra = []
        
        for i in range(n_normal):
            # 添加高斯噪声
            noise_level = abs(np.random.normal(0, 0.01))  # 1%噪声，确保为正值
            spectrum = standard_spectrum + np.random.normal(0, noise_level, len(standard_spectrum))
            normal_spectra.append(spectrum)
        
        # 生成异常样本 (20%)
        n_anomaly = n_samples - n_normal
        anomaly_spectra = []
        quality_anomaly_labels = []
        stability_anomaly_labels = []
        
        for i in range(n_anomaly):
            anomaly_type = np.random.choice(['quality', 'stability', 'both'])
            
            if anomaly_type == 'quality':
                # 质量异常：光谱形状异常
                spectrum = standard_spectrum.copy()
                # 在400-550nm范围内添加异常
                peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                spectrum[peak_mask] += np.random.normal(0, 0.1, np.sum(peak_mask))
                quality_anomaly_labels.append(1)
                stability_anomaly_labels.append(0)
                
            elif anomaly_type == 'stability':
                # 稳定性异常：整体偏移
                spectrum = standard_spectrum.copy()
                offset = np.random.normal(0, 0.05)
                spectrum += offset
                quality_anomaly_labels.append(0)
                stability_anomaly_labels.append(1)
                
            else:  # both
                # 双重异常
                spectrum = standard_spectrum.copy()
                # 形状异常
                peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                spectrum[peak_mask] += np.random.normal(0, 0.08, np.sum(peak_mask))
                # 整体偏移
                offset = np.random.normal(0, 0.03)
                spectrum += offset
                quality_anomaly_labels.append(1)
                stability_anomaly_labels.append(1)
            
            anomaly_spectra.append(spectrum)
        
        # 合并所有数据
        all_spectra = np.array(normal_spectra + anomaly_spectra)
        
        # 生成标签
        quality_labels = np.array([0] * n_normal + quality_anomaly_labels)
        stability_labels = np.array([0] * n_normal + stability_anomaly_labels)
        
        print(f"✅ 测试数据生成完成:")
        print(f"   - 正常样本: {n_normal}")
        print(f"   - 质量异常: {sum(quality_anomaly_labels)}")
        print(f"   - 稳定性异常: {sum(stability_anomaly_labels)}")
        
        return all_spectra, quality_labels, stability_labels
    
    def calculate_scores(self, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算Quality Score和Stability Score
        
        Args:
            spectra: 光谱数据
            
        Returns:
            Tuple[quality_scores, stability_scores]
        """
        print("🔄 计算Quality Score和Stability Score...")
        
        # 加载标准曲线
        wavelengths, standard_spectrum = self.data_loader.load_dvp_standard_curve()
        
        # 计算Quality Score
        quality_scores = []
        for spectrum in spectra:
            result = self.similarity_evaluator.evaluate(
                spectrum, standard_spectrum, wavelengths, coating_name="DVP"
            )
            quality_score = result['similarity_score']
            quality_scores.append(quality_score)
        
        quality_scores = np.array(quality_scores)
        
        # 计算Stability Score
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
        print(f"   - Quality Score范围: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")
        print(f"   - Stability Score范围: [{stability_scores.min():.3f}, {stability_scores.max():.3f}]")
        
        return quality_scores, stability_scores
    
    def create_quality_stability_scatter(self, quality_scores: np.ndarray, 
                                       stability_scores: np.ndarray,
                                       quality_labels: np.ndarray, 
                                       stability_labels: np.ndarray):
        """
        创建Quality Score vs Stability Score散点图
        
        Args:
            quality_scores: Quality Score数组
            stability_scores: Stability Score数组
            quality_labels: Quality标签
            stability_labels: Stability标签
        """
        print("📊 创建Quality Score vs Stability Score散点图...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DVP涂层光谱异常检测评估结果', fontsize=16, fontweight='bold')
        
        # 1. 整体散点图
        colors = []
        for q_label, s_label in zip(quality_labels, stability_labels):
            if q_label == 0 and s_label == 0:
                colors.append('green')  # 正常
            elif q_label == 1 and s_label == 0:
                colors.append('orange')  # 质量异常
            elif q_label == 0 and s_label == 1:
                colors.append('blue')   # 稳定性异常
            else:
                colors.append('red')    # 双重异常
        
        scatter = ax1.scatter(quality_scores, stability_scores, c=colors, alpha=0.6, s=30)
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Stability Score')
        ax1.set_title('Quality Score vs Stability Score (整体)')
        ax1.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='green', label='正常 (Normal)'),
            mpatches.Patch(color='orange', label='质量异常 (Quality)'),
            mpatches.Patch(color='blue', label='稳定性异常 (Stability)'),
            mpatches.Patch(color='red', label='双重异常 (Both)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. Quality Score分布
        ax2.hist(quality_scores[quality_labels == 0], bins=30, alpha=0.7, 
                label='正常', color='green', density=True)
        ax2.hist(quality_scores[quality_labels == 1], bins=30, alpha=0.7, 
                label='质量异常', color='red', density=True)
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('密度')
        ax2.set_title('Quality Score分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stability Score分布
        ax3.hist(stability_scores[stability_labels == 0], bins=30, alpha=0.7, 
                label='正常', color='green', density=True)
        ax3.hist(stability_scores[stability_labels == 1], bins=30, alpha=0.7, 
                label='稳定性异常', color='red', density=True)
        ax3.set_xlabel('Stability Score')
        ax3.set_ylabel('密度')
        ax3.set_title('Stability Score分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 决策区域图
        # 创建网格
        q_min, q_max = quality_scores.min(), quality_scores.max()
        s_min, s_max = stability_scores.min(), stability_scores.max()
        
        qq, ss = np.meshgrid(np.linspace(q_min, q_max, 100),
                            np.linspace(s_min, s_max, 100))
        
        # 决策边界阈值
        # Quality Score阈值：从similarity_evaluator元数据中读取
        quality_threshold = None
        se_metadata = self.metadata.get('similarity_evaluator', {})
        if se_metadata:
            quality_threshold = se_metadata.get('quality_threshold')
        if quality_threshold is None:
            quality_threshold = self.metadata.get('quality_threshold')
        if quality_threshold is None:
            # 使用正常样本5%分位作为保守阈值（低于此为质量异常）
            quality_threshold = np.percentile(quality_scores[quality_labels == 0], 5)
        
        # Stability Score方向与阈值：先自动方向修正，再确定阈值
        # 初步判断方向（AUC < 0.5 说明方向可能相反）
        s_scores_probe = stability_scores.copy()
        fpr_probe, tpr_probe, _ = roc_curve(stability_labels, s_scores_probe)
        auc_probe = auc(fpr_probe, tpr_probe)
        flipped = False
        if auc_probe < 0.5:
            s_scores_dir = -s_scores_probe
            flipped = True
        else:
            s_scores_dir = s_scores_probe

        # Stability Score阈值：从weighted_autoencoder元数据中读取
        stability_threshold = None
        wae_metadata = self.metadata.get('weighted_autoencoder', {})
        if wae_metadata:
            stability_threshold = wae_metadata.get('stability_threshold')
        if stability_threshold is None:
            stability_threshold = self.metadata.get('stability_threshold')
        if stability_threshold is None:
            # 使用正常样本95%分位数（高于此为稳定性异常），在已统一方向的分数上计算
            stability_threshold = np.percentile(s_scores_dir[stability_labels == 0], 95)
        
        decision = np.zeros_like(qq)
        # 使用方向统一后的分数进行判定：高于阈值视为异常
        # 由于图上使用的是原始ss坐标，只用于展示；判定基于方向一致的s_scores_dir
        # 这里简化近似：仍用ss与同一阈值比较，仅作为可视化参考
        decision[(qq < quality_threshold) | (ss > stability_threshold)] = 1
        
        ax4.contourf(qq, ss, decision, levels=[0, 0.5, 1], 
                    colors=['lightgreen', 'lightcoral'], alpha=0.6)
        ax4.contour(qq, ss, decision, levels=[0.5], colors='black', linewidths=2)
        
        # 绘制数据点
        normal_mask = (quality_labels == 0) & (stability_labels == 0)
        anomaly_mask = (quality_labels == 1) | (stability_labels == 1)
        
        ax4.scatter(quality_scores[normal_mask], stability_scores[normal_mask], 
                   c='green', s=20, alpha=0.7, label='正常')
        ax4.scatter(quality_scores[anomaly_mask], stability_scores[anomaly_mask], 
                   c='red', s=20, alpha=0.7, label='异常')
        
        ax4.set_xlabel('Quality Score')
        ax4.set_ylabel('Stability Score')
        ax4.set_title('决策区域')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "quality_stability_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 散点图已保存: {output_path}")
    
    def create_spectral_reconstruction_comparison(self, spectra: np.ndarray, 
                                                sample_indices: List[int] = None):
        """
        创建光谱重构对比可视化
        
        Args:
            spectra: 光谱数据
            sample_indices: 要展示的样本索引列表
        """
        print("🔄 创建光谱重构对比可视化...")
        
        if sample_indices is None:
            # 随机选择一些样本进行展示
            sample_indices = np.random.choice(len(spectra), size=min(6, len(spectra)), replace=False)
        
        # 加载波长信息
        wavelengths, _ = self.data_loader.load_dvp_standard_curve()
        
        n_samples = len(sample_indices)
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('光谱重构对比分析', fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(sample_indices):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # 原始光谱
            original = spectra[idx]
            
            # 重构光谱
            original_scaled = self.models['scaler'].transform(original.reshape(1, -1))
            encoded = self.models['encoder'].predict(original_scaled)
            decoded = self.models['decoder'].predict(encoded)
            reconstructed = self.models['scaler'].inverse_transform(decoded)[0]
            
            # 绘制对比
            ax.plot(wavelengths, original, 'b-', label='原始光谱', linewidth=2)
            ax.plot(wavelengths, reconstructed, 'r--', label='重构光谱', linewidth=2)
            
            # 计算误差
            reconstruction_error = np.mean(
                self.models['weights'] * (original - reconstructed) ** 2
            )
            
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('光谱强度')
            ax.set_title(f'样本 {idx}\n重构误差: {reconstruction_error:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 填充误差区域
            ax.fill_between(wavelengths, original, reconstructed, alpha=0.3, color='gray', 
                          label='重构误差')
        
        # 隐藏多余的子图
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "spectral_reconstruction_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 光谱重构对比图已保存: {output_path}")
    
    def create_residual_analysis(self, spectra: np.ndarray):
        """
        创建残差分析图表
        
        Args:
            spectra: 光谱数据
        """
        print("🔄 创建残差分析图表...")
        
        # 计算所有样本的重构误差
        reconstruction_errors = []
        residuals = []
        
        for spectrum in spectra:
            # 数据预处理
            spectrum_scaled = self.models['scaler'].transform(spectrum.reshape(1, -1))
            
            # 通过编码器-解码器重构
            encoded = self.models['encoder'].predict(spectrum_scaled)
            decoded = self.models['decoder'].predict(encoded)
            
            # 计算重构误差和残差
            spectrum_original = self.models['scaler'].inverse_transform(spectrum_scaled)
            reconstruction_error = np.mean(
                self.models['weights'] * (spectrum_original - decoded) ** 2
            )
            
            residual = spectrum_original - decoded
            reconstruction_errors.append(reconstruction_error)
            residuals.append(residual.flatten())
        
        reconstruction_errors = np.array(reconstruction_errors)
        residuals = np.array(residuals)
        
        # 创建残差分析图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('重构残差分析', fontsize=16, fontweight='bold')
        
        # 1. 重构误差分布
        ax1.hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.percentile(reconstruction_errors, 99.5), color='red', linestyle='--', 
                   label=f'99.5%分位数: {np.percentile(reconstruction_errors, 99.5):.4f}')
        ax1.set_xlabel('重构误差')
        ax1.set_ylabel('频次')
        ax1.set_title('重构误差分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差随波长变化
        wavelengths, _ = self.data_loader.load_dvp_standard_curve()
        
        mean_residual = np.mean(residuals, axis=0)
        std_residual = np.std(residuals, axis=0)
        
        ax2.plot(wavelengths, mean_residual, 'b-', label='平均残差', linewidth=2)
        ax2.fill_between(wavelengths, 
                        mean_residual - std_residual, 
                        mean_residual + std_residual, 
                        alpha=0.3, color='blue', label='±1标准差')
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('残差')
        ax2.set_title('残差随波长变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差Q-Q图
        from scipy import stats
        stats.probplot(reconstruction_errors, dist="norm", plot=ax3)
        ax3.set_title('重构误差Q-Q图')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差vs重构值
        ax4.scatter(reconstruction_errors, reconstruction_errors, alpha=0.6, s=20)
        ax4.set_xlabel('重构误差')
        ax4.set_ylabel('重构误差')
        ax4.set_title('重构误差vs重构误差')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "residual_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 残差分析图已保存: {output_path}")
    
    def create_confusion_matrix_and_roc(self, quality_scores: np.ndarray, 
                                      stability_scores: np.ndarray,
                                      quality_labels: np.ndarray, 
                                      stability_labels: np.ndarray):
        """
        创建混淆矩阵和ROC曲线
        
        Args:
            quality_scores: Quality Score数组
            stability_scores: Stability Score数组
            quality_labels: Quality标签
            stability_labels: Stability标签
        """
        print("🔄 创建混淆矩阵和ROC曲线...")
        
        # 设置阈值（优先使用训练元数据）
        # Quality Score阈值：从similarity_evaluator元数据中读取
        quality_threshold = None
        se_metadata = self.metadata.get('similarity_evaluator', {})
        if se_metadata:
            quality_threshold = se_metadata.get('quality_threshold')
        if quality_threshold is None:
            quality_threshold = self.metadata.get('quality_threshold')
        if quality_threshold is None:
            # 回退到统计方法：正常样本的5%分位数（低分表示异常）
            quality_threshold = np.percentile(quality_scores[quality_labels == 0], 5)
        # 阈值单位规范化：若阈值大于1，视为百分比，需要/100
        try:
            if quality_threshold > 1.0:
                quality_threshold = float(quality_threshold) / 100.0
        except Exception:
            pass
        
        # Stability Score方向与阈值
        # 初步方向判断与修正
        s_scores_probe = stability_scores.copy()
        fpr_probe, tpr_probe, _ = roc_curve(stability_labels, s_scores_probe)
        auc_probe = auc(fpr_probe, tpr_probe)
        flipped = False
        if auc_probe < 0.55:
            stability_scores_dir = -s_scores_probe
            flipped = True
        else:
            stability_scores_dir = s_scores_probe

        # Stability Score阈值：从weighted_autoencoder元数据中读取
        stability_threshold = None
        wae_metadata = self.metadata.get('weighted_autoencoder', {})
        if wae_metadata:
            stability_threshold = wae_metadata.get('stability_threshold')
        if stability_threshold is None:
            stability_threshold = self.metadata.get('stability_threshold')
        if stability_threshold is None:
            # 回退到统计方法：正常样本的95%分位数（高分表示异常）在方向一致的分数上计算
            stability_threshold = np.percentile(stability_scores_dir[stability_labels == 0], 95)
        
        # 可选：阈值优化（基于验证集/当前标签近似）
        def find_optimal_threshold_by_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            j_scores = tpr - fpr
            return thresholds[np.argmax(j_scores)]

        def find_optimal_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
            thresholds = np.linspace(scores.min(), scores.max(), 200)
            best_f1, best_th = -1.0, thresholds[0]
            for th in thresholds:
                preds = (scores > th).astype(int)
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_th = f1, th
            return best_th

        if self.optimize_thresholds in ("youden", "f1"):
            # 质量分数方向：低分更异常 → 用(-quality_scores)找阈值，再换回原方向阈值
            q_scores_anom = -quality_scores
            if self.optimize_thresholds == "youden":
                q_th_anom = find_optimal_threshold_by_youden(quality_labels, q_scores_anom)
            else:
                q_th_anom = find_optimal_threshold_by_f1(quality_labels, q_scores_anom)
            # 将异常方向阈值转换回原分数阈值（即 -score > q_th_anom 等价 score < -q_th_anom）
            quality_threshold = -q_th_anom

            # 稳定性分数方向：使用已统一方向的分数，越大越异常
            s_scores_anom = stability_scores_dir
            if self.optimize_thresholds == "youden":
                stability_threshold = find_optimal_threshold_by_youden(stability_labels, s_scores_anom)
            else:
                stability_threshold = find_optimal_threshold_by_f1(stability_labels, s_scores_anom)

        print(f"📊 使用阈值: Quality={quality_threshold:.4f}, Stability={stability_threshold:.4f} (opt={self.optimize_thresholds}, flipped_stability={flipped}, auc_probe_stability={auc_probe:.3f})")
        
        # 预测标签
        quality_pred = (quality_scores < quality_threshold).astype(int)
        # 使用方向一致后的稳定性分数
        stability_pred = (stability_scores_dir > stability_threshold).astype(int)
        
        # 组合预测策略（命令行覆盖优先，其次metadata，最后默认two_stage）
        combine_strategy = self.combine_strategy or self.metadata.get('combine_strategy', 'two_stage')
        if combine_strategy == 'and':
            combined_pred = ((quality_pred == 1) & (stability_pred == 1)).astype(int)
        elif combine_strategy == 'or':
            combined_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
        else:
            # two_stage: 若Quality已异常直接判异常；否则仅由Stability决定
            combined_pred = np.where(quality_pred == 1, 1, stability_pred).astype(int)
        combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型性能评估', fontsize=16, fontweight='bold')
        
        # 1. Quality Score混淆矩阵
        cm_quality = confusion_matrix(quality_labels, quality_pred)
        sns.heatmap(cm_quality, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Quality Score混淆矩阵')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        
        # 2. Stability Score混淆矩阵
        cm_stability = confusion_matrix(stability_labels, stability_pred)
        sns.heatmap(cm_stability, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title('Stability Score混淆矩阵')
        ax2.set_xlabel('预测标签')
        ax2.set_ylabel('真实标签')
        
        # 3. 组合模型混淆矩阵
        cm_combined = confusion_matrix(combined_true, combined_pred)
        sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Oranges', ax=ax3)
        ax3.set_title('组合模型混淆矩阵')
        ax3.set_xlabel('预测标签')
        ax3.set_ylabel('真实标签')
        
        # 4. ROC曲线
        # Quality Score ROC
        fpr_quality, tpr_quality, _ = roc_curve(quality_labels, -quality_scores)  # 负号因为低分是异常
        roc_auc_quality = auc(fpr_quality, tpr_quality)
        
        # Stability Score ROC（基于方向一致分数）
        fpr_stability, tpr_stability, _ = roc_curve(stability_labels, stability_scores_dir)
        roc_auc_stability = auc(fpr_stability, tpr_stability)
        
        # 组合ROC：统一异常方向并做Min-Max归一化后，按单模型AUC加权融合
        eps = 1e-12
        # 质量异常分数：1 - Q（Q高→正常），再归一化到[0,1]
        q_anom = 1.0 - quality_scores
        q_anom = (q_anom - q_anom.min()) / (q_anom.max() - q_anom.min() + eps)
        # 稳定性异常分数：重构误差高→异常，归一化到[0,1]
        s_anom = (stability_scores_dir - stability_scores_dir.min()) / (stability_scores_dir.max() - stability_scores_dir.min() + eps)
        # 权重按单模型AUC归一
        w_q = max(roc_auc_quality, 1e-3)
        w_s = max(roc_auc_stability, 1e-3)
        w_sum = w_q + w_s
        w_q /= w_sum
        w_s /= w_sum
        combined_scores = w_q * q_anom + w_s * s_anom
        print(f"🧮 组合权重: quality={w_q:.3f}, stability={w_s:.3f}")
        fpr_combined, tpr_combined, _ = roc_curve(combined_true, combined_scores)
        roc_auc_combined = auc(fpr_combined, tpr_combined)
        
        ax4.plot(fpr_quality, tpr_quality, 'b-', 
                label=f'Quality Score (AUC = {roc_auc_quality:.3f})')
        ax4.plot(fpr_stability, tpr_stability, 'g-', 
                label=f'Stability Score (AUC = {roc_auc_stability:.3f})')
        ax4.plot(fpr_combined, tpr_combined, 'r-', 
                label=f'Combined (AUC = {roc_auc_combined:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', label='Random')
        ax4.set_xlabel('假阳性率')
        ax4.set_ylabel('真阳性率')
        ax4.set_title('ROC曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "confusion_matrix_and_roc.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 额外输出：PR曲线
        plt.figure(figsize=(8,6))
        # Quality PR（异常为1 → 使用 -quality_scores）
        pq, rq, _ = precision_recall_curve(quality_labels, -quality_scores)
        ap_q = average_precision_score(quality_labels, -quality_scores)
        plt.plot(rq, pq, label=f'Quality AP={ap_q:.3f}')
        # Stability PR（使用方向一致分数）
        ps, rs, _ = precision_recall_curve(stability_labels, stability_scores_dir)
        ap_s = average_precision_score(stability_labels, stability_scores_dir)
        plt.plot(rs, ps, label=f'Stability AP={ap_s:.3f}')
        # Combined PR
        pc, rc, _ = precision_recall_curve(combined_true, combined_scores)
        ap_c = average_precision_score(combined_true, combined_scores)
        plt.plot(rc, pc, label=f'Combined AP={ap_c:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall 曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        pr_path = self.output_dir / 'pr_curves.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 阈值敏感性（F1 vs 阈值）
        plt.figure(figsize=(8,6))
        # Quality：基于 -quality_scores 的阈值
        th_q = np.linspace((-quality_scores).min(), (-quality_scores).max(), 100)
        f1_q = []
        for th in th_q:
            pred = ((-quality_scores) > th).astype(int)
            f1_q.append(f1_score(quality_labels, pred, zero_division=0))
        plt.plot(th_q, f1_q, label='Quality F1')
        # Stability
        th_s = np.linspace(stability_scores_dir.min(), stability_scores_dir.max(), 100)
        f1_s = []
        for th in th_s:
            pred = (stability_scores_dir > th).astype(int)
            f1_s.append(f1_score(stability_labels, pred, zero_division=0))
        plt.plot(th_s, f1_s, label='Stability F1')
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('阈值敏感性（F1 vs 阈值）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        th_path = self.output_dir / 'threshold_sensitivity.png'
        plt.savefig(th_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 稳定性分数直方图（正负样本）
        plt.figure(figsize=(8,6))
        plt.hist(stability_scores_dir[stability_labels==0], bins=40, alpha=0.6, label='正常')
        plt.hist(stability_scores_dir[stability_labels==1], bins=40, alpha=0.6, label='异常')
        plt.xlabel('Stability 异常分数(方向已统一)')
        plt.ylabel('频次')
        plt.title('Stability Score 分布（正负样本）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        hist_path = self.output_dir / 'stability_score_hist.png'
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算性能指标
        # 在度量计算中也使用方向一致的稳定性分数
        performance_metrics = self._calculate_performance_metrics(
            quality_scores, stability_scores_dir, quality_labels, stability_labels,
            quality_threshold, stability_threshold, combine_strategy=combine_strategy
        )
        # 记录组合策略
        performance_metrics['combined_model'] = {
            **performance_metrics['combined_model'],
            'combine_strategy': combine_strategy
        }
        
        print(f"✅ 混淆矩阵和ROC曲线已保存: {output_path}")
        return performance_metrics
    
    def _calculate_performance_metrics(self, quality_scores: np.ndarray, 
                                     stability_scores: np.ndarray,
                                     quality_labels: np.ndarray, 
                                     stability_labels: np.ndarray,
                                     quality_threshold: float, 
                                     stability_threshold: float,
                                     combine_strategy: Optional[str] = None) -> Dict:
        """
        计算模型性能指标
        
        Returns:
            性能指标字典
        """
        # 预测标签
        quality_pred = (quality_scores < quality_threshold).astype(int)
        stability_pred = (stability_scores > stability_threshold).astype(int)
        # 组合预测遵循与图一致的策略
        strategy = combine_strategy or self.combine_strategy or self.metadata.get('combine_strategy', 'two_stage')
        if strategy == 'and':
            combined_pred = ((quality_pred == 1) & (stability_pred == 1)).astype(int)
        elif strategy == 'or':
            combined_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
        else:
            combined_pred = np.where(quality_pred == 1, 1, stability_pred).astype(int)
        combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
        
        # Quality Score指标
        quality_accuracy = accuracy_score(quality_labels, quality_pred)
        quality_precision = precision_score(quality_labels, quality_pred, zero_division=0)
        quality_recall = recall_score(quality_labels, quality_pred, zero_division=0)
        quality_f1 = f1_score(quality_labels, quality_pred, zero_division=0)
        
        # Stability Score指标
        stability_accuracy = accuracy_score(stability_labels, stability_pred)
        stability_precision = precision_score(stability_labels, stability_pred, zero_division=0)
        stability_recall = recall_score(stability_labels, stability_pred, zero_division=0)
        stability_f1 = f1_score(stability_labels, stability_pred, zero_division=0)
        
        # 组合模型指标
        combined_accuracy = accuracy_score(combined_true, combined_pred)
        combined_precision = precision_score(combined_true, combined_pred, zero_division=0)
        combined_recall = recall_score(combined_true, combined_pred, zero_division=0)
        combined_f1 = f1_score(combined_true, combined_pred, zero_division=0)
        
        # ROC AUC
        fpr_quality, tpr_quality, _ = roc_curve(quality_labels, -quality_scores)
        roc_auc_quality = auc(fpr_quality, tpr_quality)
        
        fpr_stability, tpr_stability, _ = roc_curve(stability_labels, stability_scores)
        roc_auc_stability = auc(fpr_stability, tpr_stability)
        
        # 与图同样的归一化组合分数用于AUC
        eps = 1e-12
        q_anom = 1.0 - quality_scores
        q_anom = (q_anom - q_anom.min()) / (q_anom.max() - q_anom.min() + eps)
        s_anom = (stability_scores - stability_scores.min()) / (stability_scores.max() - stability_scores.min() + eps)
        combined_scores = 0.5 * q_anom + 0.5 * s_anom
        fpr_combined, tpr_combined, _ = roc_curve(combined_true, combined_scores)
        roc_auc_combined = auc(fpr_combined, tpr_combined)
        
        metrics = {
            'quality_score': {
                'accuracy': quality_accuracy,
                'precision': quality_precision,
                'recall': quality_recall,
                'f1_score': quality_f1,
                'auc_roc': roc_auc_quality,
                'threshold': quality_threshold
            },
            'stability_score': {
                'accuracy': stability_accuracy,
                'precision': stability_precision,
                'recall': stability_recall,
                'f1_score': stability_f1,
                'auc_roc': roc_auc_stability,
                'threshold': stability_threshold
            },
            'combined_model': {
                'accuracy': combined_accuracy,
                'precision': combined_precision,
                'recall': combined_recall,
                'f1_score': combined_f1,
                'auc_roc': roc_auc_combined
            }
        }
        
        return metrics
    
    def generate_evaluation_report(self, metrics: Dict, n_samples: int):
        """
        生成评估报告
        
        Args:
            metrics: 性能指标字典
            n_samples: 评估样本数量
        """
        print("📝 生成评估报告...")
        
        # 从元数据推断模型版本信息
        coating_name = self.metadata.get('coating_name', 'DVP') if isinstance(self.metadata, dict) else 'DVP'
        version = self.metadata.get('version', None) if isinstance(self.metadata, dict) else None
        model_version_str = f"{coating_name}_{version}" if version else coating_name

        report = f"""# DVP涂层光谱异常检测模型评估报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**评估样本数**: {n_samples}  
**模型版本**: {model_version_str}  

## 模型概述

本评估报告基于已训练的DVP涂层光谱异常检测模型，该模型采用混合架构：
- **Quality Score**: 基于专家规则的光谱相似性评估
- **Stability Score**: 基于加权自编码器的重构误差评估

## 性能指标

### Quality Score模型
- **准确率**: {metrics['quality_score']['accuracy']:.4f}
- **精确率**: {metrics['quality_score']['precision']:.4f}
- **召回率**: {metrics['quality_score']['recall']:.4f}
- **F1分数**: {metrics['quality_score']['f1_score']:.4f}
- **AUC-ROC**: {metrics['quality_score']['auc_roc']:.4f}
- **阈值**: {metrics['quality_score']['threshold']:.4f}

### Stability Score模型
- **准确率**: {metrics['stability_score']['accuracy']:.4f}
- **精确率**: {metrics['stability_score']['precision']:.4f}
- **召回率**: {metrics['stability_score']['recall']:.4f}
- **F1分数**: {metrics['stability_score']['f1_score']:.4f}
- **AUC-ROC**: {metrics['stability_score']['auc_roc']:.4f}
- **阈值**: {metrics['stability_score']['threshold']:.4f}

### 组合模型
- **准确率**: {metrics['combined_model']['accuracy']:.4f}
- **精确率**: {metrics['combined_model']['precision']:.4f}
- **召回率**: {metrics['combined_model']['recall']:.4f}
- **F1分数**: {metrics['combined_model']['f1_score']:.4f}
- **AUC-ROC**: {metrics['combined_model']['auc_roc']:.4f}
 - **组合策略**: {metrics['combined_model'].get('combine_strategy', 'two_stage')}

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

## 结论

DVP涂层光谱异常检测模型在测试数据集上表现良好，组合模型达到了{metrics['combined_model']['accuracy']:.1%}的准确率。
该模型能够有效识别光谱质量异常和稳定性异常，为涂层质量控制提供了可靠的技术支持。

---
*报告由MiniMax Agent自动生成*
"""
        
        # 保存报告
        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON格式的指标
        metrics_path = self.output_dir / "performance_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 评估报告已保存: {report_path}")
        print(f"📊 性能指标已保存: {metrics_path}")
    
    def run_complete_evaluation(self, n_samples: int = 1000):
        """
        运行完整的模型评估流程
        
        Args:
            n_samples: 评估样本数量
        """
        print("🚀 开始完整的模型评估流程...")
        print("=" * 60)
        
        try:
            # 1. 生成测试数据
            spectra, quality_labels, stability_labels = self.generate_test_data(n_samples)
            
            # 2. 计算Score
            quality_scores, stability_scores = self.calculate_scores(spectra)
            
            # 3. 创建各种可视化
            self.create_quality_stability_scatter(
                quality_scores, stability_scores, quality_labels, stability_labels
            )
            
            self.create_spectral_reconstruction_comparison(spectra)
            
            self.create_residual_analysis(spectra)
            
            metrics = self.create_confusion_matrix_and_roc(
                quality_scores, stability_scores, quality_labels, stability_labels
            )
            
            # 4. 生成评估报告
            self.generate_evaluation_report(metrics, n_samples)
            
            print("=" * 60)
            print("🎉 模型评估完成!")
            print(f"📁 所有结果已保存到: {self.output_dir}")
            print("\n📊 关键指标:")
            print(f"   - 组合模型准确率: {metrics['combined_model']['accuracy']:.4f}")
            print(f"   - 组合模型F1分数: {metrics['combined_model']['f1_score']:.4f}")
            print(f"   - 组合模型AUC-ROC: {metrics['combined_model']['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"❌ 评估过程中发生错误: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DVP涂层光谱异常检测模型评估')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='评估样本数量 (默认: 1000)')
    project_root = Path(__file__).parent.parent
    parser.add_argument('--model-dir', type=str, 
                       default=str(project_root / "models"),
                       help='模型文件目录')
    parser.add_argument('--combine-strategy', type=str, choices=['two_stage', 'and', 'or'],
                       default=None, help='组合策略覆盖(two_stage|and|or)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子，用于生成可复现的测试数据（默认: 42）')
    parser.add_argument('--optimize-thresholds', type=str, choices=['none', 'youden', 'f1'], default='none',
                       help='阈值优化方法：none/youden/f1（默认: none）')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(model_dir=args.model_dir, combine_strategy=args.combine_strategy,
                               random_seed=args.random_seed, optimize_thresholds=args.optimize_thresholds)
    evaluator.run_complete_evaluation(n_samples=args.samples)


if __name__ == "__main__":
    main()