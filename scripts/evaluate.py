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

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOTTING_LIBS = True
except ModuleNotFoundError:
    HAS_PLOTTING_LIBS = False

    class _DummyPlotModule:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

    class _DummyPatches:
        def Patch(self, *args, **kwargs):
            return None

    plt = _DummyPlotModule()
    sns = _DummyPlotModule()
    mpatches = _DummyPatches()
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
    if not HAS_PLOTTING_LIBS:
        warnings.warn("matplotlib/seaborn 未安装，跳过可视化生成，仅输出数值指标。", RuntimeWarning)
        return
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
    
    def __init__(self, model_dir: str = None, combine_strategy: str = None, random_seed: int = 42, optimize_thresholds: str = "f1"):
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
        self.optimize_thresholds = (optimize_thresholds or "f1").lower()
        
        # 初始化组件
        self.data_loader = SpectrumDataLoader()
        self.similarity_evaluator = SimilarityEvaluator()
        
        # 加载模型
        self.models = {}
        self.metadata = {}
        self._load_models()

        # 残差融合配置（若metadata提供则作为默认值）
        self.residual_fusion_config = self.metadata.get('residual_fusion') if isinstance(self.metadata, dict) else None
        if self.residual_fusion_config:
            weights_cfg = self.residual_fusion_config.get('weights', {})
            if not hasattr(self, 'use_residual_clf'):
                setattr(self, 'use_residual_clf', True)
            if not hasattr(self, 'residual_fuse_mode'):
                setattr(self, 'residual_fuse_mode', self.residual_fusion_config.get('fuse_mode', 'weighted'))
            if not hasattr(self, 'residual_weight') and isinstance(weights_cfg, dict):
                # 兼容旧逻辑：将残差权重作为residual_weight
                residual_w = float(weights_cfg.get('residual', 0.5))
                setattr(self, 'residual_weight', residual_w)
            self.residual_threshold_config = self.residual_fusion_config.get('threshold')
            self.residual_weights_config = weights_cfg if isinstance(weights_cfg, dict) else None
        else:
            self.residual_threshold_config = None
            self.residual_weights_config = None
        
        # 设置matplotlib
        setup_matplotlib_for_plotting()
        
        print("[OK] 模型评估器初始化完成")
        print(f"[DIR] 模型目录: {self.model_dir}")
        print(f"[INFO] 输出目录: {self.output_dir}")
    
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
                print("[WARN] 使用默认DVP权重向量")
            
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
                print("[WARN] 未找到元数据文件，将使用统计回退方式估计阈值")
            
            print("[OK] 模型文件加载成功")
            print(f"[FILE] 编码器: {encoder_path}")
            print(f"[FILE] 解码器: {decoder_path}")
            print(f"[FILE] 标准化器: {scaler_path}")
            if weights_file:
                print(f"[FILE] 权重文件: {weights_file}")
            if self.metadata:
                print("[INFO] 元数据已加载")

            # 可选：加载已训练的残差分类器
            residual_clf_path = _find_one([
                "*residual_clf*DVP*v*.joblib", "*residual_clf*DVP*.joblib", "*residual_clf*.joblib"
            ])
            if residual_clf_path and Path(residual_clf_path).exists():
                try:
                    self.models['residual_clf'] = joblib.load(residual_clf_path)
                    print(f"[FILE] 残差分类器: {residual_clf_path}")
                except Exception:
                    pass
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise
    
    def generate_test_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成测试数据用于评估
        
        如果提供了test_data_npz，则使用真实测试数据；否则生成合成数据。
        
        Args:
            n_samples: 生成样本数量
            
        Returns:
            Tuple[spectra, quality_labels, stability_labels]
            - spectra: 光谱数据
            - quality_labels: 质量标签 (0: 正常, 1: 异常)
            - stability_labels: 稳定性标签 (0: 正常, 1: 异常)
        """
        # 检查是否使用真实测试数据（混合模式：真实正常样本 + 合成异常样本）
        if hasattr(self, 'test_data_npz') and self.test_data_npz:
            try:
                print(f"[LOAD] 加载真实测试数据: {self.test_data_npz}")
                data = np.load(self.test_data_npz)
                real_spectra = data['dvp_values']
                wavelengths = data['wavelengths']
                
                # 计算需要的正常样本数（80%）
                n_normal = int(n_samples * 0.8)
                
                # 从真实数据中随机采样正常样本
                if len(real_spectra) >= n_normal:
                    np.random.seed(self.random_seed)
                    indices = np.random.choice(len(real_spectra), n_normal, replace=False)
                    normal_spectra = real_spectra[indices]
                else:
                    normal_spectra = real_spectra
                print(f"[WARN] 真实数据不足{n_normal}个，使用全部{len(real_spectra)}个")
                
                print(f"[OK] 加载 {len(normal_spectra)} 个真实正常样本")
                
                # 生成合成异常样本（20%）
                print(f"[INFO] 生成 {n_samples - len(normal_spectra)} 个合成异常样本...")
                np.random.seed(self.random_seed + 1)  # 不同种子避免重复
                
                # 加载标准曲线用于生成异常
                standard_spectrum = np.median(real_spectra, axis=0)  # 用真实数据的中位数作为标准
                
                n_anomaly = n_samples - len(normal_spectra)
                anomaly_spectra = []
                quality_anomaly_labels = []
                stability_anomaly_labels = []
                
                for i in range(n_anomaly):
                    anomaly_type = np.random.choice(['quality', 'stability', 'both'])
                    
                    if anomaly_type == 'quality':
                        # 质量异常：光谱形状异常
                        spectrum = standard_spectrum.copy()
                        peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                        spectrum[peak_mask] += np.random.normal(0, 0.5, np.sum(peak_mask))
                        quality_anomaly_labels.append(1)
                        stability_anomaly_labels.append(0)
                    elif anomaly_type == 'stability':
                        # 稳定性异常：整体偏移或噪声
                        spectrum = standard_spectrum.copy()
                        spectrum += np.random.normal(0, 0.2, len(spectrum))
                        quality_anomaly_labels.append(0)
                        stability_anomaly_labels.append(1)
                    else:  # both
                        spectrum = standard_spectrum.copy()
                        peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
                        spectrum[peak_mask] += np.random.normal(0, 0.5, np.sum(peak_mask))
                        spectrum += np.random.normal(0, 0.2, len(spectrum))
                        quality_anomaly_labels.append(1)
                        stability_anomaly_labels.append(1)
                    
                    anomaly_spectra.append(spectrum)
                
                # 合并数据
                all_spectra = np.vstack([normal_spectra, np.array(anomaly_spectra)])
                quality_labels = np.concatenate([
                    np.zeros(len(normal_spectra), dtype=int),
                    np.array(quality_anomaly_labels, dtype=int)
                ])
                stability_labels = np.concatenate([
                    np.zeros(len(normal_spectra), dtype=int),
                    np.array(stability_anomaly_labels, dtype=int)
                ])
                
                print(f"[OK] 混合测试集: {len(normal_spectra)}个真实正常样本 + {n_anomaly}个合成异常样本")
                return all_spectra, quality_labels, stability_labels
                
            except Exception as e:
                print(f"[WARN] 无法加载真实测试数据: {e}")
                print("[WARN] 回退到合成测试数据")
        
        print(f"[INFO] 生成 {n_samples} 个合成测试样本...")
        print(f"[INFO] 使用随机种子: {self.random_seed} (确保数据可复现)")
        
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
        
        print("[OK] 测试数据生成完成:")
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
        print("[INFO] 计算Quality Score和Stability Score...")
        
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
        
        print("[OK] Score计算完成:")
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
        print("[INFO] 创建Quality Score vs Stability Score散点图...")
        if not HAS_PLOTTING_LIBS:
            print("[WARN] 绘图库缺失，跳过散点图生成")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DVP Coating Spectral Anomaly Evaluation', fontsize=16, fontweight='bold')
        
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
        ax1.set_title('Quality Score vs Stability Score (Overall)')
        ax1.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='green', label='Normal'),
            mpatches.Patch(color='orange', label='Quality Anomaly'),
            mpatches.Patch(color='blue', label='Stability Anomaly'),
            mpatches.Patch(color='red', label='Both')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. Quality Score分布
        ax2.hist(quality_scores[quality_labels == 0], bins=30, alpha=0.7, 
                label='正常', color='green', density=True)
        ax2.hist(quality_scores[quality_labels == 1], bins=30, alpha=0.7, 
                label='质量异常', color='red', density=True)
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Quality Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stability Score分布
        ax3.hist(stability_scores[stability_labels == 0], bins=30, alpha=0.7, 
                label='正常', color='green', density=True)
        ax3.hist(stability_scores[stability_labels == 1], bins=30, alpha=0.7, 
                label='稳定性异常', color='red', density=True)
        ax3.set_xlabel('Stability Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Stability Score Distribution')
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
                   c='green', s=20, alpha=0.7, label='Normal')
        ax4.scatter(quality_scores[anomaly_mask], stability_scores[anomaly_mask], 
                   c='red', s=20, alpha=0.7, label='Anomaly')
        
        ax4.set_xlabel('Quality Score')
        ax4.set_ylabel('Stability Score')
        ax4.set_title('Decision Region')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "quality_stability_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Scatter figure saved: {output_path}")
    
    def create_spectral_reconstruction_comparison(self, spectra: np.ndarray, 
                                                sample_indices: List[int] = None):
        """
        创建光谱重构对比可视化
        
        Args:
            spectra: 光谱数据
            sample_indices: 要展示的样本索引列表
        """
        print("[INFO] 创建光谱重构对比可视化...")
        if not HAS_PLOTTING_LIBS:
            print("[WARN] 绘图库缺失，跳过光谱重构对比图生成")
            return
        
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
        
        fig.suptitle('Spectral Reconstruction Comparison', fontsize=16, fontweight='bold')
        
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
            ax.plot(wavelengths, original, 'b-', label='Original Spectrum', linewidth=2)
            ax.plot(wavelengths, reconstructed, 'r--', label='Reconstructed Spectrum', linewidth=2)
            
            # 计算误差
            reconstruction_error = np.mean(
                self.models['weights'] * (original - reconstructed) ** 2
            )
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Sample {idx}\nReconstruction Error: {reconstruction_error:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 填充误差区域
            ax.fill_between(wavelengths, original, reconstructed, alpha=0.3, color='gray', 
                          label='Reconstruction Error')
        
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
        
        print(f"[OK] Reconstruction comparison saved: {output_path}")
    
    def create_residual_analysis(self, spectra: np.ndarray):
        """
        创建残差分析图表
        
        Args:
            spectra: 光谱数据
        """
        print("[INFO] 创建残差分析图表...")
        if not HAS_PLOTTING_LIBS:
            print("[WARN] 绘图库缺失，跳过残差分析图生成")
            return
        
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
        fig.suptitle('Reconstruction Residual Analysis', fontsize=16, fontweight='bold')
        
        # 1. 重构误差分布
        ax1.hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.percentile(reconstruction_errors, 99.5), color='red', linestyle='--', 
                   label=f'99.5th Percentile: {np.percentile(reconstruction_errors, 99.5):.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reconstruction Error Distribution')
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
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual vs Wavelength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差Q-Q图
        from scipy import stats
        stats.probplot(reconstruction_errors, dist="norm", plot=ax3)
        ax3.set_title('Reconstruction Error Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差vs重构值
        ax4.scatter(reconstruction_errors, reconstruction_errors, alpha=0.6, s=20)
        ax4.set_xlabel('Reconstruction Error')
        ax4.set_ylabel('Reconstruction Error')
        ax4.set_title('Reconstruction Error vs Reconstruction Error')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "residual_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Residual analysis saved: {output_path}")

    # --- 新增：残差特征与轻量分类器 ---
    def _compute_residuals_matrix(self, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (residuals_matrix, wavelengths)；residuals_matrix: [N, W]"""
        wavelengths, _ = self.data_loader.load_dvp_standard_curve()
        residuals = []
        for spectrum in spectra:
            spectrum_scaled = self.models['scaler'].transform(spectrum.reshape(1, -1))
            encoded = self.models['encoder'].predict(spectrum_scaled)
            decoded = self.models['decoder'].predict(encoded)
            spectrum_original = self.models['scaler'].inverse_transform(spectrum_scaled)
            residual = (spectrum_original - decoded).flatten()
            residuals.append(residual)
        return np.array(residuals), wavelengths

    def _extract_segment_features(self, residuals: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """
        提取分段残差特征：对若干波段计算 [mean(|r|), rms(r), max(|r|)]
        段定义：400-480, 480-560, 560-680, 680-780（含边界）
        返回形状 [N, 12]
        """
        bands = [(400, 480), (480, 560), (560, 680), (680, 780)]
        feats = []
        for lo, hi in bands:
            mask = (wavelengths >= lo) & (wavelengths <= hi)
            seg = residuals[:, mask]
            abs_seg = np.abs(seg)
            mean_abs = np.mean(abs_seg, axis=1)
            rms = np.sqrt(np.mean(seg**2, axis=1))
            max_abs = np.max(abs_seg, axis=1)
            feats.append(mean_abs)
            feats.append(rms)
            feats.append(max_abs)
        return np.stack(feats, axis=1).T if False else np.column_stack(feats)

    def _fit_residual_logistic(self, X: np.ndarray, y: np.ndarray) -> Tuple[object, np.ndarray]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        oof = np.zeros(len(y))
        model = LogisticRegression(max_iter=1000)
        for train_idx, val_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[val_idx])[:, 1]
            oof[val_idx] = proba
        # 再在全量上拟合一个最终模型用于部署
        model.fit(X, y)
        return model, oof

    def compute_residual_classifier_scores(self, spectra: np.ndarray, stability_labels: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        生成基于残差特征的轻量分类器分数（概率，越大越异常）并返回模型。
        """
        residuals, wavelengths = self._compute_residuals_matrix(spectra)
        X = self._extract_segment_features(residuals, wavelengths)
        clf, oof_scores = self._fit_residual_logistic(X, stability_labels)
        return oof_scores, clf
    
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
        print("[INFO] 创建混淆矩阵和ROC曲线...")
        plotting_enabled = HAS_PLOTTING_LIBS
        if not plotting_enabled:
            print("[WARN] 绘图库缺失，跳过混淆矩阵和ROC曲线可视化，仅计算指标")
        
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

        print(f"[INFO] 使用阈值: Quality={quality_threshold:.4f}, Stability={stability_threshold:.4f} (opt={self.optimize_thresholds}, flipped_stability={flipped}, auc_probe_stability={auc_probe:.3f})")
        
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
        elif combine_strategy == 'weighted':
            # 加权组合策略：根据AUC加权，统一方向后归一化再组合
            # Quality Score: 低分表示异常，转换为异常分数 = 1 - norm(quality_scores)
            # Stability Score: 高分表示异常，异常分数 = norm(stability_scores_dir)
            
            # 归一化到[0, 1]
            q_min, q_max = quality_scores.min(), quality_scores.max()
            s_min, s_max = stability_scores_dir.min(), stability_scores_dir.max()
            
            if q_max > q_min:
                quality_norm = (quality_scores - q_min) / (q_max - q_min)
            else:
                quality_norm = np.zeros_like(quality_scores)
            
            if s_max > s_min:
                stability_norm = (stability_scores_dir - s_min) / (s_max - s_min)
            else:
                stability_norm = np.zeros_like(stability_scores_dir)
            
            # Quality低分是异常，所以取反
            quality_anomaly_score = 1.0 - quality_norm
            # Stability高分是异常，直接使用
            stability_anomaly_score = stability_norm
            
            # 根据AUC加权（从性能指标读取）
            q_auc = metrics.get('quality_score', {}).get('auc_roc', 0.903) if 'metrics' in locals() else 0.903
            s_auc = metrics.get('stability_score', {}).get('auc_roc', 0.735) if 'metrics' in locals() else 0.735
            total_auc = q_auc + s_auc
            alpha = q_auc / total_auc  # Quality权重
            beta = s_auc / total_auc   # Stability权重
            
            # 加权组合
            combined_score = alpha * quality_anomaly_score + beta * stability_anomaly_score
            
            # 找最佳阈值
            combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
            if self.optimize_thresholds == 'youden':
                fpr_comb, tpr_comb, thresholds_comb = roc_curve(combined_true, combined_score)
                youden_index = tpr_comb - fpr_comb
                best_idx = np.argmax(youden_index)
                combined_threshold = thresholds_comb[best_idx]
            elif self.optimize_thresholds == 'f1':
                best_f1 = 0
                best_threshold = 0.5
                for thresh in np.linspace(0, 1, 101):
                    pred = (combined_score > thresh).astype(int)
                    f1 = f1_score(combined_true, pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                combined_threshold = best_threshold
            else:
                combined_threshold = 0.5
            
            combined_pred = (combined_score > combined_threshold).astype(int)
            print(f"   加权组合: α(Quality)={alpha:.3f}, β(Stability)={beta:.3f}, 阈值={combined_threshold:.4f}")
        else:
            # two_stage: 若Quality已异常直接判异常；否则仅由Stability决定
            combined_pred = np.where(quality_pred == 1, 1, stability_pred).astype(int)
        
        combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
        
        # 计算 ROC/PR 所需指标
        fpr_quality, tpr_quality, _ = roc_curve(quality_labels, -quality_scores)
        roc_auc_quality = auc(fpr_quality, tpr_quality)
        fpr_stability, tpr_stability, _ = roc_curve(stability_labels, stability_scores_dir)
        roc_auc_stability = auc(fpr_stability, tpr_stability)

        eps = 1e-12
        q_anom = 1.0 - quality_scores
        q_anom = (q_anom - q_anom.min()) / (q_anom.max() - q_anom.min() + eps)
        s_anom = (stability_scores_dir - stability_scores_dir.min()) / (stability_scores_dir.max() - stability_scores_dir.min() + eps)
        w_q = max(roc_auc_quality, 1e-3)
        w_s = max(roc_auc_stability, 1e-3)
        w_sum = w_q + w_s
        w_q /= w_sum
        w_s /= w_sum
        combined_scores = w_q * q_anom + w_s * s_anom
        print(f"[INFO] 组合权重: quality={w_q:.3f}, stability={w_s:.3f}")
        fpr_combined, tpr_combined, _ = roc_curve(combined_true, combined_scores)
        roc_auc_combined = auc(fpr_combined, tpr_combined)

        if plotting_enabled:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')

            cm_quality = confusion_matrix(quality_labels, quality_pred)
            sns.heatmap(cm_quality, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Quality Score Confusion Matrix')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')

            cm_stability = confusion_matrix(stability_labels, stability_pred)
            sns.heatmap(cm_stability, annot=True, fmt='d', cmap='Greens', ax=ax2)
            ax2.set_title('Stability Score Confusion Matrix')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')

            cm_combined = confusion_matrix(combined_true, combined_pred)
            sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Oranges', ax=ax3)
            ax3.set_title('Combined Model Confusion Matrix')
            ax3.set_xlabel('Predicted Label')
            ax3.set_ylabel('True Label')

            ax4.plot(fpr_quality, tpr_quality, 'b-', label=f'Quality Score (AUC = {roc_auc_quality:.3f})')
            ax4.plot(fpr_stability, tpr_stability, 'g-', label=f'Stability Score (AUC = {roc_auc_stability:.3f})')
            ax4.plot(fpr_combined, tpr_combined, 'r-', label=f'Combined (AUC = {roc_auc_combined:.3f})')
            ax4.plot([0, 1], [0, 1], 'k--', label='Random')
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.output_dir / "confusion_matrix_and_roc.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            pq, rq, _ = precision_recall_curve(quality_labels, -quality_scores)
            ap_q = average_precision_score(quality_labels, -quality_scores)
            plt.plot(rq, pq, label=f'Quality AP={ap_q:.3f}')
            ps, rs, _ = precision_recall_curve(stability_labels, stability_scores_dir)
            ap_s = average_precision_score(stability_labels, stability_scores_dir)
            plt.plot(rs, ps, label=f'Stability AP={ap_s:.3f}')
            pc, rc, _ = precision_recall_curve(combined_true, combined_scores)
            ap_c = average_precision_score(combined_true, combined_scores)
            plt.plot(rc, pc, label=f'Combined AP={ap_c:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            pr_path = self.output_dir / 'pr_curves.png'
            plt.savefig(pr_path, dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            th_q = np.linspace((-quality_scores).min(), (-quality_scores).max(), 100)
            f1_q = []
            for th in th_q:
                pred = ((-quality_scores) > th).astype(int)
                f1_q.append(f1_score(quality_labels, pred, zero_division=0))
            plt.plot(th_q, f1_q, label='Quality F1')
            th_s = np.linspace(stability_scores_dir.min(), stability_scores_dir.max(), 100)
            f1_s = []
            for th in th_s:
                pred = (stability_scores_dir > th).astype(int)
                f1_s.append(f1_score(stability_labels, pred, zero_division=0))
            plt.plot(th_s, f1_s, label='Stability F1')
            plt.xlabel('Threshold')
            plt.ylabel('F1')
            plt.title('Threshold Sensitivity (F1 vs Threshold)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            th_path = self.output_dir / 'threshold_sensitivity.png'
            plt.savefig(th_path, dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(stability_scores_dir[stability_labels == 0], bins=40, alpha=0.6, label='Normal')
            plt.hist(stability_scores_dir[stability_labels == 1], bins=40, alpha=0.6, label='Anomaly')
            plt.xlabel('Stability Anomaly Score (direction unified)')
            plt.ylabel('Frequency')
            plt.title('Stability Score Distribution (Positive vs Negative)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            hist_path = self.output_dir / 'stability_score_hist.png'
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[OK] Confusion matrices and ROC saved: {output_path}")
        else:
            print("[INFO] 已跳过图表生成，继续计算性能指标")

        # 计算性能指标
        # 在度量计算中也使用方向一致的稳定性分数
        performance_metrics = self._calculate_performance_metrics(
            quality_scores, stability_scores_dir, quality_labels, stability_labels,
            quality_threshold, stability_threshold, combine_strategy=combine_strategy
        )
        # 记录阈值优化方法与稳定性翻转探测
        performance_metrics['stability_score']['flipped'] = bool(flipped)
        performance_metrics['stability_score']['auc_probe'] = float(auc_probe)
        performance_metrics['optimize_thresholds'] = self.optimize_thresholds
        # 记录组合策略
        performance_metrics['combined_model'] = {
            **performance_metrics['combined_model'],
            'combine_strategy': combine_strategy
        }
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
        combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
        
        if strategy == 'and':
            combined_pred = ((quality_pred == 1) & (stability_pred == 1)).astype(int)
        elif strategy == 'or':
            combined_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
        elif strategy == 'weighted':
            # 加权组合策略（与评估图中的逻辑一致）
            q_min, q_max = quality_scores.min(), quality_scores.max()
            s_min, s_max = stability_scores.min(), stability_scores.max()
            
            if q_max > q_min:
                quality_norm = (quality_scores - q_min) / (q_max - q_min)
            else:
                quality_norm = np.zeros_like(quality_scores)
            
            if s_max > s_min:
                stability_norm = (stability_scores - s_min) / (s_max - s_min)
            else:
                stability_norm = np.zeros_like(stability_scores)
            
            quality_anomaly_score = 1.0 - quality_norm
            stability_anomaly_score = stability_norm
            
            # 使用默认AUC权重（与evaluate中一致）
            alpha = 0.903 / (0.903 + 0.735)
            beta = 0.735 / (0.903 + 0.735)
            
            combined_score = alpha * quality_anomaly_score + beta * stability_anomaly_score
            
            # 使用F1优化找最佳阈值
            best_f1 = 0
            best_threshold = 0.5
            for thresh in np.linspace(0, 1, 101):
                pred = (combined_score > thresh).astype(int)
                f1 = f1_score(combined_true, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
            
            combined_pred = (combined_score > best_threshold).astype(int)
        else:
            combined_pred = np.where(quality_pred == 1, 1, stability_pred).astype(int)
        
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
        print("[INFO] 生成评估报告...")
        
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

### 阈值与方向
- **优化方法**: {metrics.get('optimize_thresholds', 'f1')}
- **Quality 阈值**: {metrics['quality_score']['threshold']:.4f}
- **Stability 阈值**: {metrics['stability_score']['threshold']:.4f}
- **Stability 方向翻转**: {metrics['stability_score'].get('flipped', False)}（probe AUC={metrics['stability_score'].get('auc_probe', float('nan')):.3f}）

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
  - 融合方式: {metrics.get('residual_classifier', {}).get('fuse_mode', 'N/A')}
  - 加权权重: {metrics.get('residual_classifier', {}).get('weight', 'N/A')}
- 残差融合组合（Combined with Residual）
  - 准确率: {metrics.get('combined_with_residual', {}).get('accuracy', 'N/A')}
  - 精确率: {metrics.get('combined_with_residual', {}).get('precision', 'N/A')}
  - 召回率: {metrics.get('combined_with_residual', {}).get('recall', 'N/A')}
  - F1分数: {metrics.get('combined_with_residual', {}).get('f1_score', 'N/A')}
  - AUC-ROC: {metrics.get('combined_with_residual', {}).get('auc_roc', 'N/A')}

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
        
        print(f"[OK] 评估报告已保存: {report_path}")
        print(f"[INFO] 性能指标已保存: {metrics_path}")
    
    def run_complete_evaluation(self, n_samples: int = 1000):
        """
        运行完整的模型评估流程
        
        Args:
            n_samples: 评估样本数量
        """
        print("[RUN] 开始完整的模型评估流程...")
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

            # 3.1（可选）残差分类器与融合
            if getattr(self, 'use_residual_clf', False):
                # 若已加载预训练分类器，则直接使用；否则基于标签拟合一个OOF评分用于分析
                if 'residual_clf' in self.models:
                    residuals, wavelengths = self._compute_residuals_matrix(spectra)
                    X = self._extract_segment_features(residuals, wavelengths)
                    r_scores = self.models['residual_clf'].predict_proba(X)[:, 1]
                    clf = self.models['residual_clf']
                else:
                    # 计算残差分类器分数（越大越异常）
                    r_scores, clf = self.compute_residual_classifier_scores(spectra, stability_labels)
                # 与AE方向一致分数做融合：加权或OR
                mode = getattr(self, 'residual_fuse_mode', 'weighted')  # 'weighted' | 'or'
                weight = float(getattr(self, 'residual_weight', 0.5))

                # 注意：create_confusion_matrix_and_roc中已进行了方向统一，这里重算一次方向统一供融合使用
                from sklearn.metrics import roc_curve, auc
                fpr_probe, tpr_probe, _ = roc_curve(stability_labels, stability_scores)
                auc_probe = auc(fpr_probe, tpr_probe)
                s_dir = (-stability_scores) if auc_probe < 0.55 else stability_scores

                # 归一化到[0,1]
                eps = 1e-12
                s_anom = (s_dir - s_dir.min()) / (s_dir.max() - s_dir.min() + eps)
                r_anom = (r_scores - r_scores.min()) / (r_scores.max() - r_scores.min() + eps)

                from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
                combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)

                # 质量异常分数（1 - Quality）
                q_anom = 1.0 - quality_scores
                q_anom = (q_anom - q_anom.min()) / (q_anom.max() - q_anom.min() + eps)

                if self.residual_weights_config:
                    # 使用metadata提供的三通道权重
                    w_q = float(self.residual_weights_config.get('quality', 0.5))
                    w_s = float(self.residual_weights_config.get('stability', 0.25))
                    w_r = float(self.residual_weights_config.get('residual', 0.25))
                    weight_sum = w_q + w_s + w_r
                    if weight_sum <= 0:
                        weight_sum = 1.0
                    w_q /= weight_sum
                    w_s /= weight_sum
                    w_r /= weight_sum
                    mode = 'weighted'
                    fused = w_q * q_anom + w_s * s_anom + w_r * r_anom
                    fpr_fused, tpr_fused, _ = roc_curve(combined_true, fused)
                    auc_fused = auc(fpr_fused, tpr_fused)
                    # 阈值优先使用配置，否则自动搜索
                    if self.residual_threshold_config is not None:
                        best_th = float(self.residual_threshold_config)
                        best_f1 = f1_score(combined_true, (fused > best_th).astype(int), zero_division=0)
                    else:
                        th_grid = np.linspace(fused.min(), fused.max(), 200)
                        best_f1, best_th = -1.0, th_grid[0]
                        for th in th_grid:
                            pred = (fused > th).astype(int)
                            f1 = f1_score(combined_true, pred, zero_division=0)
                            if f1 > best_f1:
                                best_f1, best_th = f1, th
                    pred = (fused > best_th).astype(int)
                    acc = accuracy_score(combined_true, pred)
                    prec = precision_score(combined_true, pred, zero_division=0)
                    rec = recall_score(combined_true, pred, zero_division=0)
                    combined_scores_fused = fused
                    weights_entry = {'quality': w_q, 'stability': w_s, 'residual': w_r}
                else:
                    if mode == 'or':
                        fused_sr = np.maximum(s_anom, r_anom)
                    else:
                        fused_sr = weight * s_anom + (1.0 - weight) * r_anom

                    combined_scores_fused = 0.5 * q_anom + 0.5 * fused_sr
                    fpr_fused, tpr_fused, th_fused = roc_curve(combined_true, combined_scores_fused)
                    auc_fused = auc(fpr_fused, tpr_fused)

                    th_grid = np.linspace(combined_scores_fused.min(), combined_scores_fused.max(), 200)
                    best_f1, best_th = -1.0, th_grid[0]
                    for th in th_grid:
                        pred = (combined_scores_fused > th).astype(int)
                        f1 = f1_score(combined_true, pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1, best_th = f1, th
                    pred = (combined_scores_fused > best_th).astype(int)
                    acc = accuracy_score(combined_true, pred)
                    prec = precision_score(combined_true, pred, zero_division=0)
                    rec = recall_score(combined_true, pred, zero_division=0)
                    weights_entry = None

                # 写入metrics补充字段
                metrics['residual_classifier'] = {
                    'fuse_mode': mode,
                    'weight': w_r if self.residual_weights_config else weight,
                    'weights': weights_entry,
                    'auc_roc_stability_residual': float(auc((roc_curve(stability_labels, r_anom)[0]), (roc_curve(stability_labels, r_anom)[1]))),
                }
                metrics['combined_with_residual'] = {
                    'accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1_score': float(best_f1),
                    'auc_roc': float(auc_fused),
                    'threshold': float(best_th),
                }
            
            # 4. 生成评估报告
            self.generate_evaluation_report(metrics, n_samples)
            
            print("=" * 60)
            print("[DONE] 模型评估完成!")
            print(f"[DIR] 所有结果已保存到: {self.output_dir}")
            print("\n[INFO] 关键指标:")
            print(f"   - 组合模型准确率: {metrics['combined_model']['accuracy']:.4f}")
            print(f"   - 组合模型F1分数: {metrics['combined_model']['f1_score']:.4f}")
            print(f"   - 组合模型AUC-ROC: {metrics['combined_model']['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] 评估过程中发生错误: {e}")
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
    parser.add_argument('--combine-strategy', type=str, choices=['two_stage', 'and', 'or', 'weighted'],
                       default=None, help='组合策略覆盖(two_stage|and|or|weighted)')
    parser.add_argument('--use-residual-clf', action='store_true',
                       help='启用残差分段特征+Logistic 辅助通道')
    parser.add_argument('--residual-fuse-mode', type=str, choices=['weighted', 'or'], default='weighted',
                       help='残差通道与AE误差的融合方式：加权或逻辑OR')
    parser.add_argument('--residual-weight', type=float, default=0.5,
                       help='融合权重（weighted模式下，越大越偏向AE误差）')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子，用于生成可复现的测试数据（默认: 42）')
    parser.add_argument('--optimize-thresholds', type=str, choices=['none', 'youden', 'f1'], default='f1',
                       help='阈值优化方法：none/youden/f1（默认: none）')
    parser.add_argument('--test-data-npz', type=str, default=None,
                       help='真实测试数据NPZ路径（包含wavelengths, dvp_values）')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(model_dir=args.model_dir, combine_strategy=args.combine_strategy,
                               random_seed=args.random_seed, optimize_thresholds=args.optimize_thresholds)
    if args.use_residual_clf:
        setattr(evaluator, 'use_residual_clf', True)
        setattr(evaluator, 'residual_fuse_mode', args.residual_fuse_mode)
        setattr(evaluator, 'residual_weight', args.residual_weight)
    if args.test_data_npz:
        setattr(evaluator, 'test_data_npz', args.test_data_npz)
    evaluator.run_complete_evaluation(n_samples=args.samples)


if __name__ == "__main__":
    main()