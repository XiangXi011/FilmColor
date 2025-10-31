#!/usr/bin/env python3
"""
模型训练脚本 - train.py
完整的DVP涂层类型光谱异常检测模型训练流程

功能：
- 支持命令行参数指定coating_name
- 整合SimilarityEvaluator和WeightedAutoencoder
- 实现模型保存功能(.h5, .pkl, .json)
- 阈值计算(99.5%分位数)
- 多产品类型训练支持
- 完整的训练日志和报告
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/workspace/code/spectrum_anomaly_detection')

# 导入自定义模块
from algorithms.similarity_evaluator import SimilarityEvaluator
# from models.weighted_autoencoder import WeightedAutoencoder  # 暂时注释掉TensorFlow版本

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpectrumAnomalyTrainer:
    """
    光谱异常检测模型训练器
    
    整合Quality Score和Stability Score的训练流程
    """
    
    def __init__(self, coating_name: str = "DVP", version: str = "v1.0"):
        """
        初始化训练器
        
        Args:
            coating_name: 涂层名称
            version: 模型版本号
        """
        self.coating_name = coating_name
        self.version = version
        
        # 初始化组件
        self.similarity_evaluator = SimilarityEvaluator(coating_name)
        # self.autoencoder = WeightedAutoencoder(input_dim=81, coating_name=coating_name)  # 暂时注释掉TensorFlow版本
        
        # 训练配置
        self.config = {
            'training_samples': 200,
            'validation_split': 0.2,
            'noise_levels': [0.05, 0.1, 0.15, 0.2],
            'quality_threshold': 92.0,  # 提高默认Quality Score阈值，降低误报
            'stability_threshold_percentile': 99.9,  # 提高默认分位阈值，降低误报
            'stability_threshold_method': 'percentile',  # 'percentile' | 'mad'
            'stability_mad_k': 3.5,  # 当使用MAD法时的倍数
            # 权重相关配置（可通过命令行覆盖）
            'weights_base_range': [400.0, 680.0],  # 基础加权区间
            'weights_base_value': 3.0,            # 基础权重值
            'weights_peak_range': [400.0, 560.0], # 重点增强区间（微调上限）
            'weights_peak_multiplier': 1.5,       # 重点区间倍率
            'weights_mode': 'static',             # static | variance | variance_smooth | hybrid
            'weights_mix_alpha': 0.7,             # hybrid: w = alpha*static + (1-alpha)*variance_smooth
            'weights_smooth_pctl_lo': 5.0,        # variance_smooth: winsorize下限百分位
            'weights_smooth_pctl_hi': 95.0,       # variance_smooth: winsorize上限百分位
            'weights_smooth_window': 5,           # variance_smooth: 滑动窗口大小(奇数)
            'random_seed': 42,
            # AE默认固化配置（无需每次传参）
            'ae_encoder_layers': [64, 32, 16, 8],
            'ae_decoder_layers': [16, 32, 64, 81],
            'ae_alpha': 2e-4,
            'ae_early_stopping': True,
            'ae_n_iter_no_change': 12,
            'ae_validation_fraction': 0.1
        }
        
        logger.info(f"SpectrumAnomalyTrainer 初始化完成 [{coating_name} {version}]")
    
    def _prepare_weights(self, wavelengths: np.ndarray, coating_name: str) -> np.ndarray:
        """
        准备权重向量（与WeightedAutoencoder中的方法相同）
        
        Args:
            wavelengths: 波长数组
            coating_name: 涂层名称
            
        Returns:
            np.ndarray: 权重向量
        """
        weights = np.ones_like(wavelengths, dtype=np.float64)
        
        # 基础权重区间与数值
        base_lo, base_hi = self.config.get('weights_base_range', [400.0, 680.0])
        base_val = float(self.config.get('weights_base_value', 3.0))
        mask = (wavelengths >= base_lo) & (wavelengths <= base_hi)
        weights[mask] = base_val
        
        # 重点增强区间
        if coating_name == "DVP":
            peak_lo, peak_hi = self.config.get('weights_peak_range', [400.0, 560.0])
            peak_mul = float(self.config.get('weights_peak_multiplier', 1.5))
            peak_mask = (wavelengths >= peak_lo) & (wavelengths <= peak_hi)
            weights[peak_mask] *= peak_mul
        
        # 归一化权重
        weights = weights / np.mean(weights)
        
        return weights
    def _prepare_weights(self, wavelengths: np.ndarray, coating_name: str) -> np.ndarray:
        """
        准备权重向量（与WeightedAutoencoder中的方法相同）
        
        Args:
            wavelengths: 波长数组
            coating_name: 涂层名称
            
        Returns:
            np.ndarray: 权重向量
        """
        # 按配置选择权重模式
        mode = (self.config.get('weights_mode') or 'static').lower()
        if mode in ('variance', 'variance_smooth', 'hybrid'):
            # 基于训练数据NPZ矩阵的波长方差构造权重：w ∝ 1/(std+eps)
            try:
                project_root = Path(__file__).parent.parent
                default_path = str(project_root / "data" / "dvp_processed_data.npz")
                data_path = getattr(self, 'data_npz_path', None) or default_path
                data = np.load(data_path)
                spectra = data['dvp_values']
                if spectra.ndim == 2 and spectra.shape[1] == len(wavelengths):
                    std = np.std(spectra, axis=0)
                    # winsorize截断
                    if mode in ('variance_smooth', 'hybrid'):
                        lo = float(self.config.get('weights_smooth_pctl_lo', 5.0))
                        hi = float(self.config.get('weights_smooth_pctl_hi', 95.0))
                        lo_v = np.percentile(std, lo)
                        hi_v = np.percentile(std, hi)
                        std = np.clip(std, lo_v, hi_v)
                    eps = 1e-8
                    inv_std = 1.0 / (std + eps)
                    weights_var = inv_std.astype(np.float64)
                    # 平滑
                    if mode in ('variance_smooth', 'hybrid'):
                        win = int(self.config.get('weights_smooth_window', 5))
                        win = max(1, win)
                        if win > 1:
                            # 简单移动平均
                            kernel = np.ones(win, dtype=np.float64) / win
                            pad = win // 2
                            padded = np.pad(weights_var, (pad, pad), mode='edge')
                            weights_var = np.convolve(padded, kernel, mode='valid')
                    # 归一化
                    weights_var = weights_var / np.mean(weights_var)
                    if mode == 'hybrid':
                        # 计算static权重
                        w_static = np.ones_like(wavelengths, dtype=np.float64)
                        base_lo, base_hi = self.config.get('weights_base_range', [400.0, 680.0])
                        base_val = float(self.config.get('weights_base_value', 3.0))
                        mask = (wavelengths >= base_lo) & (wavelengths <= base_hi)
                        w_static[mask] = base_val
                        if coating_name == "DVP":
                            peak_lo, peak_hi = self.config.get('weights_peak_range', [400.0, 560.0])
                            peak_mul = float(self.config.get('weights_peak_multiplier', 1.5))
                            peak_mask = (wavelengths >= peak_lo) & (wavelengths <= peak_hi)
                            w_static[peak_mask] *= peak_mul
                        w_static = w_static / np.mean(w_static)
                        alpha = float(self.config.get('weights_mix_alpha', 0.7))
                        weights = alpha * w_static + (1.0 - alpha) * weights_var
                        weights = weights / np.mean(weights)
                        return weights
                    else:
                        return weights_var
            except Exception:
                pass  # 回退到static

        weights = np.ones_like(wavelengths, dtype=np.float64)
        
        # 基础权重: 400-680nm范围权重为3
        mask = (wavelengths >= 400) & (wavelengths <= 680)
        weights[mask] = 3.0
        
        # 根据涂层类型调整权重
        if coating_name == "DVP":
            # DVP涂层: 增强400-560nm波段的权重
            peak_mask = (wavelengths >= 400) & (wavelengths <= 560)
            weights[peak_mask] *= 1.5
        
        # 归一化权重
        weights = weights / np.mean(weights)
        
        return weights
    def load_standard_curve(self) -> tuple:
        """
        加载标准曲线数据
        
        Returns:
            tuple: (波长数组, 标准光谱数组)
        """
        # 使用项目相对路径，或命令行传入的自定义NPZ路径
        project_root = Path(__file__).parent.parent
        default_path = str(project_root / "data" / "dvp_processed_data.npz")
        data_path = getattr(self, 'data_npz_path', None) or default_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"标准曲线数据文件不存在: {data_path}")
        
        data = np.load(data_path)
        # 基本数据校验
        if 'wavelengths' not in data or 'dvp_values' not in data:
            raise ValueError("NPZ缺少必须字段: wavelengths 或 dvp_values")
        wavelengths = data['wavelengths']
        if np.isnan(wavelengths).any():
            raise ValueError("wavelengths 含有 NaN")
        
        # 根据涂层名称选择对应数据矩阵（样本数 × 波长数）
        if self.coating_name == "DVP":
            spectra = data['dvp_values']
        else:
            spectra = data['dvp_values']
            logger.warning(f"涂层 {self.coating_name} 使用DVP数据作为示例")

        # 当输入为样本矩阵时，需汇聚为一条标准曲线（长度==波长数）
        if spectra.ndim == 2:
            if spectra.shape[1] != len(wavelengths):
                raise ValueError(f"光谱矩阵列数({spectra.shape[1]})与波长数({len(wavelengths)})不一致")
            if np.isnan(spectra).any():
                raise ValueError("光谱矩阵含有 NaN")
            # 采用中位数更稳健（防异常值），也可切换为均值
            agg = getattr(self, 'std_curve_agg', 'median')
            if agg == 'mean':
                standard_spectrum = np.mean(spectra, axis=0)
            else:
                standard_spectrum = np.median(spectra, axis=0)
            logger.info(f"从 {spectra.shape[0]} 条样本汇聚得到标准曲线（使用 {agg}）")
        else:
            standard_spectrum = spectra
        
        logger.info(f"标准曲线加载完成: {len(wavelengths)}个波长点 | 来源: {data_path}")
        return wavelengths, standard_spectrum
    
    def generate_training_data(self, standard_spectrum: np.ndarray, 
                             wavelengths: np.ndarray) -> tuple:
        """
        生成训练数据
        
        Args:
            standard_spectrum: 标准光谱
            wavelengths: 波长数组
            
        Returns:
            tuple: (训练光谱, 验证光谱, 标签)
        """
        np.random.seed(self.config['random_seed'])
        
        # 生成正常样本（低噪声）
        normal_spectra = []
        normal_labels = []
        
        n_samples = self.config['training_samples']
        noise_levels = self.config['noise_levels']
        
        for i in range(n_samples):
            # 随机选择噪声水平
            noise_level = np.random.choice(noise_levels)
            
            # 生成噪声
            noise = np.random.normal(0, noise_level, len(standard_spectrum))
            
            # 生成光谱
            spectrum = standard_spectrum + noise
            
            normal_spectra.append(spectrum)
            normal_labels.append(1)  # 1表示正常
        
        # 生成一些异常样本（高噪声）用于验证
        anomaly_spectra = []
        anomaly_labels = []
        
        for i in range(50):  # 生成50个异常样本
            # 高噪声水平
            noise_level = np.random.uniform(0.5, 1.5)
            
            # 生成噪声
            noise = np.random.normal(0, noise_level, len(standard_spectrum))
            
            # 生成光谱
            spectrum = standard_spectrum + noise
            
            anomaly_spectra.append(spectrum)
            anomaly_labels.append(0)  # 0表示异常
        
        # 合并数据
        all_spectra = np.array(normal_spectra + anomaly_spectra)
        all_labels = np.array(normal_labels + anomaly_labels)
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            all_spectra, all_labels, 
            test_size=self.config['validation_split'], 
            random_state=self.config['random_seed'],
            stratify=all_labels
        )
        
        logger.info(f"训练数据生成完成:")
        logger.info(f"  - 训练集: {len(X_train)}个样本 (正常{y_train.sum()}, 异常{len(y_train)-y_train.sum()})")
        logger.info(f"  - 验证集: {len(X_val)}个样本 (正常{y_val.sum()}, 异常{len(y_val)-y_val.sum()})")
        
        return X_train, X_val, y_train, y_val
    
    def train_similarity_evaluator(self, X_train: np.ndarray, X_val: np.ndarray,
                                 standard_spectrum: np.ndarray, wavelengths: np.ndarray) -> dict:
        """
        训练SimilarityEvaluator（实际上不需要训练，只是配置）
        
        Args:
            X_train: 训练光谱
            X_val: 验证光谱
            standard_spectrum: 标准光谱
            wavelengths: 波长数组
            
        Returns:
            dict: 评估结果
        """
        logger.info("配置SimilarityEvaluator...")
        
        # 对训练集和验证集进行Quality Score评估
        train_results = self.similarity_evaluator.batch_evaluate(
            X_train, standard_spectrum, wavelengths, self.coating_name
        )
        
        val_results = self.similarity_evaluator.batch_evaluate(
            X_val, standard_spectrum, wavelengths, self.coating_name
        )
        
        # 计算Quality Score阈值
        quality_threshold = self.config['quality_threshold']
        
        # 统计正常样本的Quality Score分布
        normal_train_scores = train_results[train_results['spectrum_id'] < len(X_train)//2]['similarity_score_percent']
        normal_val_scores = val_results[val_results['spectrum_id'] < len(X_val)//2]['similarity_score_percent']
        
        all_normal_scores = np.concatenate([normal_train_scores, normal_val_scores])
        
        evaluator_result = {
            'training_scores': train_results.to_dict('records'),
            'validation_scores': val_results.to_dict('records'),
            'quality_threshold': quality_threshold,
            'normal_scores_stats': {
                'mean': float(all_normal_scores.mean()),
                'std': float(all_normal_scores.std()),
                'min': float(all_normal_scores.min()),
                'max': float(all_normal_scores.max()),
                'percentile_5': float(np.percentile(all_normal_scores, 5)),
                'percentile_95': float(np.percentile(all_normal_scores, 95))
            }
        }
        
        logger.info(f"SimilarityEvaluator配置完成:")
        logger.info(f"  - Quality Score阈值: {quality_threshold}%")
        logger.info(f"  - 正常样本平均分数: {evaluator_result['normal_scores_stats']['mean']:.2f}%")
        
        return evaluator_result
    
    def train_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray, 
                         y_train: np.ndarray, wavelengths: np.ndarray) -> dict:
        """
        训练WeightedAutoencoder
        
        Args:
            X_train: 训练光谱
            X_val: 验证光谱
            wavelengths: 波长数组
            
        Returns:
            dict: 训练结果
        """
        logger.info("开始训练WeightedAutoencoder...")
        
        # 只使用正常样本训练自编码器
        normal_train = X_train[y_train == 1]
        
        # 使用简化的训练方法（基于Phase 3的测试结果）
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        
        # 数据预处理
        scaler = StandardScaler()
        normal_train_scaled = scaler.fit_transform(normal_train)
        
        # 划分训练集和验证集
        X_train_ae, X_val_ae = train_test_split(
            normal_train_scaled, 
            test_size=0.2, 
            random_state=self.config['random_seed']
        )
        
        # 准备权重
        weights = self._prepare_weights(wavelengths, self.coating_name)
        
        # 构建自编码器
        # 可配置的网络结构与正则化
        enc_layers = tuple(self.config.get('ae_encoder_layers', [64, 32, 16, 8]))
        dec_layers = tuple(self.config.get('ae_decoder_layers', [16, 32, 64, 81]))
        alpha = float(self.config.get('ae_alpha', 1e-4))
        early_stopping = bool(self.config.get('ae_early_stopping', True))
        n_iter_no_change = int(self.config.get('ae_n_iter_no_change', 10))
        val_fraction = float(self.config.get('ae_validation_fraction', 0.1))

        encoder = MLPRegressor(
            hidden_layer_sizes=enc_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=400,
            alpha=alpha,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=val_fraction,
            random_state=self.config['random_seed']
        )
        
        decoder = MLPRegressor(
            hidden_layer_sizes=dec_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=400,
            alpha=alpha,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=val_fraction,
            random_state=self.config['random_seed']
        )
        
        # 训练编码器
        logger.info("训练编码器...")
        encoder.fit(X_train_ae, X_train_ae)
        
        # 训练解码器
        logger.info("训练解码器...")
        latent_train = encoder.predict(X_train_ae)
        decoder.fit(latent_train, X_train_ae)
        
        # 计算重构误差
        def calculate_reconstruction_errors(X):
            errors = []
            for i in range(len(X)):
                latent = encoder.predict(X[i].reshape(1, -1))
                reconstructed = decoder.predict(latent)
                error = np.mean(weights * (X[i] - reconstructed[0]) ** 2)
                errors.append(error)
            return np.array(errors)
        
        train_errors = calculate_reconstruction_errors(X_train_ae)
        val_errors = calculate_reconstruction_errors(X_val_ae)
        
        # 计算阈值（支持分位数或MAD稳健方法）
        method = self.config.get('stability_threshold_method', 'percentile')
        if method == 'mad':
            median = np.median(val_errors)
            mad = np.median(np.abs(val_errors - median)) + 1e-12
            k = float(self.config.get('stability_mad_k', 3.5))
            stability_threshold = float(median + k * 1.4826 * mad)
            threshold_percentile = None
        else:
            threshold_percentile = float(self.config['stability_threshold_percentile'])
            stability_threshold = float(np.percentile(val_errors, threshold_percentile))
        
        autoencoder_result = {
            'encoder_score': float(encoder.score(X_train_ae, X_train_ae)),
            'decoder_score': float(decoder.score(latent_train, X_train_ae)),
            'train_errors': train_errors.tolist(),
            'val_errors': val_errors.tolist(),
            'stability_threshold': float(stability_threshold),
            'threshold_method': method,
            'threshold_percentile': threshold_percentile,
            'error_stats': {
                'train_mean': float(train_errors.mean()),
                'train_std': float(train_errors.std()),
                'val_mean': float(val_errors.mean()),
                'val_std': float(val_errors.std())
            }
        }
        
        logger.info(f"WeightedAutoencoder训练完成:")
        logger.info(f"  - 编码器R²: {autoencoder_result['encoder_score']:.4f}")
        logger.info(f"  - 解码器R²: {autoencoder_result['decoder_score']:.4f}")
        if method == 'mad':
            logger.info(f"  - 稳定性阈值: {stability_threshold:.6f} (MAD法 k={self.config.get('stability_mad_k', 3.5)})")
        else:
            logger.info(f"  - 稳定性阈值: {stability_threshold:.6f} ({threshold_percentile}%分位数)")
        
        return autoencoder_result, encoder, decoder, scaler, weights
    
    def save_trained_models(self, encoder, decoder, scaler, weights,
                          evaluator_result: dict, autoencoder_result: dict) -> dict:
        """
        保存训练好的模型
        
        Args:
            encoder: 训练好的编码器
            decoder: 训练好的解码器
            scaler: 训练好的预处理器
            weights: 使用的权重向量
            evaluator_result: SimilarityEvaluator结果
            autoencoder_result: WeightedAutoencoder结果
            
        Returns:
            dict: 保存的文件路径
        """
        # 创建保存目录
        model_dir = Path("/workspace/code/spectrum_anomaly_detection/models")
        coating_dir = model_dir / self.coating_name / self.version
        coating_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存编码器
        encoder_path = coating_dir / f"encoder_{self.coating_name}_{self.version}.joblib"
        import joblib
        joblib.dump(encoder, encoder_path)
        saved_files['encoder'] = str(encoder_path)
        
        # 保存解码器
        decoder_path = coating_dir / f"decoder_{self.coating_name}_{self.version}.joblib"
        joblib.dump(decoder, decoder_path)
        saved_files['decoder'] = str(decoder_path)
        
        # 保存预处理器
        scaler_path = coating_dir / f"scaler_{self.coating_name}_{self.version}.joblib"
        joblib.dump(scaler, scaler_path)
        saved_files['scaler'] = str(scaler_path)
        
        # 保存权重
        weights_path = coating_dir / f"weights_{self.coating_name}_{self.version}.npy"
        np.save(weights_path, weights)
        saved_files['weights'] = str(weights_path)
        
        # 保存元数据
        metadata = {
            'coating_name': self.coating_name,
            'version': self.version,
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'similarity_evaluator': {
                'quality_threshold': evaluator_result['quality_threshold'],
                'normal_scores_stats': evaluator_result['normal_scores_stats']
            },
            'weighted_autoencoder': {
                'stability_threshold': autoencoder_result['stability_threshold'],
                'threshold_method': autoencoder_result.get('threshold_method', 'percentile'),
                'threshold_percentile': autoencoder_result.get('threshold_percentile'),
                'error_stats': autoencoder_result['error_stats']
            },
            'combine_strategy': 'two_stage',
            'model_architecture': {
                'input_dim': 81,
                'encoder_dims': [48, 16, 4],
                'decoder_dims': [16, 48, 81],
                'latent_dim': 4
            }
        }
        # 记录权重模式与关键参数
        metadata['weighted_autoencoder']['weights_mode'] = self.config.get('weights_mode', 'static')
        metadata['weighted_autoencoder']['weights_params'] = {
            'mix_alpha': self.config.get('weights_mix_alpha', 0.7),
            'smooth_pctl_lo': self.config.get('weights_smooth_pctl_lo', 5.0),
            'smooth_pctl_hi': self.config.get('weights_smooth_pctl_hi', 95.0),
            'smooth_window': self.config.get('weights_smooth_window', 5),
            'base_range': self.config.get('weights_base_range', [400.0, 680.0]),
            'base_value': self.config.get('weights_base_value', 3.0),
            'peak_range': self.config.get('weights_peak_range', [400.0, 560.0]),
            'peak_multiplier': self.config.get('weights_peak_multiplier', 1.5),
        }
        
        metadata_path = coating_dir / f"metadata_{self.coating_name}_{self.version}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        logger.info(f"模型保存完成: {coating_dir}")
        return saved_files
    
    def generate_training_report(self, evaluator_result: dict, autoencoder_result: dict,
                               saved_files: dict, training_time: float) -> dict:
        """
        生成训练报告
        
        Args:
            evaluator_result: SimilarityEvaluator结果
            autoencoder_result: WeightedAutoencoder结果
            saved_files: 保存的文件路径
            training_time: 训练时间
            
        Returns:
            dict: 训练报告
        """
        report = {
            'training_summary': {
                'coating_name': self.coating_name,
                'version': self.version,
                'training_date': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'status': 'COMPLETED'
            },
            'similarity_evaluator': {
                'quality_threshold': evaluator_result['quality_threshold'],
                'normal_scores_distribution': evaluator_result['normal_scores_stats'],
                'description': '基于专家规则的质量评估，阈值用于判断光谱是否符合标准'
            },
            'weighted_autoencoder': {
                'stability_threshold': autoencoder_result['stability_threshold'],
                'threshold_percentile': autoencoder_result['threshold_percentile'],
                'error_distribution': autoencoder_result['error_stats'],
                'description': '基于机器学习的稳定性评估，阈值用于检测过程异常'
            },
            'model_files': saved_files,
            'next_steps': [
                'Phase 5: 模型评估和可视化系统',
                'Phase 6: 决策引擎和API接口',
                '部署到生产环境'
            ]
        }
        
        # 保存报告
        report_path = Path("/workspace/code/spectrum_anomaly_detection/output") / f"training_report_{self.coating_name}_{self.version}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存: {report_path}")
        return report
    
    def train(self) -> dict:
        """
        执行完整的训练流程
        
        Returns:
            dict: 训练结果和报告
        """
        start_time = datetime.now()
        logger.info(f"开始训练流程: {self.coating_name} {self.version}")
        
        try:
            # 步骤1: 加载标准曲线
            logger.info("步骤1: 加载标准曲线")
            wavelengths, standard_spectrum = self.load_standard_curve()
            
            # 步骤2: 生成训练数据
            logger.info("步骤2: 生成训练数据")
            X_train, X_val, y_train, y_val = self.generate_training_data(standard_spectrum, wavelengths)
            
            # 步骤3: 训练SimilarityEvaluator
            logger.info("步骤3: 配置SimilarityEvaluator")
            evaluator_result = self.train_similarity_evaluator(X_train, X_val, standard_spectrum, wavelengths)
            
            # 步骤4: 训练WeightedAutoencoder
            logger.info("步骤4: 训练WeightedAutoencoder")
            autoencoder_result, encoder, decoder, scaler, weights = self.train_autoencoder(
                X_train, X_val, y_train, wavelengths
            )
            
            # 步骤5: 保存模型
            logger.info("步骤5: 保存模型")
            saved_files = self.save_trained_models(
                encoder, decoder, scaler, weights, evaluator_result, autoencoder_result
            )
            
            # 步骤6: 生成训练报告
            logger.info("步骤6: 生成训练报告")
            training_time = (datetime.now() - start_time).total_seconds()
            report = self.generate_training_report(
                evaluator_result, autoencoder_result, saved_files, training_time
            )
            
            logger.info(f"训练流程完成，耗时 {training_time:.2f} 秒")
            
            return {
                'status': 'SUCCESS',
                'report': report,
                'saved_files': saved_files,
                'training_time': training_time
            }
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'FAILED',
                'error': str(e),
                'training_time': (datetime.now() - start_time).total_seconds()
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='光谱异常检测模型训练脚本')
    parser.add_argument('--coating_name', type=str, default='DVP', 
                       help='涂层名称 (默认: DVP)')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='模型版本号 (默认: v1.0)')
    parser.add_argument('--training_samples', type=int, default=200,
                       help='训练样本数量 (默认: 200)')
    parser.add_argument('--quality_threshold', type=float, default=92.0,
                       help='Quality Score阈值 (默认: 92.0)')
    parser.add_argument('--stability_threshold_percentile', type=float, default=99.9,
                       help='Stability Score阈值百分位 (默认: 99.9)')
    parser.add_argument('--stability_threshold_method', type=str, default='percentile',
                       choices=['percentile', 'mad'],
                       help='稳定性阈值方法：percentile 或 mad (默认: percentile)')
    parser.add_argument('--stability_mad_k', type=float, default=3.5,
                       help='MAD法的k倍数 (默认: 3.5)')
    parser.add_argument('--std_curve_agg', type=str, choices=['median','mean'], default='median',
                       help='标准曲线汇聚方法（median/mean），默认median')
    parser.add_argument('--data-npz', type=str, default=None,
                       help='自定义训练数据NPZ路径（包含 wavelengths, dvp_values）')
    parser.add_argument('--weights-mode', type=str, default='static', choices=['static','variance','variance_smooth','hybrid'],
                       help='权重模式：static 固定；variance 方差倒数；variance_smooth 方差倒数+截断+平滑；hybrid 混合')
    parser.add_argument('--weights-mix-alpha', type=float, default=0.7,
                       help='hybrid混合权重α，越大越偏向static (默认0.7)')
    parser.add_argument('--weights-smooth-pctl-lo', type=float, default=5.0,
                       help='variance_smooth 截断下百分位 (默认5)')
    parser.add_argument('--weights-smooth-pctl-hi', type=float, default=95.0,
                       help='variance_smooth 截断上百分位 (默认95)')
    parser.add_argument('--weights-smooth-window', type=int, default=5,
                       help='variance_smooth 平滑窗口(奇数，默认5)')
    # AE参数
    parser.add_argument('--ae-encoder-layers', type=int, nargs='+', default=None,
                       help='AE编码器层，如 64 32 16 8')
    parser.add_argument('--ae-decoder-layers', type=int, nargs='+', default=None,
                       help='AE解码器层，如 16 32 64 81')
    parser.add_argument('--ae-alpha', type=float, default=None, help='AE L2正则alpha')
    parser.add_argument('--ae-early-stopping', action='store_true', help='启用AE early_stopping')
    parser.add_argument('--ae-n-iter-no-change', type=int, default=None, help='early_stopping容忍迭代数')
    parser.add_argument('--ae-validation-fraction', type=float, default=None, help='early_stopping验证集比例')
    # 权重相关
    parser.add_argument('--weights_base_range', type=float, nargs=2, metavar=('LO','HI'),
                       default=None, help='基础权重区间，例如 400 680')
    parser.add_argument('--weights_base_value', type=float, default=None,
                       help='基础权重值，例如 3.0')
    parser.add_argument('--weights_peak_range', type=float, nargs=2, metavar=('LO','HI'),
                       default=None, help='重点增强区间，例如 400 550')
    parser.add_argument('--weights_peak_multiplier', type=float, default=None,
                       help='重点区间倍率，例如 1.5')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = SpectrumAnomalyTrainer(args.coating_name, args.version)
    
    # 更新配置
    trainer.config.update({
        'training_samples': args.training_samples,
        'quality_threshold': args.quality_threshold,
        'stability_threshold_percentile': args.stability_threshold_percentile,
        'stability_threshold_method': args.stability_threshold_method,
        'stability_mad_k': args.stability_mad_k,
    })
    trainer.config['weights_mode'] = (args.weights_mode or 'static')
    trainer.config['weights_mix_alpha'] = args.weights_mix_alpha
    trainer.config['weights_smooth_pctl_lo'] = args.weights_smooth_pctl_lo
    trainer.config['weights_smooth_pctl_hi'] = args.weights_smooth_pctl_hi
    trainer.config['weights_smooth_window'] = args.weights_smooth_window
    # 覆盖权重配置（仅当提供时）
    if args.weights_base_range is not None:
        trainer.config['weights_base_range'] = args.weights_base_range
    if args.weights_base_value is not None:
        trainer.config['weights_base_value'] = args.weights_base_value
    if args.weights_peak_range is not None:
        trainer.config['weights_peak_range'] = args.weights_peak_range
    if args.weights_peak_multiplier is not None:
        trainer.config['weights_peak_multiplier'] = args.weights_peak_multiplier
    
    # 传递标准曲线汇聚方法
    setattr(trainer, 'std_curve_agg', args.std_curve_agg)
    # 自定义数据路径
    if args.data_npz:
        setattr(trainer, 'data_npz_path', args.data_npz)
    # 覆盖AE配置
    if args.ae_encoder_layers is not None:
        trainer.config['ae_encoder_layers'] = args.ae_encoder_layers
    if args.ae_decoder_layers is not None:
        trainer.config['ae_decoder_layers'] = args.ae_decoder_layers
    if args.ae_alpha is not None:
        trainer.config['ae_alpha'] = args.ae_alpha
    if args.ae_early_stopping:
        trainer.config['ae_early_stopping'] = True
    if args.ae_n_iter_no_change is not None:
        trainer.config['ae_n_iter_no_change'] = args.ae_n_iter_no_change
    if args.ae_validation_fraction is not None:
        trainer.config['ae_validation_fraction'] = args.ae_validation_fraction
    logger.info(f"训练配置: {trainer.config} | 标准曲线聚合: {args.std_curve_agg}")
    
    # 执行训练
    result = trainer.train()
    
    # 输出结果
    if result['status'] == 'SUCCESS':
        print("\n" + "="*60)
        print("🎉 训练成功完成！")
        print("="*60)
        print(f"涂层: {args.coating_name}")
        print(f"版本: {args.version}")
        print(f"训练时间: {result['training_time']:.2f} 秒")
        print(f"Quality Score阈值: {result['report']['similarity_evaluator']['quality_threshold']}%")
        print(f"Stability Score阈值: {result['report']['weighted_autoencoder']['stability_threshold']:.6f}")
        print("\n保存的文件:")
        for file_type, file_path in result['saved_files'].items():
            print(f"  - {file_type}: {os.path.basename(file_path)}")
    else:
        print("\n" + "="*60)
        print("❌ 训练失败！")
        print("="*60)
        print(f"错误: {result['error']}")
    
    return result['status'] == 'SUCCESS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)