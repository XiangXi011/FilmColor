"""
Weighted Autoencoder - 加权自编码器模型
专门用于DVP涂层类型的光谱异常检测

实现规格：
- 架构: 81→48→16→4→16→48→81 (全连接网络)
- 损失函数: 加权均方误差 (Weighted MSE)
- 预处理: StandardScaler
- 训练: EarlyStopping回调
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from typing import Dict, Any, Tuple, Optional, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置TensorFlow日志级别
tf.get_logger().setLevel('WARNING')

class WeightedAutoencoder:
    """
    加权自编码器模型
    
    用于DVP涂层类型的光谱异常检测，通过重构误差评估过程稳定性
    """
    
    def __init__(self, input_dim: int = 81, coating_name: str = "DVP"):
        """
        初始化加权自编码器
        
        Args:
            input_dim: 输入维度（波长点数）
            coating_name: 涂层名称，用于权重计算
        """
        self.input_dim = input_dim
        self.coating_name = coating_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 模型配置
        self.encoder_dims = [48, 16, 4]  # 编码器维度
        self.decoder_dims = [16, 48]     # 解码器维度
        self.latent_dim = 4              # 潜在空间维度
        
        # 训练配置
        self.learning_rate = 1e-3
        self.epochs = 100
        self.batch_size = 32
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        
        logger.info(f"WeightedAutoencoder 初始化完成 [{coating_name}]")
        logger.info(f"输入维度: {input_dim}, 潜在维度: {self.latent_dim}")
    
    def build_model(self) -> keras.Model:
        """
        构建加权自编码器模型
        
        Returns:
            keras.Model: 构建的模型
        """
        # 输入层
        input_layer = layers.Input(shape=(self.input_dim,), name='spectrum_input')
        
        # 编码器
        encoded = input_layer
        for i, dim in enumerate(self.encoder_dims):
            encoded = layers.Dense(
                dim, 
                activation='relu', 
                name=f'encoder_layer_{i+1}'
            )(encoded)
        
        # 潜在空间
        latent = layers.Dense(
            self.latent_dim, 
            activation='relu', 
            name='latent_space'
        )(encoded)
        
        # 解码器
        decoded = latent
        for i, dim in enumerate(self.decoder_dims):
            decoded = layers.Dense(
                dim, 
                activation='relu', 
                name=f'decoder_layer_{i+1}'
            )(decoded)
        
        # 输出层（线性激活）
        output_layer = layers.Dense(
            self.input_dim, 
            activation=None, 
            name='reconstructed_spectrum'
        )(decoded)
        
        # 创建模型
        model = keras.Model(input_layer, output_layer, name=f'weighted_autoencoder_{self.coating_name}')
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.weighted_mse_loss,
            metrics=['mse']
        )
        
        logger.info(f"模型构建完成: {model.count_params():,} 参数")
        return model
    
    def weighted_mse_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        自定义加权均方误差损失函数
        
        Args:
            y_true: 真实光谱数据
            y_pred: 重构光谱数据
            
        Returns:
            tf.Tensor: 加权损失值
        """
        # 获取权重（需要在训练时提供）
        # 这里使用全局权重，实际使用时需要在fit方法中传递
        weights = getattr(self, 'current_weights', tf.ones_like(y_true))
        
        # 计算加权均方误差
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = weights * squared_diff
        
        return tf.reduce_mean(weighted_squared_diff)
    
    def prepare_weights(self, wavelengths: np.ndarray, coating_name: str = None) -> np.ndarray:
        """
        准备权重向量（用于损失函数）
        
        Args:
            wavelengths: 波长数组
            coating_name: 涂层名称
            
        Returns:
            np.ndarray: 权重向量
        """
        if coating_name is None:
            coating_name = self.coating_name
        
        # 使用与SimilarityEvaluator相同的权重计算逻辑
        weights = np.ones_like(wavelengths, dtype=np.float64)
        
        # 基础权重: 400-680nm范围权重为3
        mask = (wavelengths >= 400) & (wavelengths <= 680)
        weights[mask] = 3.0
        
        # 根据涂层类型调整权重
        if coating_name == "DVP":
            # DVP涂层: 增强400-550nm波段的权重
            peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
            weights[peak_mask] *= 1.5
        
        # 归一化权重
        weights = weights / np.mean(weights)
        
        logger.debug(f"权重准备完成: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        return weights
    
    def preprocess_data(self, spectra: np.ndarray, wavelengths: np.ndarray, 
                       fit_scaler: bool = True) -> np.ndarray:
        """
        数据预处理
        
        Args:
            spectra: 光谱数据 (n_samples, n_features)
            wavelengths: 波长数组
            fit_scaler: 是否拟合StandardScaler
            
        Returns:
            np.ndarray: 预处理后的数据
        """
        if fit_scaler:
            logger.info("拟合StandardScaler...")
            spectra_scaled = self.scaler.fit_transform(spectra)
        else:
            logger.info("使用已拟合的StandardScaler...")
            spectra_scaled = self.scaler.transform(spectra)
        
        logger.info(f"数据预处理完成: {spectra.shape} -> {spectra_scaled.shape}")
        return spectra_scaled
    
    def train(self, normal_spectra: np.ndarray, wavelengths: np.ndarray,
             validation_split: Optional[float] = None, verbose: int = 1) -> Dict[str, Any]:
        """
        训练加权自编码器模型
        
        Args:
            normal_spectra: 正常光谱数据 (n_samples, n_features)
            wavelengths: 波长数组
            validation_split: 验证集比例
            verbose: 训练详细程度
            
        Returns:
            Dict[str, Any]: 训练历史和结果
        """
        if validation_split is None:
            validation_split = self.validation_split
        
        logger.info(f"开始训练模型: {normal_spectra.shape[0]}个正常样本")
        
        # 准备权重
        weights = self.prepare_weights(wavelengths, self.coating_name)
        
        # 数据预处理
        spectra_scaled = self.preprocess_data(normal_spectra, wavelengths, fit_scaler=True)
        
        # 划分训练集和验证集
        if validation_split > 0:
            X_train, X_val = train_test_split(
                spectra_scaled, 
                test_size=validation_split, 
                random_state=42
            )
            logger.info(f"数据划分: 训练集{len(X_train)}个, 验证集{len(X_val)}个")
        else:
            X_train = spectra_scaled
            X_val = None
            logger.info("使用全部数据训练，无验证集")
        
        # 构建模型
        if self.model is None:
            self.model = self.build_model()
        
        # 设置当前权重（用于损失函数）
        self.current_weights = tf.constant(weights, dtype=tf.float32)
        
        # 准备回调函数
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 训练模型
        logger.info("开始模型训练...")
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        # 标记为已训练
        self.is_trained = True
        
        # 计算训练集和验证集的重构误差
        train_predictions = self.model.predict(X_train, verbose=0)
        train_errors = self.calculate_reconstruction_errors(X_train, train_predictions, weights)
        
        if X_val is not None:
            val_predictions = self.model.predict(X_val, verbose=0)
            val_errors = self.calculate_reconstruction_errors(X_val, val_predictions, weights)
        else:
            val_errors = None
        
        # 计算阈值（99.5%分位数）
        if val_errors is not None:
            threshold = np.percentile(val_errors, 99.5)
            logger.info(f"阈值计算完成: 99.5%分位数 = {threshold:.6f}")
        else:
            threshold = np.percentile(train_errors, 99.5)
            logger.info(f"阈值计算完成: 99.5%分位数 = {threshold:.6f} (基于训练集)")
        
        training_result = {
            'training_history': history.history,
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
            'train_reconstruction_errors': train_errors.tolist(),
            'val_reconstruction_errors': val_errors.tolist() if val_errors is not None else None,
            'threshold': float(threshold),
            'epochs_trained': len(history.history['loss']),
            'weights_used': weights.tolist(),
            'model_config': {
                'input_dim': self.input_dim,
                'encoder_dims': self.encoder_dims,
                'decoder_dims': self.decoder_dims,
                'latent_dim': self.latent_dim,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'coating_name': self.coating_name
            }
        }
        
        logger.info(f"训练完成: {training_result['epochs_trained']}个epoch, 最终损失={training_result['final_train_loss']:.6f}")
        return training_result
    
    def calculate_reconstruction_errors(self, X: np.ndarray, X_pred: np.ndarray, 
                                      weights: np.ndarray) -> np.ndarray:
        """
        计算重构误差（Stability Score）
        
        Args:
            X: 原始光谱数据
            X_pred: 重构光谱数据
            weights: 权重向量
            
        Returns:
            np.ndarray: 重构误差数组
        """
        # 计算每个样本的重构误差
        squared_diff = (X - X_pred) ** 2
        weighted_squared_diff = weights * squared_diff
        reconstruction_errors = np.mean(weighted_squared_diff, axis=1)
        
        return reconstruction_errors
    
    def predict(self, spectra: np.ndarray, wavelengths: np.ndarray) -> Dict[str, Any]:
        """
        预测和异常检测
        
        Args:
            spectra: 待预测的光谱数据 (n_samples, n_features)
            wavelengths: 波长数组
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 数据预处理
        spectra_scaled = self.preprocess_data(spectra, wavelengths, fit_scaler=False)
        
        # 重构
        predictions = self.model.predict(spectra_scaled, verbose=0)
        
        # 计算重构误差
        weights = self.prepare_weights(wavelengths, self.coating_name)
        reconstruction_errors = self.calculate_reconstruction_errors(spectra_scaled, predictions, weights)
        
        # 计算每个波长的重构误差（用于诊断）
        per_wavelength_errors = []
        for i in range(len(spectra)):
            squared_diff = (spectra_scaled[i] - predictions[i]) ** 2
            weighted_squared_diff = weights * squared_diff
            per_wavelength_errors.append(weighted_squared_diff.tolist())
        
        results = {
            'reconstructed_spectra': predictions.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'per_wavelength_errors': per_wavelength_errors,
            'threshold': getattr(self, 'threshold', None),
            'metadata': {
                'coating_name': self.coating_name,
                'num_samples': len(spectra),
                'wavelength_range': f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm"
            }
        }
        
        logger.debug(f"预测完成: {len(spectra)}个样本")
        return results
    
    def save_model(self, model_dir: str, version: str = "v1.0") -> Dict[str, str]:
        """
        保存模型和相关组件
        
        Args:
            model_dir: 模型保存目录
            version: 版本号
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建保存目录
        coating_dir = os.path.join(model_dir, self.coating_name, version)
        os.makedirs(coating_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(coating_dir, f"model_{self.coating_name}_{version}.h5")
        self.model.save(model_path)
        
        # 保存预处理器
        scaler_path = os.path.join(coating_dir, f"scaler_{self.coating_name}_{version}.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # 保存元数据
        metadata = {
            'coating_name': self.coating_name,
            'version': version,
            'input_dim': self.input_dim,
            'encoder_dims': self.encoder_dims,
            'decoder_dims': self.decoder_dims,
            'latent_dim': self.latent_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'training_date': pd.Timestamp.now().isoformat(),
            'threshold': getattr(self, 'threshold', None)
        }
        
        metadata_path = os.path.join(coating_dir, f"metadata_{self.coating_name}_{version}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        saved_files = {
            'model': model_path,
            'scaler': scaler_path,
            'metadata': metadata_path
        }
        
        logger.info(f"模型保存完成: {coating_dir}")
        return saved_files
    
    def load_model(self, model_dir: str, version: str = "v1.0") -> None:
        """
        加载模型和相关组件
        
        Args:
            model_dir: 模型保存目录
            version: 版本号
        """
        # 加载模型
        model_path = os.path.join(model_dir, self.coating_name, version, f"model_{self.coating_name}_{version}.h5")
        self.model = keras.models.load_model(
            model_path, 
            custom_objects={'weighted_mse_loss': self.weighted_mse_loss}
        )
        
        # 加载预处理器
        scaler_path = os.path.join(model_dir, self.coating_name, version, f"scaler_{self.coating_name}_{version}.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # 加载元数据
        metadata_path = os.path.join(model_dir, self.coating_name, version, f"metadata_{self.coating_name}_{version}.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 设置阈值
        self.threshold = metadata.get('threshold')
        
        self.is_trained = True
        
        logger.info(f"模型加载完成: {model_path}")
    
    def get_model_summary(self) -> str:
        """
        获取模型摘要
        
        Returns:
            str: 模型摘要字符串
        """
        if self.model is None:
            return "模型尚未构建"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

# 测试代码
if __name__ == "__main__":
    # 创建加权自编码器
    autoencoder = WeightedAutoencoder(input_dim=81, coating_name="DVP")
    
    # 加载DVP数据
    data = np.load("/workspace/code/spectrum_anomaly_detection/data/dvp_processed_data.npz")
    wavelengths = data['wavelengths']
    dvp_standard = data['dvp_values']
    
    print("=" * 60)
    print("Weighted Autoencoder 测试")
    print("=" * 60)
    
    # 生成训练数据（正常样本）
    print("\n生成训练数据...")
    np.random.seed(42)
    n_samples = 100
    noise_levels = [0.05, 0.1, 0.15]  # 低噪声水平表示正常样本
    
    normal_spectra = []
    for i in range(n_samples):
        noise_level = np.random.choice(noise_levels)
        noise = np.random.normal(0, noise_level, len(dvp_standard))
        spectrum = dvp_standard + noise
        normal_spectra.append(spectrum)
    
    normal_spectra = np.array(normal_spectra)
    print(f"✓ 生成 {len(normal_spectra)} 个正常样本")
    
    # 训练模型
    print("\n训练模型...")
    training_result = autoencoder.train(normal_spectra, wavelengths, validation_split=0.2)
    
    print(f"✓ 训练完成: {training_result['epochs_trained']} 个epoch")
    print(f"✓ 最终训练损失: {training_result['final_train_loss']:.6f}")
    if training_result['final_val_loss']:
        print(f"✓ 最终验证损失: {training_result['final_val_loss']:.6f}")
    print(f"✓ 阈值: {training_result['threshold']:.6f}")
    
    # 测试预测
    print("\n测试预测...")
    test_spectrum = dvp_standard + np.random.normal(0, 0.1, len(dvp_standard))
    test_spectra = test_spectrum.reshape(1, -1)
    
    prediction_result = autoencoder.predict(test_spectra, wavelengths)
    reconstruction_error = prediction_result['reconstruction_errors'][0]
    
    print(f"✓ 重构误差: {reconstruction_error:.6f}")
    print(f"✓ 阈值: {prediction_result['threshold']:.6f}")
    print(f"✓ 异常检测: {'是' if reconstruction_error > prediction_result['threshold'] else '否'}")
    
    # 保存模型
    print("\n保存模型...")
    saved_files = autoencoder.save_model("/workspace/code/spectrum_anomaly_detection/models", "v1.0")
    print(f"✓ 模型保存完成:")
    for key, path in saved_files.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    print("\n🎉 Weighted Autoencoder 测试完成！")