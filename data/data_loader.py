"""
光谱数据加载和预处理模块
专门用于DVP涂层类型的标准曲线数据处理
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectrumDataLoader:
    """光谱数据加载器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，默认为当前目录
        """
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.wavelengths = None
        self.dvp_standard = None
        self.coating_data = {}
        
    def load_dvp_standard_curve(self, filename: str = "HunterLab DVP.csv") -> Tuple[np.ndarray, np.ndarray]:
        """
        加载DVP标准曲线数据
        
        Args:
            filename: CSV文件名
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (波长数组, DVP反射率数组)
        """
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 获取波长列（第一行是波长）
            wavelengths = df.columns[1:].astype(float)  # 跳过第一列（空列）
            
            # 获取DVP数据（第二行）
            dvp_values = df.iloc[0, 1:].astype(float)  # 跳过第一列
            
            # 存储数据
            self.wavelengths = wavelengths.values
            self.dvp_standard = dvp_values.values
            
            logger.info(f"成功加载DVP标准曲线: {len(self.wavelengths)}个波长点")
            logger.info(f"波长范围: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
            logger.info(f"DVP反射率范围: {self.dvp_standard.min():.4f} - {self.dvp_standard.max():.4f}")
            
            return self.wavelengths, self.dvp_standard
            
        except Exception as e:
            logger.error(f"加载DVP标准曲线失败: {e}")
            raise
    
    def get_wavelength_range(self, start: int = 380, end: int = 780, step: int = 5) -> np.ndarray:
        """
        生成标准波长范围数组
        
        Args:
            start: 起始波长 (nm)
            end: 结束波长 (nm)  
            step: 波长间隔 (nm)
            
        Returns:
            np.ndarray: 波长数组
        """
        wavelengths = np.arange(start, end + step, step)
        logger.info(f"生成波长范围: {start}-{end}nm, 步长{step}nm, 共{len(wavelengths)}个点")
        return wavelengths
    
    def interpolate_to_standard_grid(self, wavelengths: np.ndarray, values: np.ndarray, 
                                   target_wavelengths: np.ndarray = None) -> np.ndarray:
        """
        将光谱数据插值到标准波长网格
        
        Args:
            wavelengths: 原始波长数组
            values: 原始光谱值数组
            target_wavelengths: 目标波长网格，如果为None则使用标准范围
            
        Returns:
            np.ndarray: 插值后的光谱值数组
        """
        if target_wavelengths is None:
            target_wavelengths = self.get_wavelength_range()
        
        # 检查数据范围
        if wavelengths.min() > target_wavelengths.min() or wavelengths.max() < target_wavelengths.max():
            logger.warning("原始数据波长范围不足以覆盖目标范围，将进行外推")
        
        # 线性插值
        interpolated_values = np.interp(target_wavelengths, wavelengths, values)
        
        logger.info(f"插值完成: {len(wavelengths)} -> {len(interpolated_values)}个数据点")
        return interpolated_values
    
    def validate_spectrum_data(self, wavelengths: np.ndarray, values: np.ndarray, 
                             coating_name: str = "DVP") -> Dict[str, Any]:
        """
        验证光谱数据质量
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_result = {
            'coating_name': coating_name,
            'data_points': len(values),
            'wavelength_range': (float(wavelengths.min()), float(wavelengths.max())),
            'value_range': (float(values.min()), float(values.max())),
            'has_nan': np.isnan(values).any(),
            'has_negative': (values < 0).any(),
            'is_valid': True,
            'warnings': []
        }
        
        # 检查数据有效性
        if validation_result['has_nan']:
            validation_result['warnings'].append("数据包含NaN值")
            validation_result['is_valid'] = False
            
        if validation_result['has_negative']:
            validation_result['warnings'].append("数据包含负值")
            
        if len(values) < 10:
            validation_result['warnings'].append("数据点过少")
            validation_result['is_valid'] = False
            
        if validation_result['value_range'][1] > 100:
            validation_result['warnings'].append("数据值过大，可能需要标准化")
            
        logger.info(f"数据验证完成 [{coating_name}]: {'通过' if validation_result['is_valid'] else '失败'}")
        for warning in validation_result['warnings']:
            logger.warning(f"  - {warning}")
            
        return validation_result
    
    def load_training_data(self, coating_name: str = "DVP", data_format: str = "csv") -> Dict[str, Any]:
        """
        加载训练数据（为未来扩展准备）
        
        Args:
            coating_name: 涂层名称
            data_format: 数据格式
            
        Returns:
            Dict[str, Any]: 训练数据字典
        """
        # 目前只返回标准曲线数据
        # 实际使用时需要加载历史正常生产数据
        if self.wavelengths is None or self.dvp_standard is None:
            self.load_dvp_standard_curve()
            
        training_data = {
            'coating_name': coating_name,
            'wavelengths': self.wavelengths,
            'standard_curve': self.dvp_standard,
            'normal_samples': [],  # 待补充正常样本数据
            'anomalous_samples': [],  # 待补充异常样本数据
            'metadata': {
                'data_source': 'HunterLab DVP.csv',
                'wavelength_range': f"{self.wavelengths.min():.0f}-{self.wavelengths.max():.0f}nm",
                'data_points': len(self.wavelengths),
                'loaded_at': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"训练数据加载完成 [{coating_name}]: {len(self.wavelengths)}个波长点")
        return training_data
    
    def save_processed_data(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        保存处理后的数据
        
        Args:
            data: 要保存的数据字典
            filename: 保存文件名
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"processed_{data['coating_name']}_data.npz"
            
        file_path = os.path.join(self.data_dir, filename)
        
        # 保存为npz格式
        np.savez_compressed(file_path, **data)
        
        logger.info(f"数据已保存到: {file_path}")
        return file_path

if __name__ == "__main__":
    # 测试代码
    loader = SpectrumDataLoader()
    
    # 加载DVP标准曲线
    wavelengths, dvp_values = loader.load_dvp_standard_curve()
    
    # 验证数据
    validation = loader.validate_spectrum_data(wavelengths, dvp_values, "DVP")
    print(f"验证结果: {validation}")
    
    # 生成标准波长网格
    standard_wavelengths = loader.get_wavelength_range()
    print(f"标准波长网格: {len(standard_wavelengths)}个点")
    
    # 如果需要插值（当前数据已经是标准格式）
    if not np.array_equal(wavelengths, standard_wavelengths):
        interpolated_values = loader.interpolate_to_standard_grid(wavelengths, dvp_values, standard_wavelengths)
        print(f"插值后数据: {len(interpolated_values)}个点")
    
    # 加载训练数据
    training_data = loader.load_training_data()
    print(f"训练数据加载完成: {training_data['metadata']}")