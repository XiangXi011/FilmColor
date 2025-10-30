"""
SimilarityEvaluator - 基于专家规则的光谱质量评估器
实现Quality Score计算、权重计算、加权统计指标等功能
专门为DVP涂层类型优化
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityEvaluator:
    """
    光谱相似性评估器
    
    基于专家规则的光谱质量评估系统，专门用于DVP涂层类型的光谱分析
    计算Quality Score，评估光谱与预定义"黄金标准"的符合程度
    """
    
    def __init__(self, coating_name: str = "DVP"):
        """
        初始化评估器
        
        Args:
            coating_name: 涂层名称，用于选择特定的权重配置
        """
        self.coating_name = coating_name
        self.wavelengths = np.arange(380, 781, 5)  # 380-780nm, 步长5nm
        
        logger.info(f"SimilarityEvaluator 初始化完成 [{coating_name}]")
        logger.info(f"波长范围: {self.wavelengths.min()}-{self.wavelengths.max()}nm, 共{len(self.wavelengths)}个点")
    
    def evaluate(self, y1: np.ndarray, y2: np.ndarray, 
                 wavelengths: Optional[np.ndarray] = None,
                 weight_range: Tuple[int, int] = (400, 680),
                 coating_name: Optional[str] = None) -> Dict[str, Any]:
        """
        评估两个光谱的相似性
        
        Args:
            y1: 实时传入的光谱数据 (反射率曲线)
            y2: 预存的"黄金标准"光谱 (反射率曲线)
            wavelengths: 光谱对应的波长数组，如果为None则使用默认波长
            weight_range: 权重计算的范围 (start, end)
            coating_name: 涂层名称，如果为None则使用初始化时的名称
            
        Returns:
            Dict[str, Any]: 包含所有评估指标的字典
        """
        # 参数验证
        if coating_name is None:
            coating_name = self.coating_name
            
        if wavelengths is None:
            wavelengths = self.wavelengths
        
        # 数据长度检查和截断
        min_len = min(len(y1), len(y2), len(wavelengths))
        y1, y2, wavelengths = y1[:min_len], y2[:min_len], wavelengths[:min_len]
        
        logger.debug(f"数据长度检查: y1={len(y1)}, y2={len(y2)}, wavelengths={len(wavelengths)}")
        
        # 计算权重向量
        weights = self._calculate_weights(wavelengths, weight_range, coating_name)
        
        # 计算加权指标
        weighted_pearson = self._weighted_pearson(y1, y2, weights)
        rmse = self._weighted_rmse(y1, y2, weights)
        
        # 综合得分计算 (根据规格文档)
        # similarity_score = 0.3 * (1 + weighted_pearson) / 2 + 0.7 * (1 / (1 + rmse))
        # 修正：确保在理想情况下(pearson=1, rmse=0)得到100%的分数
        similarity_score = 0.3 * (1 + weighted_pearson) / 2 + 0.7 * (1 / (1 + rmse))
        
        # 转换为百分比形式
        similarity_score_percent = similarity_score * 100
        
        result = {
            "weighted_pearson": float(weighted_pearson),
            "rmse": float(rmse),
            "similarity_score": float(similarity_score),
            "similarity_score_percent": float(similarity_score_percent),
            "weights": weights,
            "metadata": {
                "coating_name": coating_name,
                "weight_range": weight_range,
                "data_points": len(y1),
                "wavelength_range": f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm"
            }
        }
        
        logger.debug(f"评估完成: Quality Score = {similarity_score_percent:.2f}%")
        return result
    
    def _calculate_weights(self, wavelengths: np.ndarray, weight_range: Tuple[int, int], 
                          coating_name: str) -> np.ndarray:
        """
        计算权重向量
        
        根据涂层类型和波长范围，为不同波段分配不同的权重
        这是专家知识的直接体现
        
        Args:
            wavelengths: 波长数组
            weight_range: 权重计算范围
            coating_name: 涂层名称
            
        Returns:
            np.ndarray: 权重向量
        """
        weights = np.ones_like(wavelengths, dtype=np.float64)
        
        # 基础权重: 在指定范围内权重为3
        mask = (wavelengths >= weight_range[0]) & (wavelengths <= weight_range[1])
        weights[mask] = 3.0
        
        logger.debug(f"基础权重设置: 范围{weight_range[0]}-{weight_range[1]}nm, 权重=3")
        
        # 根据涂层类型调整权重
        if coating_name == "DVS":
            # DVS涂层暂无特殊调整
            logger.debug("DVS涂层: 使用默认权重配置")
            
        elif coating_name == "DVP":
            # DVP涂层: 增强400-550nm波段的权重
            peak_mask = (wavelengths >= 400) & (wavelengths <= 550)
            weights[peak_mask] *= 1.5
            logger.debug("DVP涂层: 400-550nm波段权重增强1.5倍")
            
        elif coating_name == "DVK":
            # DVK涂层暂无特殊调整
            logger.debug("DVK涂层: 使用默认权重配置")
            
        elif coating_name == "C2":
            # C2涂层: 多个重点波段增强
            # 445-465nm (蓝色峰值)
            peak_mask1 = (wavelengths >= 445) & (wavelengths <= 465)
            weights[peak_mask1] *= 20
            
            # 515-555nm (绿色峰值)
            peak_mask2 = (wavelengths >= 515) & (wavelengths <= 555)
            weights[peak_mask2] *= 30
            
            # 645-685nm (红色峰值)
            peak_mask3 = (wavelengths >= 645) & (wavelengths <= 685)
            weights[peak_mask3] *= 10
            
            logger.debug("C2涂层: 多波段权重增强 (蓝色20x, 绿色30x, 红色10x)")
            
        elif coating_name == "BPCN_CX":
            # BPCN_CX涂层: 400-500nm波段增强
            peak_mask = (wavelengths >= 400) & (wavelengths <= 500)
            weights[peak_mask] *= 1.5
            logger.debug("BPCN_CX涂层: 400-500nm波段权重增强1.5倍")
            
        elif coating_name == "BPCN_CC":
            # BPCN_CC涂层: 400-500nm波段增强
            peak_mask = (wavelengths >= 400) & (wavelengths <= 500)
            weights[peak_mask] *= 1.5
            logger.debug("BPCN_CC涂层: 400-500nm波段权重增强1.5倍")
            
        elif coating_name == "DVG":
            # DVG涂层: 400-500nm波段增强
            peak_mask = (wavelengths >= 400) & (wavelengths <= 500)
            weights[peak_mask] *= 1.5
            logger.debug("DVG涂层: 400-500nm波段权重增强1.5倍")
        
        # 权重统计信息
        logger.debug(f"权重统计: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        
        return weights
    
    def _weighted_pearson(self, y1: np.ndarray, y2: np.ndarray, weights: np.ndarray) -> float:
        """
        计算加权皮尔逊相关系数
        
        Args:
            y1: 第一个光谱数组
            y2: 第二个光谱数组
            weights: 权重数组
            
        Returns:
            float: 加权皮尔逊相关系数 (-1 到 1)
        """
        # 计算加权均值
        mean_y1 = np.average(y1, weights=weights)
        mean_y2 = np.average(y2, weights=weights)
        
        # 计算加权协方差和方差
        numerator = np.sum(weights * (y1 - mean_y1) * (y2 - mean_y2))
        denom_y1 = np.sqrt(np.sum(weights * (y1 - mean_y1) ** 2))
        denom_y2 = np.sqrt(np.sum(weights * (y2 - mean_y2) ** 2))
        
        # 避免除零错误
        denominator = denom_y1 * denom_y2
        if denominator == 0:
            logger.warning("加权方差为零，返回相关系数0")
            return 0.0
        
        correlation = numerator / denominator
        
        logger.debug(f"加权皮尔逊相关系数: {correlation:.4f}")
        return float(correlation)
    
    def _weighted_rmse(self, y1: np.ndarray, y2: np.ndarray, weights: np.ndarray) -> float:
        """
        计算加权均方根误差
        
        Args:
            y1: 第一个光谱数组
            y2: 第二个光谱数组
            weights: 权重数组
            
        Returns:
            float: 加权RMSE
        """
        # 计算加权差值的平方
        weighted_diff_sq = weights * (y1 - y2) ** 2
        
        # 计算加权RMSE
        rmse = np.sqrt(np.sum(weighted_diff_sq) / np.sum(weights))
        
        logger.debug(f"加权RMSE: {rmse:.4f}")
        return float(rmse)
    
    def get_quality_score_threshold(self, coating_name: str = None, 
                                   quality_level: str = "good") -> float:
        """
        获取质量分数阈值
        
        Args:
            coating_name: 涂层名称
            quality_level: 质量水平 ("excellent", "good", "acceptable", "poor")
            
        Returns:
            float: 质量分数阈值 (0-100)
        """
        if coating_name is None:
            coating_name = self.coating_name
            
        # 根据涂层类型和质量水平定义阈值
        thresholds = {
            "DVP": {
                "excellent": 95.0,
                "good": 90.0,
                "acceptable": 85.0,
                "poor": 80.0
            },
            "default": {
                "excellent": 95.0,
                "good": 90.0,
                "acceptable": 85.0,
                "poor": 80.0
            }
        }
        
        coating_thresholds = thresholds.get(coating_name, thresholds["default"])
        threshold = coating_thresholds.get(quality_level, coating_thresholds["good"])
        
        logger.debug(f"质量阈值 [{coating_name} - {quality_level}]: {threshold}%")
        return threshold
    
    def batch_evaluate(self, spectra_list: list, golden_standard: np.ndarray,
                      wavelengths: Optional[np.ndarray] = None,
                      coating_name: Optional[str] = None) -> pd.DataFrame:
        """
        批量评估多个光谱
        
        Args:
            spectra_list: 光谱数据列表
            golden_standard: 黄金标准光谱
            wavelengths: 波长数组
            coating_name: 涂层名称
            
        Returns:
            pd.DataFrame: 评估结果DataFrame
        """
        if wavelengths is None:
            wavelengths = self.wavelengths
            
        if coating_name is None:
            coating_name = self.coating_name
        
        results = []
        
        for i, spectrum in enumerate(spectra_list):
            try:
                result = self.evaluate(spectrum, golden_standard, wavelengths, coating_name=coating_name)
                result['spectrum_id'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"光谱 {i} 评估失败: {e}")
                results.append({
                    'spectrum_id': i,
                    'weighted_pearson': np.nan,
                    'rmse': np.nan,
                    'similarity_score': np.nan,
                    'similarity_score_percent': np.nan,
                    'error': str(e)
                })
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        logger.info(f"批量评估完成: {len(spectra_list)} 个光谱")
        return df_results
    
    def get_weight_visualization_data(self, wavelengths: Optional[np.ndarray] = None,
                                    coating_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取权重可视化数据
        
        Args:
            wavelengths: 波长数组
            coating_name: 涂层名称
            
        Returns:
            Dict[str, Any]: 可视化数据
        """
        if wavelengths is None:
            wavelengths = self.wavelengths
            
        if coating_name is None:
            coating_name = self.coating_name
        
        weights = self._calculate_weights(wavelengths, (400, 680), coating_name)
        
        return {
            'wavelengths': wavelengths,
            'weights': weights,
            'coating_name': coating_name,
            'weight_stats': {
                'min': float(weights.min()),
                'max': float(weights.max()),
                'mean': float(weights.mean()),
                'std': float(weights.std())
            }
        }

# 测试代码
if __name__ == "__main__":
    # 创建评估器
    evaluator = SimilarityEvaluator("DVP")
    
    # 加载处理后的DVP数据
    data = np.load("/workspace/code/spectrum_anomaly_detection/data/dvp_processed_data.npz")
    wavelengths = data['wavelengths']
    dvp_standard = data['dvp_values']
    
    print("=" * 60)
    print("SimilarityEvaluator 测试")
    print("=" * 60)
    
    # 测试1: 相同光谱的评估（应该得到接近100%的分数）
    print("\n测试1: 相同光谱评估")
    result1 = evaluator.evaluate(dvp_standard, dvp_standard, wavelengths, coating_name="DVP")
    print(f"Quality Score: {result1['similarity_score_percent']:.2f}%")
    print(f"加权皮尔逊相关系数: {result1['weighted_pearson']:.4f}")
    print(f"加权RMSE: {result1['rmse']:.6f}")
    
    # 测试2: 添加噪声的光谱评估
    print("\n测试2: 噪声光谱评估")
    noise_level = 0.1
    noisy_spectrum = dvp_standard + np.random.normal(0, noise_level, len(dvp_standard))
    result2 = evaluator.evaluate(noisy_spectrum, dvp_standard, wavelengths, coating_name="DVP")
    print(f"Quality Score: {result2['similarity_score_percent']:.2f}%")
    print(f"加权皮尔逊相关系数: {result2['weighted_pearson']:.4f}")
    print(f"加权RMSE: {result2['rmse']:.6f}")
    
    # 测试3: 权重可视化数据
    print("\n测试3: 权重分析")
    weight_data = evaluator.get_weight_visualization_data(wavelengths, "DVP")
    print(f"权重范围: {weight_data['weight_stats']['min']:.2f} - {weight_data['weight_stats']['max']:.2f}")
    print(f"权重均值: {weight_data['weight_stats']['mean']:.2f}")
    
    # 测试4: 质量阈值
    print("\n测试4: 质量阈值")
    thresholds = {
        'excellent': evaluator.get_quality_score_threshold("DVP", "excellent"),
        'good': evaluator.get_quality_score_threshold("DVP", "good"),
        'acceptable': evaluator.get_quality_score_threshold("DVP", "acceptable"),
        'poor': evaluator.get_quality_score_threshold("DVP", "poor")
    }
    print("DVP涂层质量阈值:")
    for level, threshold in thresholds.items():
        print(f"  {level}: {threshold}%")
    
    print("\n🎉 SimilarityEvaluator 测试完成！")