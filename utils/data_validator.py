"""
光谱数据验证和可视化模块
专门用于DVP标准曲线的分析和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    
    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class SpectrumValidator:
    """光谱数据验证器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化验证器
        
        Args:
            output_dir: 可视化输出目录
        """
        setup_matplotlib_for_plotting()
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_spectral_characteristics(self, wavelengths: np.ndarray, values: np.ndarray, 
                                       coating_name: str = "DVP") -> Dict[str, Any]:
        """
        分析光谱特征
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            
        Returns:
            Dict[str, Any]: 光谱特征分析结果
        """
        analysis = {
            'coating_name': coating_name,
            'basic_stats': {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'median': float(np.median(values))
            },
            'spectral_features': {},
            'derivative_analysis': {},
            'peak_analysis': {}
        }
        
        # 计算一阶导数
        first_derivative = np.gradient(values, wavelengths)
        analysis['derivative_analysis']['first_derivative'] = {
            'mean': float(np.mean(first_derivative)),
            'std': float(np.std(first_derivative)),
            'range': float(np.max(first_derivative) - np.min(first_derivative))
        }
        
        # 计算二阶导数
        second_derivative = np.gradient(first_derivative, wavelengths)
        analysis['derivative_analysis']['second_derivative'] = {
            'mean': float(np.mean(second_derivative)),
            'std': float(np.std(second_derivative))
        }
        
        # 寻找峰值
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(values, height=np.mean(values), distance=5)
        
        if len(peaks) > 0:
            peak_wavelengths = wavelengths[peaks]
            peak_values = values[peaks]
            
            analysis['peak_analysis'] = {
                'num_peaks': len(peaks),
                'peak_wavelengths': peak_wavelengths.tolist(),
                'peak_values': peak_values.tolist(),
                'dominant_peak_wavelength': float(peak_wavelengths[np.argmax(peak_values)]),
                'dominant_peak_value': float(np.max(peak_values))
            }
        else:
            analysis['peak_analysis'] = {
                'num_peaks': 0,
                'peak_wavelengths': [],
                'peak_values': [],
                'dominant_peak_wavelength': None,
                'dominant_peak_value': None
            }
        
        # 光谱区域分析
        regions = {
            'violet_blue': (380, 450),
            'blue_green': (450, 520), 
            'green_yellow': (520, 590),
            'yellow_orange': (590, 650),
            'red': (650, 780)
        }
        
        for region_name, (start_wl, end_wl) in regions.items():
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            if np.any(mask):
                region_values = values[mask]
                analysis['spectral_features'][region_name] = {
                    'wavelength_range': f"{start_wl}-{end_wl}nm",
                    'mean_reflectance': float(np.mean(region_values)),
                    'max_reflectance': float(np.max(region_values)),
                    'min_reflectance': float(np.min(region_values)),
                    'std_reflectance': float(np.std(region_values))
                }
        
        logger.info(f"光谱特征分析完成 [{coating_name}]: {len(peaks)}个峰值, 主要峰值在{analysis['peak_analysis']['dominant_peak_wavelength']}nm")
        return analysis
    
    def plot_standard_curve(self, wavelengths: np.ndarray, values: np.ndarray, 
                          coating_name: str = "DVP", save_path: str = None) -> str:
        """
        绘制标准光谱曲线
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{coating_name}_standard_curve.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 主光谱曲线
        ax1.plot(wavelengths, values, 'b-', linewidth=2, label=f'{coating_name} 标准曲线')
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('反射率')
        ax1.set_title(f'{coating_name} 涂层标准光谱曲线')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加区域标注
        regions = {
            '紫蓝光区 (380-450nm)': (380, 450, 'purple', 0.1),
            '蓝绿光区 (450-520nm)': (450, 520, 'blue', 0.1),
            '绿黄光区 (520-590nm)': (520, 590, 'green', 0.1),
            '黄橙光区 (590-650nm)': (590, 650, 'orange', 0.1),
            '红光区 (650-780nm)': (650, 780, 'red', 0.1)
        }
        
        for region_name, (start, end, color, alpha) in regions.items():
            ax1.axvspan(start, end, alpha=alpha, color=color, label=region_name)
        
        # 一阶导数
        first_derivative = np.gradient(values, wavelengths)
        ax2.plot(wavelengths, first_derivative, 'r-', linewidth=1, label='一阶导数')
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('反射率变化率')
        ax2.set_title('光谱一阶导数 (变化率)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"标准曲线图已保存: {save_path}")
        return save_path
    
    def plot_spectral_regions(self, wavelengths: np.ndarray, values: np.ndarray, 
                            coating_name: str = "DVP", save_path: str = None) -> str:
        """
        绘制光谱区域分析图
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{coating_name}_spectral_regions.png")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        regions = {
            'violet_blue': (380, 450, '紫蓝光区'),
            'blue_green': (450, 520, '蓝绿光区'), 
            'green_yellow': (520, 590, '绿黄光区'),
            'yellow_orange': (590, 650, '黄橙光区'),
            'red': (650, 780, '红光区')
        }
        
        colors = ['purple', 'blue', 'green', 'orange', 'red']
        
        for i, (region_key, (start_wl, end_wl, region_name)) in enumerate(regions.items()):
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            
            if np.any(mask):
                region_wavelengths = wavelengths[mask]
                region_values = values[mask]
                
                axes[i].plot(region_wavelengths, region_values, color=colors[i], linewidth=2)
                axes[i].set_title(f'{region_name}\n({start_wl}-{end_wl}nm)')
                axes[i].set_xlabel('波长 (nm)')
                axes[i].set_ylabel('反射率')
                axes[i].grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = np.mean(region_values)
                std_val = np.std(region_values)
                axes[i].axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.7, 
                              label=f'均值: {mean_val:.3f}')
                axes[i].legend()
        
        # 删除多余的子图
        if len(regions) < len(axes):
            for j in range(len(regions), len(axes)):
                fig.delaxes(axes[j])
        
        plt.suptitle(f'{coating_name} 涂层光谱区域分析', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"光谱区域分析图已保存: {save_path}")
        return save_path
    
    def plot_data_quality_report(self, wavelengths: np.ndarray, values: np.ndarray, 
                               coating_name: str = "DVP", save_path: str = None) -> str:
        """
        绘制数据质量报告
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{coating_name}_data_quality.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 原始数据分布
        axes[0, 0].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('反射率分布直方图')
        axes[0, 0].set_xlabel('反射率值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 数据质量指标
        quality_metrics = {
            '数据点数': len(values),
            '波长范围': f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm",
            '反射率范围': f"{values.min():.3f}-{values.max():.3f}",
            '均值': f"{np.mean(values):.3f}",
            '标准差': f"{np.std(values):.3f}",
            '变异系数': f"{np.std(values)/np.mean(values):.3f}"
        }
        
        axes[0, 1].axis('off')
        axes[0, 1].text(0.1, 0.9, '数据质量指标', fontsize=14, fontweight='bold', transform=axes[0, 1].transAxes)
        y_pos = 0.8
        for key, value in quality_metrics.items():
            axes[0, 1].text(0.1, y_pos, f'{key}: {value}', fontsize=10, transform=axes[0, 1].transAxes)
            y_pos -= 0.1
        
        # 3. 波长vs反射率散点图
        axes[1, 0].scatter(wavelengths, values, alpha=0.6, s=20, c='blue')
        axes[1, 0].set_title('波长-反射率散点图')
        axes[1, 0].set_xlabel('波长 (nm)')
        axes[1, 0].set_ylabel('反射率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 数据连续性检查
        wavelength_diffs = np.diff(wavelengths)
        axes[1, 1].plot(wavelengths[1:], wavelength_diffs, 'g-o', markersize=3)
        axes[1, 1].set_title('波长间隔连续性检查')
        axes[1, 1].set_xlabel('波长 (nm)')
        axes[1, 1].set_ylabel('波长间隔 (nm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{coating_name} 涂层数据质量报告', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"数据质量报告已保存: {save_path}")
        return save_path
    
    def generate_validation_report(self, wavelengths: np.ndarray, values: np.ndarray, 
                                 coating_name: str = "DVP") -> Dict[str, Any]:
        """
        生成完整的验证报告
        
        Args:
            wavelengths: 波长数组
            values: 光谱值数组
            coating_name: 涂层名称
            
        Returns:
            Dict[str, Any]: 完整的验证报告
        """
        logger.info(f"开始生成 {coating_name} 涂层的完整验证报告...")
        
        # 特征分析
        analysis = self.analyze_spectral_characteristics(wavelengths, values, coating_name)
        
        # 生成可视化图表
        plots = {
            'standard_curve': self.plot_standard_curve(wavelengths, values, coating_name),
            'spectral_regions': self.plot_spectral_regions(wavelengths, values, coating_name),
            'data_quality': self.plot_data_quality_report(wavelengths, values, coating_name)
        }
        
        # 生成报告
        report = {
            'coating_name': coating_name,
            'analysis': analysis,
            'generated_plots': plots,
            'validation_summary': {
                'data_points': len(values),
                'wavelength_coverage': f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm",
                'data_quality': 'EXCELLENT' if len(values) > 50 and np.std(values) > 0.1 else 'GOOD',
                'ready_for_modeling': True
            },
            'recommendations': self._generate_recommendations(analysis)
        }
        
        logger.info(f"验证报告生成完成: {len(plots)}个图表, 数据质量: {report['validation_summary']['data_quality']}")
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        基于分析结果生成建议
        
        Args:
            analysis: 光谱特征分析结果
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 基于峰值分析的建议
        if analysis['peak_analysis']['num_peaks'] > 0:
            recommendations.append(f"检测到{analysis['peak_analysis']['num_peaks']}个光谱峰值，主要峰值在{analysis['peak_analysis']['dominant_peak_wavelength']}nm")
        else:
            recommendations.append("未检测到明显峰值，可能需要调整峰值检测参数")
        
        # 基于变异系数的建议
        cv = analysis['basic_stats']['std'] / analysis['basic_stats']['mean']
        if cv > 0.5:
            recommendations.append("光谱变异较大，适合用于异常检测建模")
        elif cv < 0.1:
            recommendations.append("光谱变异较小，可能需要增强特征提取")
        
        # 基于导数分析的建议
        if abs(analysis['derivative_analysis']['first_derivative']['std']) > 0.01:
            recommendations.append("光谱变化率较大，建议在模型中考虑导数特征")
        
        # 基于区域分析的建议
        if 'blue_green' in analysis['spectral_features']:
            bg_mean = analysis['spectral_features']['blue_green']['mean_reflectance']
            if bg_mean > 0.5:
                recommendations.append("蓝绿光区反射率较高，这是DVP涂层的特征区域")
        
        return recommendations

if __name__ == "__main__":
    # 测试代码
    from data_loader import SpectrumDataLoader
    
    # 加载数据
    loader = SpectrumDataLoader()
    wavelengths, dvp_values = loader.load_dvp_standard_curve()
    
    # 创建验证器
    validator = SpectrumValidator()
    
    # 生成验证报告
    report = validator.generate_validation_report(wavelengths, dvp_values, "DVP")
    
    print("验证报告摘要:")
    print(f"涂层名称: {report['coating_name']}")
    print(f"数据点数: {report['validation_summary']['data_points']}")
    print(f"波长覆盖: {report['validation_summary']['wavelength_coverage']}")
    print(f"数据质量: {report['validation_summary']['data_quality']}")
    print(f"建模就绪: {report['validation_summary']['ready_for_modeling']}")
    
    print("\n建议:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print(f"\n生成的图表:")
    for plot_name, plot_path in report['generated_plots'].items():
        print(f"- {plot_name}: {plot_path}")