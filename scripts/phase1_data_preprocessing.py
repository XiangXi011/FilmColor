#!/usr/bin/env python3
"""
Phase 1: 数据预处理和标准曲线分析 - 主执行脚本
专门用于DVP涂层类型的光谱数据处理和分析
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_loader import SpectrumDataLoader
from utils.data_validator import SpectrumValidator

def main():
    """主执行函数"""
    print("=" * 60)
    print("Phase 1: 数据预处理和标准曲线分析")
    print("目标涂层: DVP")
    print("=" * 60)
    
    # 设置工作目录
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 步骤1: 加载DVP标准曲线数据
        print("\n📊 步骤1: 加载DVP标准曲线数据")
        print("-" * 40)
        
        loader = SpectrumDataLoader(str(data_dir))
        wavelengths, dvp_values = loader.load_dvp_standard_curve()
        
        print(f"✓ 成功加载 {len(wavelengths)} 个波长点")
        print(f"✓ 波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
        print(f"✓ DVP反射率范围: {dvp_values.min():.4f} - {dvp_values.max():.4f}")
        
        # 步骤2: 数据验证
        print("\n🔍 步骤2: 数据质量验证")
        print("-" * 40)
        
        validation_result = loader.validate_spectrum_data(wavelengths, dvp_values, "DVP")
        print(f"✓ 数据验证结果: {'通过' if validation_result['is_valid'] else '失败'}")
        
        if validation_result['warnings']:
            print("⚠️  验证警告:")
            for warning in validation_result['warnings']:
                print(f"   - {warning}")
        
        # 步骤3: 生成标准波长网格
        print("\n📏 步骤3: 生成标准波长网格")
        print("-" * 40)
        
        standard_wavelengths = loader.get_wavelength_range(380, 780, 5)
        print(f"✓ 标准波长网格: {len(standard_wavelengths)} 个点")
        print(f"✓ 波长间隔: 5 nm")
        print(f"✓ 覆盖范围: 380-780 nm")
        
        # 检查是否需要插值
        if not np.array_equal(wavelengths, standard_wavelengths):
            print("⚠️  需要进行插值处理...")
            interpolated_values = loader.interpolate_to_standard_grid(
                wavelengths, dvp_values, standard_wavelengths
            )
            print(f"✓ 插值完成: {len(interpolated_values)} 个数据点")
            # 更新数据为插值后的结果
            wavelengths = standard_wavelengths
            dvp_values = interpolated_values
        else:
            print("✓ 数据已经是标准格式，无需插值")
        
        # 步骤4: 光谱特征分析
        print("\n📈 步骤4: 光谱特征分析")
        print("-" * 40)
        
        validator = SpectrumValidator(str(output_dir))
        analysis = validator.analyze_spectral_characteristics(wavelengths, dvp_values, "DVP")
        
        print(f"✓ 基本统计:")
        print(f"   - 均值: {analysis['basic_stats']['mean']:.4f}")
        print(f"   - 标准差: {analysis['basic_stats']['std']:.4f}")
        print(f"   - 范围: {analysis['basic_stats']['range']:.4f}")
        print(f"   - 中位数: {analysis['basic_stats']['median']:.4f}")
        
        print(f"✓ 峰值分析:")
        print(f"   - 峰值数量: {analysis['peak_analysis']['num_peaks']}")
        if analysis['peak_analysis']['dominant_peak_wavelength']:
            print(f"   - 主要峰值: {analysis['peak_analysis']['dominant_peak_wavelength']:.1f} nm "
                  f"(值: {analysis['peak_analysis']['dominant_peak_value']:.4f})")
        
        # 步骤5: 生成可视化图表
        print("\n🎨 步骤5: 生成可视化图表")
        print("-" * 40)
        
        plots = {}
        plots['standard_curve'] = validator.plot_standard_curve(wavelengths, dvp_values, "DVP")
        plots['spectral_regions'] = validator.plot_spectral_regions(wavelengths, dvp_values, "DVP")
        plots['data_quality'] = validator.plot_data_quality_report(wavelengths, dvp_values, "DVP")
        
        print(f"✓ 生成图表数量: {len(plots)}")
        for plot_name, plot_path in plots.items():
            print(f"   - {plot_name}: {os.path.basename(plot_path)}")
        
        # 步骤6: 生成完整验证报告
        print("\n📋 步骤6: 生成完整验证报告")
        print("-" * 40)
        
        validation_report = validator.generate_validation_report(wavelengths, dvp_values, "DVP")
        
        # 保存验证报告
        report_path = output_dir / "dvp_validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 验证报告已保存: {report_path}")
        
        # 步骤7: 保存处理后的数据
        print("\n💾 步骤7: 保存处理后的数据")
        print("-" * 40)
        
        training_data = loader.load_training_data("DVP")
        training_data['wavelengths'] = wavelengths
        training_data['standard_curve'] = dvp_values
        training_data['analysis'] = analysis
        
        processed_data_path = loader.save_processed_data(training_data, "dvp_processed_data.npz")
        print(f"✓ 处理后数据已保存: {processed_data_path}")
        
        # 步骤8: 生成Phase 1总结报告
        print("\n📊 Phase 1 总结报告")
        print("=" * 60)
        
        summary = {
            'phase': 'Phase 1: 数据预处理和标准曲线分析',
            'target_coating': 'DVP',
            'status': 'COMPLETED',
            'data_summary': {
                'wavelength_points': len(wavelengths),
                'wavelength_range': f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm",
                'reflectance_range': f"{dvp_values.min():.4f}-{dvp_values.max():.4f}",
                'data_quality': validation_report['validation_summary']['data_quality'],
                'ready_for_modeling': validation_report['validation_summary']['ready_for_modeling']
            },
            'generated_outputs': {
                'validation_report': str(report_path),
                'processed_data': processed_data_path,
                'visualizations': list(plots.values())
            },
            'next_steps': [
                'Phase 2: 核心算法实现 - SimilarityEvaluator',
                'Phase 3: 加权自编码器模型开发',
                'Phase 4: 模型训练脚本开发'
            ]
        }
        
        # 保存总结报告
        summary_path = output_dir / "phase1_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 目标涂层: {summary['target_coating']}")
        print(f"✓ 数据质量: {summary['data_summary']['data_quality']}")
        print(f"✓ 建模就绪: {'是' if summary['data_summary']['ready_for_modeling'] else '否'}")
        print(f"✓ 输出文件: {len(summary['generated_outputs']['visualizations'])} 个图表")
        print(f"✓ 总结报告: {summary_path}")
        
        print(f"\n🎉 Phase 1 完成！数据已准备就绪，可以开始 Phase 2。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)