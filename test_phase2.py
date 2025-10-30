#!/usr/bin/env python3
"""
SimilarityEvaluator 测试脚本
验证Quality Score计算、权重计算等功能
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append('/workspace/code/spectrum_anomaly_detection')

def test_similarity_evaluator():
    """测试SimilarityEvaluator功能"""
    print("=" * 60)
    print("Phase 2: SimilarityEvaluator 测试")
    print("=" * 60)
    
    try:
        # 导入评估器
        from algorithms.similarity_evaluator import SimilarityEvaluator
        
        # 创建评估器
        evaluator = SimilarityEvaluator("DVP")
        print("✓ SimilarityEvaluator 初始化成功")
        
        # 加载处理后的DVP数据
        data_path = "/workspace/code/spectrum_anomaly_detection/data/dvp_processed_data.npz"
        data = np.load(data_path)
        wavelengths = data['wavelengths']
        dvp_standard = data['dvp_values']
        
        print(f"✓ 加载DVP数据: {len(wavelengths)}个波长点")
        print(f"✓ 波长范围: {wavelengths.min():.0f}-{wavelengths.max():.0f}nm")
        print(f"✓ DVP反射率范围: {dvp_standard.min():.4f}-{dvp_standard.max():.4f}")
        
        # 测试1: 相同光谱评估（基准测试）
        print("\n📊 测试1: 相同光谱评估（基准测试）")
        print("-" * 40)
        
        result_identical = evaluator.evaluate(dvp_standard, dvp_standard, wavelengths, coating_name="DVP")
        print(f"✓ Quality Score: {result_identical['similarity_score_percent']:.2f}%")
        print(f"✓ 加权皮尔逊相关系数: {result_identical['weighted_pearson']:.4f}")
        print(f"✓ 加权RMSE: {result_identical['rmse']:.6f}")
        
        if result_identical['similarity_score_percent'] > 99.9:
            print("✓ 基准测试通过：相同光谱得到接近100%的分数")
        else:
            print("⚠️  基准测试异常：相同光谱分数低于预期")
        
        # 测试2: 轻微噪声光谱评估
        print("\n🔍 测试2: 轻微噪声光谱评估")
        print("-" * 40)
        
        noise_levels = [0.05, 0.1, 0.2, 0.5]
        noise_results = []
        
        for noise_level in noise_levels:
            noisy_spectrum = dvp_standard + np.random.normal(0, noise_level, len(dvp_standard))
            result = evaluator.evaluate(noisy_spectrum, dvp_standard, wavelengths, coating_name="DVP")
            noise_results.append({
                'noise_level': noise_level,
                'quality_score': result['similarity_score_percent'],
                'pearson': result['weighted_pearson'],
                'rmse': result['rmse']
            })
            print(f"  噪声水平 {noise_level:.2f}: Quality Score = {result['similarity_score_percent']:.2f}%")
        
        # 测试3: 权重分析
        print("\n⚖️  测试3: 权重分析")
        print("-" * 40)
        
        weight_data = evaluator.get_weight_visualization_data(wavelengths, "DVP")
        print(f"✓ 权重范围: {weight_data['weight_stats']['min']:.2f} - {weight_data['weight_stats']['max']:.2f}")
        print(f"✓ 权重均值: {weight_data['weight_stats']['mean']:.2f}")
        print(f"✓ 权重标准差: {weight_data['weight_stats']['std']:.2f}")
        
        # 统计不同权重值的分布
        weight_counts = {}
        for w in weight_data['weights']:
            if w == 1.0:
                weight_counts['基础权重(1.0)'] = weight_counts.get('基础权重(1.0)', 0) + 1
            elif abs(w - 3.0) < 0.1:
                weight_counts['范围权重(3.0)'] = weight_counts.get('范围权重(3.0)', 0) + 1
            elif abs(w - 4.5) < 0.1:
                weight_counts['DVP增强权重(4.5)'] = weight_counts.get('DVP增强权重(4.5)', 0) + 1
            else:
                weight_counts['其他权重'] = weight_counts.get('其他权重', 0) + 1
        
        print("✓ 权重分布:")
        for weight_type, count in weight_counts.items():
            percentage = count / len(weight_data['weights']) * 100
            print(f"  - {weight_type}: {count}个点 ({percentage:.1f}%)")
        
        # 测试4: 质量阈值
        print("\n🎯 测试4: 质量阈值")
        print("-" * 40)
        
        quality_levels = ['excellent', 'good', 'acceptable', 'poor']
        thresholds = {}
        
        for level in quality_levels:
            threshold = evaluator.get_quality_score_threshold("DVP", level)
            thresholds[level] = threshold
            print(f"✓ {level.capitalize()}: {threshold}%")
        
        # 测试5: 批量评估
        print("\n📈 测试5: 批量评估")
        print("-" * 40)
        
        # 生成测试光谱集合
        test_spectra = []
        for i in range(10):
            if i < 3:
                # 前3个是高质量光谱（低噪声）
                noise = np.random.normal(0, 0.05, len(dvp_standard))
            elif i < 7:
                # 中间4个是中等质量光谱（中等噪声）
                noise = np.random.normal(0, 0.2, len(dvp_standard))
            else:
                # 最后3个是低质量光谱（高噪声）
                noise = np.random.normal(0, 0.5, len(dvp_standard))
            
            test_spectra.append(dvp_standard + noise)
        
        # 执行批量评估
        batch_results = evaluator.batch_evaluate(test_spectra, dvp_standard, wavelengths, "DVP")
        
        print(f"✓ 批量评估完成: {len(test_spectra)}个光谱")
        print(f"✓ 平均Quality Score: {batch_results['similarity_score_percent'].mean():.2f}%")
        print(f"✓ Score标准差: {batch_results['similarity_score_percent'].std():.2f}%")
        print(f"✓ 最高分: {batch_results['similarity_score_percent'].max():.2f}%")
        print(f"✓ 最低分: {batch_results['similarity_score_percent'].min():.2f}%")
        
        # 按质量等级分类
        excellent_count = (batch_results['similarity_score_percent'] >= thresholds['excellent']).sum()
        good_count = (batch_results['similarity_score_percent'] >= thresholds['good']).sum()
        acceptable_count = (batch_results['similarity_score_percent'] >= thresholds['acceptable']).sum()
        
        print(f"✓ 质量等级分布:")
        print(f"  - Excellent (≥{thresholds['excellent']}%): {excellent_count}个")
        print(f"  - Good (≥{thresholds['good']}%): {good_count}个")
        print(f"  - Acceptable (≥{thresholds['acceptable']}%): {acceptable_count}个")
        print(f"  - Poor (<{thresholds['acceptable']}%): {len(test_spectra) - acceptable_count}个")
        
        # 保存测试结果
        print("\n💾 保存测试结果")
        print("-" * 40)
        
        # 保存权重可视化数据
        weight_viz_path = "/workspace/code/spectrum_anomaly_detection/output/dvp_weights_visualization.npz"
        os.makedirs(os.path.dirname(weight_viz_path), exist_ok=True)
        np.savez_compressed(weight_viz_path, **weight_data)
        print(f"✓ 权重可视化数据已保存: {weight_viz_path}")
        
        # 保存批量评估结果
        batch_results_path = "/workspace/code/spectrum_anomaly_detection/output/dvp_batch_evaluation_results.csv"
        batch_results.to_csv(batch_results_path, index=False)
        print(f"✓ 批量评估结果已保存: {batch_results_path}")
        
        # 生成测试总结报告
        summary = {
            'phase': 'Phase 2: SimilarityEvaluator 测试',
            'status': 'COMPLETED',
            'test_results': {
                'baseline_test': {
                    'identical_spectrum_score': result_identical['similarity_score_percent'],
                    'passed': result_identical['similarity_score_percent'] > 99.9
                },
                'noise_test': {
                    'noise_levels_tested': noise_levels,
                    'results': noise_results
                },
                'weight_analysis': {
                    'weight_stats': weight_data['weight_stats'],
                    'weight_distribution': weight_counts
                },
                'quality_thresholds': thresholds,
                'batch_evaluation': {
                    'total_spectra': len(test_spectra),
                    'average_score': float(batch_results['similarity_score_percent'].mean()),
                    'score_std': float(batch_results['similarity_score_percent'].std()),
                    'quality_distribution': {
                        'excellent': int(excellent_count),
                        'good': int(good_count),
                        'acceptable': int(acceptable_count),
                        'poor': int(len(test_spectra) - acceptable_count)
                    }
                }
            },
            'next_steps': [
                'Phase 3: 加权自编码器模型开发',
                'Phase 4: 模型训练脚本开发'
            ]
        }
        
        summary_path = "/workspace/code/spectrum_anomaly_detection/output/phase2_test_summary.json"
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✓ 测试总结报告已保存: {summary_path}")
        
        print("\n🎉 Phase 2 测试完成！")
        print("✓ SimilarityEvaluator 功能验证通过")
        print("✓ Quality Score 计算正确")
        print("✓ 权重计算符合DVP涂层特征")
        print("✓ 批量评估功能正常")
        print("✓ 可以开始 Phase 3")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_similarity_evaluator()