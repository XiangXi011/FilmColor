#!/usr/bin/env python3
"""
分析不同组合策略的效果
探索如何优化Combined F1性能

Author: MiniMax Agent
Date: 2025-11-01
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """计算性能指标"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def simulate_combination_strategies(
    quality_pred: np.ndarray,
    stability_pred: np.ndarray, 
    combined_true: np.ndarray
) -> Dict:
    """
    模拟不同组合策略的性能
    
    Args:
        quality_pred: Quality Score预测结果（0=正常，1=异常）
        stability_pred: Stability Score预测结果（0=正常，1=异常）
        combined_true: 组合真实标签（0=正常，1=异常）
    
    Returns:
        各策略的性能指标
    """
    results = {}
    
    # 策略1: OR (当前的two_stage等价于OR)
    or_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
    results['OR'] = calculate_metrics(combined_true, or_pred)
    
    # 策略2: AND (两个都判异常才判异常)
    and_pred = ((quality_pred == 1) & (stability_pred == 1)).astype(int)
    results['AND'] = calculate_metrics(combined_true, and_pred)
    
    # 策略3: Quality-First (只有Quality判异常才判异常)
    quality_first_pred = quality_pred
    results['Quality-First'] = calculate_metrics(combined_true, quality_first_pred)
    
    # 策略4: Stability-First (只有Stability判异常才判异常)
    stability_first_pred = stability_pred
    results['Stability-First'] = calculate_metrics(combined_true, stability_first_pred)
    
    # 策略5: Majority Vote (至少需要一个强判断)
    # 这里我们假设Quality更可靠（AUC=0.903 > 0.735）
    # 如果Quality判异常，直接判异常；如果Quality判正常且Stability判异常，需要更严格的条件
    # 暂时使用OR的变体
    
    return results

def analyze_threshold_sensitivity(
    quality_scores: np.ndarray,
    stability_scores: np.ndarray,
    quality_labels: np.ndarray,
    stability_labels: np.ndarray,
    current_quality_threshold: float,
    current_stability_threshold: float
):
    """
    分析阈值调整对Combined F1的影响
    
    策略：在OR组合下，提高阈值可以降低误报率（提升精确率），但可能降低召回率
    """
    print("\n" + "="*80)
    print("阈值敏感性分析")
    print("="*80)
    
    combined_true = ((quality_labels == 1) | (stability_labels == 1)).astype(int)
    
    print(f"\n当前阈值: Quality={current_quality_threshold:.4f}, Stability={current_stability_threshold:.4f}")
    
    # 尝试不同的Quality阈值
    print("\n" + "-"*80)
    print("调整Quality阈值（保持Stability阈值不变）")
    print("-"*80)
    print(f"{'Quality阈值':>15} {'Precision':>12} {'Recall':>12} {'F1':>12} {'FP':>8} {'FN':>8}")
    
    for q_factor in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        q_threshold = np.percentile(quality_scores[quality_labels == 0], q_factor * 100)
        quality_pred = (quality_scores < q_threshold).astype(int)
        stability_pred = (stability_scores > current_stability_threshold).astype(int)
        or_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
        metrics = calculate_metrics(combined_true, or_pred)
        print(f"{q_threshold:>15.4f} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f} {metrics['fp']:>8} {metrics['fn']:>8}")
    
    # 尝试不同的Stability阈值
    print("\n" + "-"*80)
    print("调整Stability阈值（保持Quality阈值不变）")
    print("-"*80)
    print(f"{'Stability阈值':>15} {'Precision':>12} {'Recall':>12} {'F1':>12} {'FP':>8} {'FN':>8}")
    
    for s_factor in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        s_threshold = np.percentile(stability_scores[stability_labels == 0], s_factor * 100)
        quality_pred = (quality_scores < current_quality_threshold).astype(int)
        stability_pred = (stability_scores > s_threshold).astype(int)
        or_pred = ((quality_pred == 1) | (stability_pred == 1)).astype(int)
        metrics = calculate_metrics(combined_true, or_pred)
        print(f"{s_threshold:>15.4f} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f} {metrics['fp']:>8} {metrics['fn']:>8}")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("DVP光谱异常检测 - 组合策略分析")
    print("="*80)
    
    # 从评估结果加载性能指标
    metrics_path = project_root / "evaluation" / "performance_metrics.json"
    if not metrics_path.exists():
        print("❌ 未找到性能指标文件，请先运行评估脚本")
        print("   运行命令: python scripts/evaluate.py --model-dir models/DVP/v1.12_varsm_21k --test-data-npz data/export_v11/test_subset.npz")
        return
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print("\n[当前性能] two_stage策略（等价于OR）")
    print("-"*80)
    print(f"Quality Score:")
    print(f"  - AUC: {metrics['quality_score']['auc_roc']:.3f}")
    print(f"  - Precision: {metrics['quality_score']['precision']:.3f}")
    print(f"  - Recall: {metrics['quality_score']['recall']:.3f}")
    print(f"  - F1: {metrics['quality_score']['f1_score']:.3f}")
    print(f"\nStability Score:")
    print(f"  - AUC: {metrics['stability_score']['auc_roc']:.3f}")
    print(f"  - Precision: {metrics['stability_score']['precision']:.3f}")
    print(f"  - Recall: {metrics['stability_score']['recall']:.3f}")
    print(f"  - F1: {metrics['stability_score']['f1_score']:.3f}")
    print(f"\nCombined Model:")
    print(f"  - AUC: {metrics['combined_model']['auc_roc']:.3f}")
    print(f"  - Precision: {metrics['combined_model']['precision']:.3f} [WARNING] 低精确率")
    print(f"  - Recall: {metrics['combined_model']['recall']:.3f} [OK] 高召回率")
    print(f"  - F1: {metrics['combined_model']['f1_score']:.3f} [FAIL] 未达标，目标>=0.90")
    
    print("\n" + "="*80)
    print("问题诊断")
    print("="*80)
    print("当前two_stage策略等价于逻辑OR：")
    print("  - 只要Quality或Stability有一个判异常，就判为异常")
    print("  - 优势：召回率非常高（0.95），几乎不漏检")
    print("  - 劣势：精确率很低（0.38），误报率高达62%")
    print("  - 结果：F1=0.545，远低于0.90目标")
    
    print("\n" + "="*80)
    print("理论分析：不同组合策略的预期效果")
    print("="*80)
    
    # 基于当前指标计算理论值
    q_prec = metrics['quality_score']['precision']
    q_rec = metrics['quality_score']['recall']
    s_prec = metrics['stability_score']['precision']
    s_rec = metrics['stability_score']['recall']
    
    print(f"\n[1] OR策略（当前）:")
    print(f"   - 召回率 ≈ 1 - (1-{q_rec:.3f}) * (1-{s_rec:.3f}) = {1-(1-q_rec)*(1-s_rec):.3f}")
    print(f"   - 精确率会大幅下降（因为FP累加）")
    print(f"   - 实际: Precision={metrics['combined_model']['precision']:.3f}, Recall={metrics['combined_model']['recall']:.3f}, F1={metrics['combined_model']['f1_score']:.3f}")
    
    print(f"\n[2] AND策略（两个都判异常才判异常）:")
    print(f"   - 召回率 ≈ {q_rec:.3f} * {s_rec:.3f} = {q_rec*s_rec:.3f} [预期大幅下降]")
    print(f"   - 精确率会提升（因为只有高置信度的异常）")
    print(f"   - 权衡: 高精确率但低召回率，可能不适合质量控制场景")
    
    print(f"\n[3] Quality-First策略（只看Quality）:")
    print(f"   - Precision={q_prec:.3f}, Recall={q_rec:.3f}, F1={(2*q_prec*q_rec/(q_prec+q_rec)):.3f}")
    print(f"   - 问题: 忽略了Stability Score，浪费了0.735的AUC")
    
    print(f"\n[4] 加权组合策略（推荐尝试）:")
    print(f"   - 思路: combined_score = α * quality_score + β * stability_score")
    print(f"   - 根据AUC加权: α={0.903/(0.903+0.735):.3f}, β={0.735/(0.903+0.735):.3f}")
    print(f"   - 优势: 可以通过调整阈值平衡precision和recall")
    
    print("\n" + "="*80)
    print("改进建议")
    print("="*80)
    
    print("\n方案A: 实施加权组合策略（推荐）***")
    print("  1. 修改evaluate.py，添加'weighted'组合策略")
    print("  2. 使用公式: combined_score = 0.55*norm(quality) + 0.45*norm(stability)")
    print("  3. 通过F1优化找到最佳阈值")
    print("  4. 预期效果: F1可能提升至0.70-0.80")
    
    print("\n方案B: 尝试AND策略（仅供对比）")
    print("  命令: python scripts/evaluate.py --model-dir models/DVP/v1.12_varsm_21k \\")
    print("                                    --combine-strategy and \\")
    print("                                    --test-data-npz data/export_v11/test_subset.npz")
    print("  预期: 精确率提升，但召回率可能降至0.57（不推荐）")
    
    print("\n方案C: 调整阈值而不改变策略（临时措施）")
    print("  1. 提高Quality和Stability的判异常阈值")
    print("  2. 降低误报率，提升精确率")
    print("  3. 但召回率会下降，需要权衡")
    
    print("\n方案D: 半监督学习（长期方案）")
    print("  1. 收集100-200个标注样本（正常+异常）")
    print("  2. 训练残差分类器")
    print("  3. 融合三个通道（Quality + Stability + Residual Classifier）")
    print("  4. 预期: F1可能提升至0.80-0.90")
    
    print("\n" + "="*80)
    print("推荐行动计划")
    print("="*80)
    print("\n[P0] 优先级最高: 实施加权组合策略（预计2小时）")
    print("   - 修改evaluate.py添加'weighted'策略")
    print("   - 重新评估性能")
    print("   - 如果F1>=0.80，可直接部署")
    
    print("\n[P1] 优先级次之: 如果加权策略F1仍<0.80，考虑半监督学习（预计1-2天）")
    print("   - 人工标注100-200个样本")
    print("   - 训练残差分类器")
    print("   - 融合评估")
    
    print("\n[优势] 当前模型优势: Stability AUC=0.735已超额达标22.5%")
    print("[问题] 当前模型问题: Combined F1=0.545，主要因为OR策略导致精确率过低")
    print("[思路] 核心思路: 从逻辑OR改为加权组合，使两个分数协同而非简单叠加")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

