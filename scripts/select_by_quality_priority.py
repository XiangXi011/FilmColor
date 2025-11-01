#!/usr/bin/env python3
"""
基于Quality Score优先筛选训练数据
- 优先选择quality得分最高的样本
- 设置quality最低阈值（如P70）
- 控制edge样本占比（如10%）
- 目标：进一步提升训练集纯度
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 确保可以从项目根目录导入模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def main():
    parser = argparse.ArgumentParser(description='基于Quality Score优先筛选训练数据')
    parser.add_argument('--input', required=True, help='输入scored_all.csv路径（已包含quality评分）')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--quality-min-pctl', type=float, default=70.0, 
                        help='Quality Score最低分位数(%)，默认P70')
    parser.add_argument('--high-quality-pctl', type=float, default=90.0,
                        help='高质量阈值分位数(%)，默认P90')
    parser.add_argument('--target-samples', type=int, default=30000,
                        help='目标样本数量，默认30000')
    parser.add_argument('--edge-max-ratio', type=float, default=0.10,
                        help='边缘样本最大占比，默认10%')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--exclude-index', type=str, nargs='+', default=None,
                        help='历史训练索引CSV（含index/hash列）')
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取已评分数据
    print(f"读取评分数据: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding='utf-8-sig')
    except:
        df = pd.read_csv(args.input)
    
    print(f"总样本数: {len(df)}")
    
    # 确保有必要的列
    if 'similarity' not in df.columns:
        if 'similarity_score' in df.columns:
            df['similarity'] = df['similarity_score']
        else:
            raise ValueError("缺少similarity或similarity_score列")
    
    if 'pearson' not in df.columns:
        if 'weighted_pearson' in df.columns:
            df['pearson'] = df['weighted_pearson']
        else:
            raise ValueError("缺少pearson或weighted_pearson列")
    
    # 去重（如果有历史训练数据）
    if args.exclude_index:
        exclude_idx = set()
        for path in args.exclude_index:
            if Path(path).exists():
                try:
                    hist_df = pd.read_csv(path)
                    if 'index' in hist_df.columns:
                        exclude_idx.update(hist_df['index'].tolist())
                except Exception as e:
                    print(f"警告：无法读取历史索引 {path}: {e}")
        
        if exclude_idx:
            before = len(df)
            df = df[~df.index.isin(exclude_idx)]
            print(f"去重：剔除 {before - len(df)} 个历史样本，剩余 {len(df)} 个")
    
    # 计算Quality Score阈值
    quality_min = np.percentile(df['similarity'], args.quality_min_pctl)
    quality_high = np.percentile(df['similarity'], args.high_quality_pctl)
    
    print(f"\n【Quality Score 阈值】")
    print(f"  最低阈值 (P{args.quality_min_pctl}): {quality_min:.3f}")
    print(f"  高质量阈值 (P{args.high_quality_pctl}): {quality_high:.3f}")
    
    # 过滤：只保留>=最低阈值的样本
    df_filtered = df[df['similarity'] >= quality_min].copy()
    print(f"\n过滤后样本数: {len(df_filtered)} (保留率: {len(df_filtered)/len(df)*100:.1f}%)")
    
    # 分桶
    df_filtered['bucket'] = 'mid'
    df_filtered.loc[df_filtered['similarity'] >= quality_high, 'bucket'] = 'high'
    df_filtered.loc[df_filtered['similarity'] < quality_high, 'bucket'] = 'edge'
    
    high_count = (df_filtered['bucket'] == 'high').sum()
    edge_count = (df_filtered['bucket'] == 'edge').sum()
    
    print(f"\n【样本分桶】")
    print(f"  高质量(high): {high_count} 个")
    print(f"  边缘(edge): {edge_count} 个")
    
    # 按quality降序排序
    df_sorted = df_filtered.sort_values('similarity', ascending=False).reset_index(drop=True)
    
    # 选取样本：优先高质量，控制edge占比
    target = min(args.target_samples, len(df_sorted))
    edge_max = int(target * args.edge_max_ratio)
    
    selected_indices = []
    high_selected = 0
    edge_selected = 0
    
    for idx, row in df_sorted.iterrows():
        if len(selected_indices) >= target:
            break
        
        if row['bucket'] == 'high':
            selected_indices.append(row.name)
            high_selected += 1
        elif row['bucket'] == 'edge':
            if edge_selected < edge_max:
                selected_indices.append(row.name)
                edge_selected += 1
    
    df_selected = df.loc[selected_indices].copy()
    
    print(f"\n【筛选结果】")
    print(f"  目标样本数: {target}")
    print(f"  实际选中: {len(df_selected)}")
    print(f"  高质量: {high_selected} ({high_selected/len(df_selected)*100:.1f}%)")
    print(f"  边缘: {edge_selected} ({edge_selected/len(df_selected)*100:.1f}%)")
    
    # 划分train/val/test
    from sklearn.model_selection import train_test_split
    
    indices = df_selected.index.tolist()
    train_idx, temp_idx = train_test_split(
        indices, train_size=args.train_ratio, random_state=42
    )
    val_ratio_adjusted = args.val_ratio / (1 - args.train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, random_state=42
    )
    
    print(f"\n【数据划分】")
    print(f"  训练集: {len(train_idx)} ({len(train_idx)/len(df_selected)*100:.1f}%)")
    print(f"  验证集: {len(val_idx)} ({len(val_idx)/len(df_selected)*100:.1f}%)")
    print(f"  测试集: {len(test_idx)} ({len(test_idx)/len(df_selected)*100:.1f}%)")
    
    # 保存索引
    pd.DataFrame({'index': train_idx}).to_csv(out_dir / 'train_index.csv', index=False)
    pd.DataFrame({'index': val_idx}).to_csv(out_dir / 'val_index.csv', index=False)
    pd.DataFrame({'index': test_idx}).to_csv(out_dir / 'test_index.csv', index=False)
    
    # 保存摘要
    summary = {
        'total': len(df),
        'filtered': len(df_filtered),
        'selected': len(df_selected),
        'high': high_selected,
        'edge': edge_selected,
        'quality_min': quality_min,
        'quality_high': quality_high,
        'edge_ratio': edge_selected / len(df_selected)
    }
    
    with open(out_dir / 'selection_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())
    
    # 生成下一轮排除清单
    next_exclude = pd.DataFrame({'index': df_selected.index.tolist()})
    next_exclude.to_csv(out_dir / 'exclude_index_next.csv', index=False)
    
    print(f"\n✅ 筛选完成！")
    print(f"输出目录: {out_dir}")
    print(f"  - train_index.csv: {len(train_idx)} 个样本")
    print(f"  - val_index.csv: {len(val_idx)} 个样本")
    print(f"  - test_index.csv: {len(test_idx)} 个样本")
    print(f"  - selection_summary.csv: 筛选统计")
    print(f"  - exclude_index_next.csv: 下一轮排除清单")


if __name__ == '__main__':
    main()

