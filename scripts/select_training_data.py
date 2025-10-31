#!/usr/bin/env python3
"""
选择并扩充自编码器训练集的辅助脚本

功能：
- 读取CSV（首行波长，后续每行一个样本）
- 计算每条样本的 Quality Score 与加权皮尔逊
- 按阈值分桶（高置信/边缘/其余），并控制边缘样本占比
- 可剔除“历史训练样本”（按索引/哈希）避免重复
- 导出筛选结果与训练清单

用法示例：
python scripts/select_training_data.py \
  --input data/DVP_train_sample.csv \
  --output-dir data/selection \
  --sim-min 0.68 --pearson-min 0.88 \
  --high-sim 0.90 --high-pearson 0.95 \
  --edge-max-ratio 0.2 \
  --exclude-index data/selection/prev_train_index.csv
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# 确保可以从项目根目录导入模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 项目内导入
from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def read_csv_spectra(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """通用读取：支持首列ID/非数值列，自动识别数值列为波长。

    规则：
    - 优先将列名可转换为float的列视为波长列，按数值升序排列
    - 若无数值列名，则回退到“首行即为波长”的旧格式
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path)

    numeric_cols = []
    numeric_wls = []
    for col in df.columns:
        try:
            wl = float(col)
            numeric_cols.append(col)
            numeric_wls.append(wl)
        except Exception:
            continue

    if len(numeric_cols) >= 3:  # 期望有多个波长列
        # 按波长排序
        order = np.argsort(np.array(numeric_wls, dtype=float))
        sorted_cols = [numeric_cols[i] for i in order]
        wavelengths = np.array([numeric_wls[i] for i in order], dtype=float)
        spectra = df[sorted_cols].to_numpy(dtype=float)
        return wavelengths, spectra

    # 回退：按旧格式解析（首行波长、后续行为样本）
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    wavelengths = np.array([float(x) for x in rows[0]], dtype=float)
    spectra = np.array([[float(x) for x in row] for row in rows[1:]], dtype=float)
    return wavelengths, spectra


def hash_spectrum(values: np.ndarray) -> str:
    m = hashlib.sha256()
    m.update(values.astype(np.float32).tobytes())
    return m.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='训练集筛选与扩充工具')
    parser.add_argument('--input', required=True, help='输入CSV路径（首行波长或列名为波长；支持含ID/时间/批次列）')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    # 阈值：支持固定值或分位数
    parser.add_argument('--sim-min', type=float, default=None, help='边缘相似度下限（0-1）')
    parser.add_argument('--pearson-min', type=float, default=None, help='边缘加权皮尔逊下限（-1~1）')
    parser.add_argument('--high-sim', type=float, default=None, help='高置信相似度阈值（0-1）')
    parser.add_argument('--high-pearson', type=float, default=None, help='高置信加权皮尔逊阈值（-1~1）')
    parser.add_argument('--sim-min-pctl', type=float, default=35.0, help='相似度下限分位数(%)，当未显式提供sim-min时生效')
    parser.add_argument('--pearson-min-pctl', type=float, default=35.0, help='皮尔逊下限分位数(%)，当未显式提供pearson-min时生效')
    parser.add_argument('--high-sim-pctl', type=float, default=85.0, help='相似度高置信分位数(%)，当未显式提供high-sim时生效')
    parser.add_argument('--high-pearson-pctl', type=float, default=90.0, help='皮尔逊高置信分位数(%)，当未显式提供high-pearson时生效')
    parser.add_argument('--edge-max-ratio', type=float, default=0.2, help='边缘占比(相对于高置信数量)，当高置信=0时不生效')
    parser.add_argument('--edge-max-ratio-total', type=float, default=0.1, help='边缘占比(相对于总样本数量)')
    parser.add_argument('--edge-max-abs', type=int, default=5000, help='边缘样本最大绝对数量上限')
    parser.add_argument('--edge-sort-by', type=str, choices=['similarity','pearson','mix'], default='mix',
                        help='边缘样本排序依据，mix=平均(sim, 归一化pearson)')
    parser.add_argument('--exclude-index', type=str, nargs='+', default=None, help='一个或多个历史训练索引CSV（含 index/hash 列）')
    # 分组与拆分
    parser.add_argument('--group-col', type=str, default=None, help='按该列分组限额（如批次/日期）')
    parser.add_argument('--per-group-cap', type=int, default=None, help='每组最多选入数量上限（edge选入时应用）')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    # 余下为test比例

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取样本
    src_wl, spectra = read_csv_spectra(args.input)

    # 加载黄金标准
    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()

    # 若波长网格不一致，插值到标准网格
    if not np.array_equal(src_wl, std_wl):
        spectra_interp = np.vstack([
            loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in spectra
        ])
    else:
        spectra_interp = spectra

    # 质量评估
    evaluator = SimilarityEvaluator('DVP')
    df = evaluator.batch_evaluate(list(spectra_interp), std_curve, std_wl, 'DVP')
    # 归一化/确保数值范围
    df['similarity'] = df['similarity_score'].astype(float)  # 0-1
    df['pearson'] = df['weighted_pearson'].astype(float)

    # 计算哈希用于去重
    hashes: List[str] = [hash_spectrum(sp) for sp in spectra_interp]
    df['hash'] = hashes

    # 去重（可选）
    if args.exclude_index:
        exclude_hash = set()
        exclude_idx = set()
        total_prev = 0
        for path in args.exclude_index:
            if os.path.exists(path):
                prev = pd.read_csv(path)
                total_prev += len(prev)
                if 'hash' in prev.columns:
                    exclude_hash.update(prev['hash'].astype(str).tolist())
                if 'index' in prev.columns:
                    exclude_idx.update(prev['index'].astype(int).tolist())
        before = len(df)
        if exclude_hash or exclude_idx:
            df = df[~(df['hash'].isin(exclude_hash) | df['spectrum_id'].isin(exclude_idx))].reset_index(drop=True)
        print(f"已剔除历史训练样本: {before - len(df)} 条 (历史索引合计: {total_prev})")

    # 若未提供固定阈值，使用分位数计算
    sim_min = args.sim_min if args.sim_min is not None else float(np.percentile(df['similarity'].dropna(), args.sim_min_pctl))
    pear_min = args.pearson_min if args.pearson_min is not None else float(np.percentile(df['pearson'].dropna(), args.pearson_min_pctl))
    sim_high = args.high_sim if args.high_sim is not None else float(np.percentile(df['similarity'].dropna(), args.high_sim_pctl))
    pear_high = args.high_pearson if args.high_pearson is not None else float(np.percentile(df['pearson'].dropna(), args.high_pearson_pctl))

    # 分桶
    high_mask = (df['similarity'] >= sim_high) & (df['pearson'] >= pear_high)
    edge_mask = (~high_mask) & (df['similarity'] >= sim_min) & (df['pearson'] >= pear_min)
    df['bucket'] = np.where(high_mask, 'high', np.where(edge_mask, 'edge', 'other'))

    # 控制边缘样本占比与数量
    high_df = df[df['bucket'] == 'high']
    edge_df = df[df['bucket'] == 'edge']
    other_df = df[df['bucket'] == 'other']

    # 计算边缘选择上限：综合相对高置信占比、相对总量占比与绝对上限
    base_for_ratio = len(high_df) if len(high_df) > 0 else 0
    cap_by_high = int(np.floor(args.edge_max_ratio * base_for_ratio)) if base_for_ratio > 0 else 0
    cap_by_total = int(np.floor(args.edge_max_ratio_total * len(df)))
    cap_abs = int(args.edge_max_abs)
    max_edge = min(len(edge_df), max(cap_by_high, cap_by_total), cap_abs)

    # 边缘排序策略
    if args.edge_sort_by == 'similarity':
        edge_df_sorted = edge_df.sort_values('similarity', ascending=False)
    elif args.edge_sort_by == 'pearson':
        edge_df_sorted = edge_df.sort_values('pearson', ascending=False)
    else:
        # mix：平均(sim, 归一化pearson) ；pearson∈[-1,1] → [0,1]
        mix_score = edge_df[['similarity','pearson']].copy()
        mix_score['pearson01'] = (mix_score['pearson'] + 1.0) / 2.0
        edge_df_sorted = edge_df.assign(mix=(mix_score['similarity'] + mix_score['pearson01'])/2.0)
        edge_df_sorted = edge_df_sorted.sort_values('mix', ascending=False)

    # 分组限额（可选）
    if args.group_col and args.group_col in df.columns and args.per_group_cap:
        edge_selected_parts = []
        for g, sub in edge_df_sorted.groupby(args.group_col):
            edge_selected_parts.append(sub.head(args.per_group_cap))
        edge_selected = pd.concat(edge_selected_parts, axis=0).head(max_edge)
    else:
        edge_selected = edge_df_sorted.head(max_edge)
    selected = pd.concat([high_df, edge_selected], axis=0).reset_index(drop=True)

    # 导出
    summary = {
        'total': len(df),
        'high': int(len(high_df)),
        'edge_total': int(len(edge_df)),
        'edge_selected': int(len(edge_selected)),
        'edge_max_ratio': args.edge_max_ratio,
        'edge_max_ratio_total': args.edge_max_ratio_total,
        'edge_max_abs': args.edge_max_abs,
        'other': int(len(other_df)),
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'selection_summary.csv', index=False)
    df.to_csv(out_dir / 'scored_all.csv', index=False)
    idx_df = selected[['spectrum_id', 'hash', 'similarity', 'pearson']].rename(columns={'spectrum_id': 'index'}).reset_index(drop=True)
    # 拆分 train/val/test
    n = len(idx_df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val
    idx_train = idx_df.iloc[:n_train]
    idx_val = idx_df.iloc[n_train:n_train+n_val]
    idx_test = idx_df.iloc[n_train+n_val:]
    idx_train_path = out_dir / 'train_index.csv'
    idx_val_path = out_dir / 'val_index.csv'
    idx_test_path = out_dir / 'test_index.csv'
    idx_train.to_csv(idx_train_path, index=False)
    idx_val.to_csv(idx_val_path, index=False)
    idx_test.to_csv(idx_test_path, index=False)
    # 生成下一轮的合并剔除清单（历史合并 + 本轮train）
    try:
        exclude_next = pd.DataFrame()
        if args.exclude_index:
            dfs = []
            for path in args.exclude_index:
                if os.path.exists(path):
                    dfs.append(pd.read_csv(path))
            if dfs:
                exclude_next = pd.concat(dfs, axis=0, ignore_index=True)
        exclude_next = pd.concat([exclude_next, idx_train], axis=0, ignore_index=True).drop_duplicates(subset=['hash'], keep='first')
        exclude_next.to_csv(out_dir / 'exclude_index_next.csv', index=False)
    except Exception:
        pass

    print("筛选完成：")
    print(f"  总样本: {summary['total']}")
    print(f"  高置信: {summary['high']}")
    print(f"  边缘(候选/入选): {summary['edge_total']} / {summary['edge_selected']}  (规则: max(high*{args.edge_max_ratio}, total*{args.edge_max_ratio_total}, abs={args.edge_max_abs}))")
    print(f"  其余: {summary['other']}")
    print(f"输出: {out_dir}")
    print(f"  阈值: sim_min={sim_min:.3f} pear_min={pear_min:.3f} | sim_high={sim_high:.3f} pear_high={pear_high:.3f}")
    print(f"  拆分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print(f"  下一轮剔除清单: {out_dir/'exclude_index_next.csv'}")


if __name__ == '__main__':
    main()


