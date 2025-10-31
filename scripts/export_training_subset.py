#!/usr/bin/env python3
"""
根据筛选索引(train_index.csv)从原始CSV导出训练子集

输入：
- 原始CSV（首行波长，后续每行一个样本）
- 由 select_training_data.py 生成的 train_index.csv（含 index/hash）

输出：
- 选中样本的CSV（同原格式）
- 选中样本插值到标准网格后的NPZ（wavelengths, dvp_values）

用法示例：
python scripts/export_training_subset.py \
  --input data/DVP_train_sample.csv \
  --index-csv data/selection/train_index.csv \
  --output-dir data/selection/export
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# 确保可以从项目根目录导入模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader


def read_csv_spectra(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    wavelengths = np.array([float(x) for x in rows[0]], dtype=float)
    spectra = np.array([[float(x) for x in r] for r in rows[1:]], dtype=float)
    return wavelengths, spectra


def write_csv_spectra(csv_path: str, wavelengths: np.ndarray, spectra: np.ndarray) -> None:
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([f"{x:g}" for x in wavelengths])
        for sp in spectra:
            w.writerow([f"{x:.6f}" for x in sp])


def main():
    parser = argparse.ArgumentParser(description='根据索引导出训练子集')
    parser.add_argument('--input', required=True, help='原始CSV（首行波长，其余为样本）')
    parser.add_argument('--index-csv', required=True, help='train_index.csv，含 index 列')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始
    src_wl, spectra = read_csv_spectra(args.input)

    # 读取索引
    import pandas as pd
    idx_df = pd.read_csv(args.index_csv)
    if 'index' not in idx_df.columns:
        raise ValueError('index-csv 缺少 index 列')
    indices = idx_df['index'].astype(int).values

    # 边界检查
    max_idx = spectra.shape[0] - 1
    if np.any(indices < 0) or np.any(indices > max_idx):
        raise ValueError(f'index 越界：允许范围 0~{max_idx}')

    # 选择样本
    selected = spectra[indices]

    # 保存选中CSV（与原始格式一致）
    out_csv = out_dir / 'training_subset.csv'
    write_csv_spectra(str(out_csv), src_wl, selected)

    # 插值到标准网格并保存NPZ
    loader = SpectrumDataLoader()
    std_wl = loader.get_wavelength_range()  # 380~780, step=5
    if not np.array_equal(src_wl, std_wl):
        selected_interp = np.vstack([
            loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in selected
        ])
    else:
        selected_interp = selected

    out_npz = out_dir / 'training_subset.npz'
    np.savez_compressed(out_npz, wavelengths=std_wl, dvp_values=selected_interp)

    print('导出完成:')
    print(f'  选中样本数: {selected.shape[0]}')
    print(f'  CSV: {out_csv}')
    print(f'  NPZ: {out_npz}')


if __name__ == '__main__':
    main()


