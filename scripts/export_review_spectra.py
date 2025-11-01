#!/usr/bin/env python3
"""导出待复核误报样本的光谱数据

输入：
- evaluation/false_positive_analysis/false_positive_samples_for_review.csv
- evaluation/false_positive_analysis/false_positive_samples.npz

输出：
- evaluation/false_positive_analysis/false_positive_samples_for_review_spectra.csv

CSV 格式：首列为 sample_idx，后续列为波长（nm）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / 'evaluation' / 'false_positive_analysis'

    review_csv = eval_dir / 'false_positive_samples_for_review.csv'
    source_npz = eval_dir / 'false_positive_samples.npz'
    output_csv = eval_dir / 'false_positive_samples_for_review_spectra.csv'

    if not review_csv.exists():
        raise FileNotFoundError(f'未找到待复核列表: {review_csv}')
    if not source_npz.exists():
        raise FileNotFoundError(f'未找到误报样本光谱 NPZ: {source_npz}')

    review_df = pd.read_csv(review_csv)
    review_indices = review_df['sample_idx'].astype(int).tolist()

    data = np.load(source_npz)
    sample_indices = data['sample_indices']
    spectra = data['spectra']
    wavelengths = data['wavelengths']

    index_map = {int(idx): i for i, idx in enumerate(sample_indices)}

    rows = []
    missing = []
    for idx in review_indices:
        key = int(idx)
        pos = index_map.get(key)
        if pos is None:
            missing.append(key)
            continue
        rows.append((key, spectra[pos]))

    if missing:
        raise ValueError(f'以下 sample_idx 未在 NPZ 中找到: {missing}')

    columns = ['sample_idx'] + [f'{wl:g}' for wl in wavelengths]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for idx, spectrum in rows:
            writer.writerow([idx] + [f'{x:.6f}' for x in spectrum])

    print(f'[完成] 已导出 {len(rows)} 条光谱数据 -> {output_csv}')


if __name__ == '__main__':
    main()


