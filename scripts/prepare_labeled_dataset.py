#!/usr/bin/env python3
"""合并 evaluation/label 目录下的标注光谱数据"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='合并标注光谱数据')
    parser.add_argument('--label-dir', type=str, default='evaluation/label',
                        help='标注数据目录（包含若干CSV，需含 sample_idx 与 label 列）')
    parser.add_argument('--output', type=str, default='evaluation/label/combined_labeled_spectra.csv',
                        help='输出合并后的CSV路径')
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f'未找到标注目录: {label_dir}')

    csv_files = sorted(label_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'目录 {label_dir} 下未找到CSV文件')

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, encoding='utf-8-sig')
        if 'sample_idx' not in df.columns or 'label' not in df.columns:
            print(f'⚠️ 跳过 {path}，缺少 sample_idx 或 label 列')
            continue
        frames.append(df)
        print(f'✅ 读取 {path} 共 {len(df)} 条')

    if not frames:
        raise RuntimeError('没有可合并的CSV文件')

    combined = pd.concat(frames, ignore_index=True)
    combined['label'] = combined['label'].astype(int)
    combined = combined.drop_duplicates(subset=['sample_idx'], keep='last').sort_values('sample_idx').reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')

    label_counts = combined['label'].value_counts().to_dict()
    print(f'✅ 合并完成，共 {len(combined)} 条，标签分布: {label_counts}')
    print(f'📁 输出文件: {output_path}')


if __name__ == '__main__':
    main()


