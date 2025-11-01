#!/usr/bin/env python3
"""生成半监督标注优先级样本清单

根据最新的 Quality / Stability 分布，优先挑选位于质量边界区间的样本，
用于后续人工标注与半监督训练。

优先级策略（可通过参数覆盖）：
1. Stability 误报样本（Quality ≥ 质量阈值、Stability > 阈值）→ 按重构误差超阈值幅度升序
2. Quality 边界通过样本（质量介于[质量阈值, 生产阈值)，Stability ≤ 阈值）→ 按距离质量阈值升序
3. Quality 边界未通过样本（质量略低于阈值，默认容忍 0.01 的下限）→ 按距离阈值升序

输出 CSV 字段：
- priority_rank
- sample_idx
- priority_group
- quality_score
- stability_score
- quality_margin (quality_score - quality_threshold)
- stability_margin (stability_threshold - stability_score)
- quality_minus_prod_threshold

用法示例：

```bash
python scripts/generate_labeling_priority.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --test-data-npz data/export_v11/test_subset.npz \
    --output-csv evaluation/false_positive_analysis/labeling_priority_candidates.csv
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

# 项目内部模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in __import__('sys').path:
    __import__('sys').path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader  # type: ignore  # noqa: E402
from algorithms.similarity_evaluator import SimilarityEvaluator  # type: ignore  # noqa: E402


def _find_one(model_dir: Path, patterns: Tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_model_components(model_dir: Path):
    encoder = joblib.load(_find_one(model_dir, (
        "*encoder*DVP*v*.joblib", "*encoder*DVP*.joblib", "*encoder*.joblib"
    )))
    decoder = joblib.load(_find_one(model_dir, (
        "*decoder*DVP*v*.joblib", "*decoder*DVP*.joblib", "*decoder*.joblib"
    )))
    scaler = joblib.load(_find_one(model_dir, (
        "*scaler*DVP*v*.joblib", "*scaler*DVP*.joblib", "*scaler*.joblib"
    )))
    weights_path = _find_one(model_dir, (
        "weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"
    ))
    weights = np.load(weights_path) if weights_path else None

    metadata_path = _find_one(model_dir, (
        "*metadata*DVP*v*.json", "*metadata*DVP*.json", "*metadata*.json"
    ))
    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    return encoder, decoder, scaler, weights, metadata


def compute_scores(spectra: np.ndarray,
                   encoder, decoder, scaler, weights,
                   similarity: SimilarityEvaluator,
                   standard_wl: np.ndarray, standard_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    quality_scores = []
    stability_scores = []

    for spectrum in spectra:
        q = similarity.evaluate(spectrum, standard_curve, standard_wl, coating_name='DVP')['similarity_score']
        quality_scores.append(float(q))

        scaled = scaler.transform(spectrum.reshape(1, -1))
        latent = encoder.predict(scaled)
        reconstructed = decoder.predict(latent)
        original = scaler.inverse_transform(scaled)[0]
        reconstructed = reconstructed[0]

        if weights is not None:
            err = float(np.mean(weights * (original - reconstructed) ** 2))
        else:
            err = float(np.mean((original - reconstructed) ** 2))
        stability_scores.append(err)

    return np.array(quality_scores), np.array(stability_scores)


def main():
    parser = argparse.ArgumentParser(description='生成半监督标注优先级样本清单')
    parser.add_argument('--model-dir', type=str,
                        default=str(PROJECT_ROOT / 'models' / 'DVP' / 'v1.12_varsm_21k'),
                        help='模型目录（默认使用 v1.12_varsm_21k）')
    parser.add_argument('--test-data-npz', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'export_v11' / 'test_subset.npz'),
                        help='NPZ 测试数据，需包含 dvp_values 与 wavelengths')
    parser.add_argument('--output-csv', type=str,
                        default=str(PROJECT_ROOT / 'evaluation' / 'false_positive_analysis' / 'labeling_priority_candidates.csv'),
                        help='输出 CSV 路径')
    parser.add_argument('--quality-threshold', type=float, default=0.7622887361803471,
                        help='质量统计阈值（默认使用 P5 结果）')
    parser.add_argument('--production-threshold', type=float, default=0.80,
                        help='生产使用的判定阈值')
    parser.add_argument('--stability-threshold', type=float, default=23.36608443434134,
                        help='稳定性统计阈值（默认 P95）')
    parser.add_argument('--quality-lower-gap', type=float, default=0.01,
                        help='Quality 低于阈值时，仍纳入的最大距离（默认 0.01）')
    parser.add_argument('--fp-count', type=int, default=120,
                        help='优先挑选的 Stability 误报样本数量')
    parser.add_argument('--edge-pass-count', type=int, default=60,
                        help='优先挑选的质量边界通过样本数量')
    parser.add_argument('--edge-fail-count', type=int, default=20,
                        help='优先挑选的质量边界未通过样本数量')
    parser.add_argument('--export-spectra-csv', type=str, default=None,
                        help='可选：导出光谱数据（首列 sample_idx，后续为波长列）')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    encoder, decoder, scaler, weights, metadata = load_model_components(model_dir)

    data = np.load(args.test_data_npz)
    wavelengths = data['wavelengths']
    spectra = data['dvp_values']

    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()
    similarity = SimilarityEvaluator('DVP')

    if not np.array_equal(wavelengths, std_wl):
        spectra_aligned = np.vstack([
            loader.interpolate_to_standard_grid(wavelengths, sp, std_wl)
            for sp in spectra
        ])
    else:
        spectra_aligned = spectra

    q_scores, s_scores = compute_scores(
        spectra_aligned, encoder, decoder, scaler, weights,
        similarity, std_wl, std_curve
    )

    df = pd.DataFrame({
        'sample_idx': np.arange(len(q_scores), dtype=int),
        'quality_score': q_scores,
        'stability_score': s_scores
    })

    df['quality_margin'] = df['quality_score'] - args.quality_threshold
    df['stability_margin'] = args.stability_threshold - df['stability_score']
    df['quality_minus_prod_threshold'] = df['quality_score'] - args.production_threshold

    fp_mask = (df['quality_margin'] >= 0) & (df['stability_margin'] < 0)
    edge_pass_mask = (
        (df['quality_margin'] >= 0) &
        (df['quality_score'] < args.production_threshold) &
        (df['stability_margin'] >= 0)
    )
    edge_fail_mask = (
        (df['quality_margin'] < 0) &
        (df['quality_margin'] >= -args.quality_lower_gap)
    )

    fp_candidates = df[fp_mask].copy()
    fp_candidates['priority_group'] = 'stability_fp'
    fp_candidates['priority_score'] = np.abs(fp_candidates['stability_margin'])

    edge_pass_candidates = df[edge_pass_mask].copy()
    edge_pass_candidates['priority_group'] = 'quality_edge_pass'
    edge_pass_candidates['priority_score'] = np.abs(edge_pass_candidates['quality_margin'])

    edge_fail_candidates = df[edge_fail_mask].copy()
    edge_fail_candidates['priority_group'] = 'quality_edge_fail'
    edge_fail_candidates['priority_score'] = np.abs(edge_fail_candidates['quality_margin'])

    fp_selected = fp_candidates.sort_values(
        ['priority_score', 'stability_score'], ascending=[True, False]
    ).head(args.fp_count)

    edge_pass_selected = edge_pass_candidates.sort_values(
        ['priority_score', 'quality_score'], ascending=[True, True]
    ).head(args.edge_pass_count)

    edge_fail_selected = edge_fail_candidates.sort_values(
        'priority_score', ascending=True
    ).head(args.edge_fail_count)

    selected = pd.concat(
        [fp_selected, edge_pass_selected, edge_fail_selected],
        ignore_index=True
    ).sort_values('priority_score', ascending=True)

    selected = selected.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    selected['priority_rank'] = np.arange(1, len(selected) + 1)
    selected['quality_threshold'] = args.quality_threshold
    selected['production_threshold'] = args.production_threshold
    selected['stability_threshold'] = args.stability_threshold

    output_columns = [
        'priority_rank', 'sample_idx', 'priority_group',
        'quality_score', 'stability_score',
        'quality_margin', 'stability_margin',
        'quality_minus_prod_threshold'
    ]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(output_path, index=False)

    print(f'[完成] 已输出 {len(selected)} 条候选样本 -> {output_path}')
    group_counts = selected.groupby('priority_group')['sample_idx'].count()
    for group, count in group_counts.items():
        print(f'  - {group}: {count} 条')

    if args.export_spectra_csv:
        spectra_df = pd.DataFrame(
            spectra_aligned[selected['sample_idx'].astype(int).values],
            columns=[f'{wl:g}' for wl in std_wl]
        )
        spectra_df.insert(0, 'sample_idx', selected['sample_idx'].values)
        spectra_output = Path(args.export_spectra_csv)
        spectra_output.parent.mkdir(parents=True, exist_ok=True)
        spectra_df.to_csv(spectra_output, index=False, encoding='utf-8-sig')
        print(f'[完成] 已导出光谱数据 -> {spectra_output}')


if __name__ == '__main__':
    main()


