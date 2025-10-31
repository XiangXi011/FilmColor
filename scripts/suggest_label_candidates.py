#!/usr/bin/env python3
"""
主动学习候选样本生成脚本

目标：从未标注批次中挑选“最不确定”的样本清单，用于优先标注。

判定口径：two_stage
- 质量阈值 q_th（高于为通过）
- 稳定性阈值 s_th（低于为通过，分数为重构误差）

不确定度度量（越小越不确定→优先标注）：
- 若 q >= q_th（质量通过）：use margin_s = s_th - s
- 若 q <  q_th（质量不通过）：use margin_q = q - q_th
- uncertainty = |margin|（绝对距离边界）

输出：按 uncertainty 升序的前K条列表。

示例：
python scripts/suggest_label_candidates.py \
  --model-dir models/DVP/v1.10_p50_w560_release \
  --input-csv data/predict_test.csv \
  --top-k 50 \
  --output-csv output/active_label_suggestions.csv
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
import joblib

# 项目内导入
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def read_csv_spectra(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        numeric_cols, numeric_wls = [], []
        for col in df.columns:
            try:
                wl = float(col)
                numeric_cols.append(col)
                numeric_wls.append(wl)
            except Exception:
                continue
        if len(numeric_cols) >= 3:
            order = np.argsort(np.array(numeric_wls, dtype=float))
            sorted_cols = [numeric_cols[i] for i in order]
            wavelengths = np.array([numeric_wls[i] for i in order], dtype=float)
            spectra = df[sorted_cols].to_numpy(dtype=float)
            return wavelengths, spectra
    except Exception:
        pass
    rows = pd.read_csv(csv_path, header=None).values
    wavelengths = rows[0].astype(float)
    spectra = rows[1:].astype(float)
    return wavelengths, spectra


def load_model_components(model_dir: Path):
    def _find_one(patterns):
        for pattern in patterns:
            matches = sorted(model_dir.glob(pattern))
            if matches:
                return matches[0]
        return None
    encoder = joblib.load(_find_one(["*encoder*DVP*v*.joblib", "*encoder*DVP*.joblib", "*encoder*.joblib"]))
    decoder = joblib.load(_find_one(["*decoder*DVP*v*.joblib", "*decoder*DVP*.joblib", "*decoder*.joblib"]))
    scaler = joblib.load(_find_one(["*scaler*DVP*v*.joblib", "*scaler*DVP*.joblib", "*scaler*.joblib"]))
    weights_path = _find_one(["weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"]) 
    weights = np.load(weights_path) if weights_path else None
    metadata_path = _find_one(["*metadata*DVP*v*.json", "*metadata*DVP*.json", "*metadata*.json"]) 
    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    return encoder, decoder, scaler, weights, metadata


def main():
    parser = argparse.ArgumentParser(description='主动学习候选样本生成器（two_stage判定边界）')
    parser.add_argument('--model-dir', required=True, help='模型目录')
    parser.add_argument('--input-csv', help='输入CSV（列名为波长或首行波长）')
    parser.add_argument('--input-npz', help='输入NPZ（wavelengths, dvp_values）')
    parser.add_argument('--top-k', type=int, default=50, help='输出最不确定TopK')
    parser.add_argument('--output-csv', required=True, help='输出候选清单CSV（指标+不确定度）')
    parser.add_argument('--export-spectra-csv', type=str, default=None, help='可选：导出与训练集一致的光谱CSV（首行波长，后续为样本）')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    encoder, decoder, scaler, weights, metadata = load_model_components(model_dir)

    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()
    evaluator = SimilarityEvaluator('DVP')

    # 读取输入
    if args.input_csv:
        src_wl, spectra = read_csv_spectra(args.input_csv)
    else:
        data = np.load(args.input_npz)
        src_wl = data['wavelengths']
        spectra = data['dvp_values']

    if not np.array_equal(src_wl, std_wl):
        spectra = np.vstack([loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in spectra])

    # 阈值
    q_th = metadata.get('similarity_evaluator', {}).get('quality_threshold', 92.0)
    if q_th > 1:
        q_th = q_th / 100.0
    # 稳定性阈值
    s_th = metadata.get('weighted_autoencoder', {}).get('stability_threshold', None)
    # 计算分数
    quality_scores, stability_scores = [], []
    for sp in spectra:
        q = evaluator.evaluate(sp, std_curve, std_wl, coating_name='DVP')['similarity_score']
        quality_scores.append(q)
        sp_scaled = scaler.transform(sp.reshape(1, -1))
        z = encoder.predict(sp_scaled)
        rec = decoder.predict(z)
        sp_orig = scaler.inverse_transform(sp_scaled)[0]
        rec = rec[0]
        s = float(np.mean(weights * (sp_orig - rec) ** 2)) if weights is not None else float(np.mean((sp_orig - rec) ** 2))
        stability_scores.append(s)
    quality_scores = np.array(quality_scores)
    stability_scores = np.array(stability_scores)

    if (s_th is None) or (s_th <= 0):
        s_th = float(np.percentile(stability_scores, 95))

    # 计算不确定度
    passed_quality = (quality_scores >= q_th)
    margin_quality = quality_scores - q_th            # 质量通过时应为正
    margin_stability = s_th - stability_scores        # 稳定性通过时应为正
    effective_margin = np.where(passed_quality, margin_stability, margin_quality)
    uncertainty = np.abs(effective_margin)

    df = pd.DataFrame({
        'index': np.arange(len(spectra), dtype=int),
        'quality_score': quality_scores,
        'stability_score': stability_scores,
        'margin_quality': margin_quality,
        'margin_stability': margin_stability,
        'effective_margin': effective_margin,
        'uncertainty': uncertainty,
        'suggest_label': 'TO_ANNOTATE'
    }).sort_values('uncertainty', ascending=True).head(args.top_k)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f'已生成候选标注清单: {args.output_csv}  (TopK={args.top_k}, q_th={q_th:.6f}, s_th={s_th:.6f})')

    # 可选：导出与训练集一致的光谱CSV
    if args.export_spectra_csv:
        sel_idx = df['index'].astype(int).values
        spectra_sel = spectra[sel_idx]
        out_path = Path(args.export_spectra_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8-sig') as f:
            # 首行写波长
            f.write(','.join([f"{x:g}" for x in std_wl]) + "\n")
            for row in spectra_sel:
                f.write(','.join([f"{x:.6f}" for x in row]) + "\n")
        print(f'已导出候选光谱CSV（训练格式）: {out_path}')


if __name__ == '__main__':
    main()


