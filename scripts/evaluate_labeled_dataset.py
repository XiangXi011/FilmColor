#!/usr/bin/env python3
"""基于人工标注光谱数据评估各通道及组合策略性能"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader  # type: ignore  # noqa: E402
from algorithms.similarity_evaluator import SimilarityEvaluator  # type: ignore  # noqa: E402


def find_one(model_dir: Path, patterns: Tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_model_components(model_dir: Path):
    encoder = joblib.load(find_one(model_dir, (
        "*encoder*DVP*v*.joblib", "*encoder*DVP*.joblib", "*encoder*.joblib"
    )))
    decoder = joblib.load(find_one(model_dir, (
        "*decoder*DVP*v*.joblib", "*decoder*DVP*.joblib", "*decoder*.joblib"
    )))
    scaler = joblib.load(find_one(model_dir, (
        "*scaler*DVP*v*.joblib", "*scaler*DVP*.joblib", "*scaler*.joblib"
    )))
    weights_path = find_one(model_dir, (
        "weights*DVP*v*.npy", "weights*DVP*.npy", "weights*.npy"
    ))
    weights = np.load(weights_path) if weights_path else None

    # 残差分类器优先使用最新目录
    residual_dir = model_dir / "residual_clf"
    residual_path: Path | None = None
    if residual_dir.exists():
        joblib_files = sorted(residual_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if joblib_files:
            residual_path = joblib_files[0]
    if residual_path is None:
        residual_path = find_one(model_dir, (
            "*residual_clf*DVP*v*.joblib", "*residual_clf*DVP*.joblib", "*residual_clf*.joblib"
        ))
    residual_clf = joblib.load(residual_path) if residual_path else None

    return encoder, decoder, scaler, weights, residual_clf


def read_labeled_spectra(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError('标注数据需包含 label 列 (0=通过, 1=NG)')
    labels = df['label'].astype(int).values

    ignore_cols = {'label', 'sample_idx', 'sample_id', 'id', 'source'}
    wavelength_cols = []
    wavelengths = []
    for col in df.columns:
        if col in ignore_cols:
            continue
        try:
            wl = float(col)
        except ValueError:
            continue
        wavelength_cols.append(col)
        wavelengths.append(wl)

    if len(wavelength_cols) < 3:
        raise ValueError('无法解析光谱列，需使用列名为波长的宽表格式')

    order = np.argsort(wavelengths)
    sorted_cols = [wavelength_cols[i] for i in order]
    sorted_wl = np.array([wavelengths[i] for i in order], dtype=float)
    spectra = df[sorted_cols].to_numpy(dtype=float)
    return sorted_wl, spectra, labels


def compute_scores(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    encoder,
    decoder,
    scaler,
    weights,
    similarity: SimilarityEvaluator,
    residual_clf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()

    if not np.array_equal(wavelengths, std_wl):
        spectra = np.vstack([
            loader.interpolate_to_standard_grid(wavelengths, sp, std_wl)
            for sp in spectra
        ])
    else:
        spectra = spectra.copy()

    quality_scores = []
    stability_scores = []
    residual_features = []

    bands = [(400, 480), (480, 560), (560, 680), (680, 780)]

    for spectrum in spectra:
        res = similarity.evaluate(spectrum, std_curve, std_wl, coating_name='DVP')
        quality_scores.append(res['similarity_score'])

        scaled = scaler.transform(spectrum.reshape(1, -1))
        latent = encoder.predict(scaled)
        recon = decoder.predict(latent)
        original = scaler.inverse_transform(scaled)[0]
        recon = recon[0]
        diff = original - recon

        if weights is not None:
            stability = float(np.mean(weights * diff ** 2))
        else:
            stability = float(np.mean(diff ** 2))
        stability_scores.append(stability)

        feats = []
        for lo, hi in bands:
            mask = (std_wl >= lo) & (std_wl <= hi)
            seg = diff[mask]
            abs_seg = np.abs(seg)
            feats.extend([
                abs_seg.mean(),
                np.sqrt(np.mean(seg ** 2)),
                abs_seg.max()
            ])
        residual_features.append(feats)

    quality_scores = np.array(quality_scores)
    stability_scores = np.array(stability_scores)
    residual_features = np.array(residual_features)

    residual_scores = None
    if residual_clf is not None:
        residual_scores = residual_clf.predict_proba(residual_features)[:, 1]

    return quality_scores, stability_scores, residual_scores


def evaluate_channel(scores: np.ndarray, labels: np.ndarray, direction: str) -> Dict:
    if direction not in {'low', 'high'}:
        raise ValueError('direction must be "low" or "high"')

    search_scores = -scores if direction == 'low' else scores
    thresholds = np.linspace(search_scores.min(), search_scores.max(), 400)

    best = {
        'f1': -1.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'threshold': thresholds[0],
        'preds': np.zeros_like(labels, dtype=int)
    }

    for th in thresholds:
        preds = (search_scores > th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best['f1']:
            best.update({
                'f1': f1,
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'accuracy': accuracy_score(labels, preds),
                'threshold': th,
                'preds': preds,
            })

    auc = roc_auc_score(labels, 1 - scores if direction == 'low' else scores)
    best['auc'] = auc
    return best


def main():
    parser = argparse.ArgumentParser(description='基于标注数据评估各通道性能')
    parser.add_argument('--model-dir', required=True, help='模型目录')
    parser.add_argument('--labeled-csv', required=True, help='带标注的光谱CSV路径')
    parser.add_argument('--output-json', required=True, help='结果输出路径')
    parser.add_argument('--min-stability-weight', type=float, default=0.05,
                        help='加权组合中Stability最小权重（避免被忽略）')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    labeled_csv = Path(args.labeled_csv)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    encoder, decoder, scaler, weights, residual_clf = load_model_components(model_dir)
    wavelengths, spectra, labels = read_labeled_spectra(labeled_csv)

    similarity = SimilarityEvaluator('DVP')
    quality_scores, stability_scores, residual_scores = compute_scores(
        spectra, wavelengths, encoder, decoder, scaler, weights, similarity, residual_clf
    )

    results: Dict[str, Dict] = {}
    results['quality'] = evaluate_channel(quality_scores, labels, direction='low')
    results['stability'] = evaluate_channel(stability_scores, labels, direction='high')

    if residual_scores is not None:
        results['residual'] = evaluate_channel(residual_scores, labels, direction='high')

    # 质量优先 + 残差兜底
    if 'residual' in results:
        quality_preds = results['quality']['preds']
        residual_preds = results['residual']['preds']
        combined = np.where(quality_preds == 1, 1, residual_preds)
        results['quality_first_residual'] = {
            'f1': f1_score(labels, combined, zero_division=0),
            'precision': precision_score(labels, combined, zero_division=0),
            'recall': recall_score(labels, combined, zero_division=0),
            'accuracy': accuracy_score(labels, combined),
        }

    # 三通道加权融合（质量-稳定性-残差）
    if 'residual' in results:
        from sklearn.preprocessing import MinMaxScaler

        qq = 1 - MinMaxScaler().fit_transform(quality_scores.reshape(-1, 1)).ravel()
        ss = MinMaxScaler().fit_transform(stability_scores.reshape(-1, 1)).ravel()
        rr = MinMaxScaler().fit_transform(residual_scores.reshape(-1, 1)).ravel()

        best_combo = None
        for wq in np.linspace(0.4, 0.7, 7):
            for wr in np.linspace(0.2, 0.5, 7):
                ws = 1 - wq - wr
                if ws < args.min_stability_weight:
                    continue
                if ws <= 0:
                    continue
                fusion = wq * qq + ws * ss + wr * rr
                thresholds = np.linspace(fusion.min(), fusion.max(), 200)
                for th in thresholds:
                    preds = (fusion > th).astype(int)
                    f1 = f1_score(labels, preds, zero_division=0)
                    if (best_combo is None) or (f1 > best_combo['f1']):
                        best_combo = {
                            'weights': {
                                'quality': round(float(wq), 4),
                                'stability': round(float(ws), 4),
                                'residual': round(float(wr), 4),
                            },
                            'threshold': float(th),
                            'f1': float(f1),
                            'precision': float(precision_score(labels, preds, zero_division=0)),
                            'recall': float(recall_score(labels, preds, zero_division=0)),
                            'accuracy': float(accuracy_score(labels, preds))
                        }
        if best_combo:
            results['weighted_fusion_best'] = best_combo

    # 移除内部缓存的预测结果
    for key in ('quality', 'stability', 'residual'):
        if key in results and 'preds' in results[key]:
            results[key]['preds'] = None

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'[INFO] 评估完成，结果已保存至 {output_json}')


if __name__ == '__main__':
    main()


