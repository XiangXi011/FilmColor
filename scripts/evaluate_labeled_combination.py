#!/usr/bin/env python3
"""基于人工标注样本评估不同组合策略"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# 项目内部模块
PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_loader import SpectrumDataLoader  # type: ignore  # noqa: E402
from algorithms.similarity_evaluator import SimilarityEvaluator  # type: ignore  # noqa: E402


def compute_quality_scores(spectra: np.ndarray, evaluator: SimilarityEvaluator, wavelengths: np.ndarray) -> np.ndarray:
    scores = []
    for sp in spectra:
        res = evaluator.evaluate(sp, standard_spectrum=None, wavelengths=wavelengths, coating_name='DVP')
        scores.append(res['similarity_score'])
    return np.array(scores)


def compute_stability_scores(spectra: np.ndarray, encoder, decoder, scaler, weights: np.ndarray) -> np.ndarray:
    scores = []
    for sp in spectra:
        scaled = scaler.transform(sp.reshape(1, -1))
        latent = encoder.predict(scaled)
        recon = decoder.predict(latent)
        original = scaler.inverse_transform(scaled)
        err = np.mean(weights * (original - recon) ** 2)
        scores.append(float(err))
    return np.array(scores)


def normalize(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def evaluate_threshold_strategy(labels, score, threshold, greater_is_anomaly=True):
    if greater_is_anomaly:
        preds = (score >= threshold).astype(int)
    else:
        preds = (score <= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0, average='binary')
    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'accuracy': float(accuracy_score(labels, preds)),
        'threshold': float(threshold)
    }


def main():
    parser = argparse.ArgumentParser(description='使用人工标注数据评估组合策略')
    parser.add_argument('--model-dir', type=str,
                        default=str(PROJECT_ROOT / 'models' / 'DVP' / 'v1.12_varsm_21k'),
                        help='模型目录')
    parser.add_argument('--labeled-csv', type=str,
                        default=str(PROJECT_ROOT / 'evaluation' / 'label' / 'combined_labeled_spectra.csv'),
                        help='包含label列的光谱CSV')
    parser.add_argument('--residual-clf', type=str,
                        default=None,
                        help='残差分类器路径（若为空则尝试自动加载）')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    labeled_path = Path(args.labeled_csv)
    if not labeled_path.exists():
        raise FileNotFoundError(f'未找到标注数据: {labeled_path}')

    df = pd.read_csv(labeled_path)
    labels = df['label'].astype(int).values
    feature_cols = [c for c in df.columns if c not in ('label', 'sample_idx')]
    spectra = df[feature_cols].to_numpy(dtype=float)

    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()

    # 对齐波长（列已经是标准波长）
    evaluator = SimilarityEvaluator('DVP')
    quality_scores = []
    for sp in spectra:
        res = evaluator.evaluate(sp, std_curve, std_wl, coating_name='DVP')
        quality_scores.append(res['similarity_score'])
    quality_scores = np.array(quality_scores)

    def _find(patterns):
        for pattern in patterns:
            matches = sorted(model_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    encoder = joblib.load(_find(['*encoder*DVP*v*.joblib', '*encoder*DVP*.joblib', '*encoder*.joblib']))
    decoder = joblib.load(_find(['*decoder*DVP*v*.joblib', '*decoder*DVP*.joblib', '*decoder*.joblib']))
    scaler = joblib.load(_find(['*scaler*DVP*v*.joblib', '*scaler*DVP*.joblib', '*scaler*.joblib']))
    weights_path = _find(['weights*DVP*v*.npy', 'weights*DVP*.npy', 'weights*.npy'])
    if weights_path:
        weights = np.load(weights_path)
    else:
        weights = np.ones(len(std_wl))

    stability_scores = compute_stability_scores(spectra, encoder, decoder, scaler, weights)

    residual_path = Path(args.residual_clf) if args.residual_clf else _find(['*residual_clf*.joblib'])
    if residual_path is None:
        raise FileNotFoundError('未找到残差分类器，请使用 --residual-clf 指定路径')
    residual_clf = joblib.load(residual_path)
    # 构造残差特征
    residuals = []
    for sp in spectra:
        scaled = scaler.transform(sp.reshape(1, -1))
        latent = encoder.predict(scaled)
        recon = decoder.predict(latent)
        original = scaler.inverse_transform(scaled)
        residuals.append(original[0] - recon[0])
    residuals = np.array(residuals)

    bands = [(400, 480), (480, 560), (560, 680), (680, 780)]
    residual_feats = []
    for lo, hi in bands:
        mask = (std_wl >= lo) & (std_wl <= hi)
        seg = residuals[:, mask]
        abs_seg = np.abs(seg)
        residual_feats.append(abs_seg.mean(axis=1))
        residual_feats.append(np.sqrt((seg ** 2).mean(axis=1)))
        residual_feats.append(abs_seg.max(axis=1))
    residual_feats = np.column_stack(residual_feats)
    residual_probs = residual_clf.predict_proba(residual_feats)[:, 1]

    # 评价策略
    results = {}
    # Quality-only (阈值0.8)
    results['quality_only@0.80'] = evaluate_threshold_strategy(labels, quality_scores, 0.80, greater_is_anomaly=False)
    # 使用F1最佳阈值
    q_sorted = np.linspace(quality_scores.min(), quality_scores.max(), 200)
    best_f1, best_th = -1.0, q_sorted[0]
    for th in q_sorted:
        metric = evaluate_threshold_strategy(labels, quality_scores, th, greater_is_anomaly=False)
        if metric['f1'] > best_f1:
            best_f1, best_th = metric['f1'], th
            best_metric = metric
    results['quality_only_best'] = best_metric

    # Stability-only (F1优化)
    s_sorted = np.linspace(stability_scores.min(), stability_scores.max(), 200)
    best_f1, best_th = -1.0, s_sorted[0]
    best_metric = None
    for th in s_sorted:
        metric = evaluate_threshold_strategy(labels, stability_scores, th, greater_is_anomaly=True)
        if metric['f1'] > best_f1:
            best_f1, best_th = metric['f1'], th
            best_metric = metric
    results['stability_only_best'] = best_metric

    # Residual-only (默认0.5 & F1优化)
    results['residual_only@0.5'] = evaluate_threshold_strategy(labels, residual_probs, 0.5, greater_is_anomaly=True)
    r_sorted = np.linspace(residual_probs.min(), residual_probs.max(), 200)
    best_f1, best_metric = -1.0, None
    for th in r_sorted:
        metric = evaluate_threshold_strategy(labels, residual_probs, th, greater_is_anomaly=True)
        if metric['f1'] > best_f1:
            best_f1, best_metric = metric['f1'], metric
    results['residual_only_best'] = best_metric

    # 组合策略：Quality OR Residual（阈值可优化）
    best_f1, best_pair, best_metric = -1.0, (0.8, 0.5), None
    q_grid = np.linspace(quality_scores.min(), quality_scores.max(), 100)
    r_grid = np.linspace(residual_probs.min(), residual_probs.max(), 100)
    for q_th in q_grid:
        q_flag = (quality_scores <= q_th).astype(int)
        for r_th in r_grid:
            r_flag = (residual_probs >= r_th).astype(int)
            preds = ((q_flag == 1) | (r_flag == 1)).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_pair = (q_th, r_th)
                best_metric = {
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1': float(f1),
                    'accuracy': float(accuracy_score(labels, preds)),
                    'quality_threshold': float(q_th),
                    'residual_threshold': float(r_th)
                }
    results['quality_or_residual'] = best_metric

    # 三特征逻辑回归（五折交叉验证）
    features = np.column_stack([quality_scores, stability_scores, residual_probs])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_cv = []
    for train_idx, test_idx in skf.split(features, labels):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(features[train_idx], labels[train_idx])
        preds = clf.predict(features[test_idx])
        prob = clf.predict_proba(features[test_idx])[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(labels[test_idx], preds, zero_division=0, average='binary')
        metrics_cv.append({
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': float(accuracy_score(labels[test_idx], preds)),
            'auc': float(roc_auc_score(labels[test_idx], prob))
        })
    results['logistic_cv'] = {
        'precision': float(np.mean([m['precision'] for m in metrics_cv])),
        'recall': float(np.mean([m['recall'] for m in metrics_cv])),
        'f1': float(np.mean([m['f1'] for m in metrics_cv])),
        'accuracy': float(np.mean([m['accuracy'] for m in metrics_cv])),
        'auc': float(np.mean([m['auc'] for m in metrics_cv]))
    }

    output_dir = labeled_path.parent
    output_file = output_dir / 'combination_metrics.json'
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"指标已保存: {output_file}")

    for name, metric in results.items():
        print(f"\n{name} ->")
        for k, v in metric.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()


