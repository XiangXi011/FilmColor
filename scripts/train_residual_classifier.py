#!/usr/bin/env python3
"""
训练“残差分段特征 + Logistic”辅助分类器

输入：
- 模型目录：包含 encoder/decoder/scaler/weights/metadata
- 数据：CSV（列名为波长或首行波长，需含 label 列：1=异常, 0=正常），或 NPZ（含 wavelengths, dvp_values, labels）

输出：
- 残差分类器 joblib
- 训练报告 JSON（AUC/F1 等）

示例：
python scripts/train_residual_classifier.py \
  --model-dir models/DVP/v1.10_p50_w560_release \
  --input-csv data/labeled_spectra.csv \
  --output-dir models/DVP/v1.10_p50_w560_release/residual_clf
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


def read_labeled_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'label' not in df.columns:
        raise ValueError('CSV必须包含 label 列（1=异常，0=正常）')
    labels = df['label'].astype(int).values
    # 尝试列名为波长
    numeric_cols, numeric_wls = [], []
    for col in df.columns:
        if col == 'label':
            continue
        if col.lower() in {'sample_idx', 'sample_id', 'id'}:
            continue
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
        return wavelengths, spectra, labels
    # 回退：首行波长格式不支持带label混排，这里直接报错
    raise ValueError('CSV需为“列名即波长”格式，并包含label列')


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
    return encoder, decoder, scaler, weights


def compute_residuals_matrix(spectra: np.ndarray, encoder, decoder, scaler) -> np.ndarray:
    residuals = []
    for sp in spectra:
        sp_scaled = scaler.transform(sp.reshape(1, -1))
        z = encoder.predict(sp_scaled)
        rec = decoder.predict(z)
        sp_orig = scaler.inverse_transform(sp_scaled)[0]
        rec = rec[0]
        residuals.append(sp_orig - rec)
    return np.array(residuals)


def extract_segment_features(residuals: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    bands = [(400, 480), (480, 560), (560, 680), (680, 780)]
    feats = []
    for lo, hi in bands:
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        seg = residuals[:, mask]
        abs_seg = np.abs(seg)
        mean_abs = np.mean(abs_seg, axis=1)
        rms = np.sqrt(np.mean(seg ** 2, axis=1))
        max_abs = np.max(abs_seg, axis=1)
        feats.append(mean_abs)
        feats.append(rms)
        feats.append(max_abs)
    return np.column_stack(feats)


def main():
    parser = argparse.ArgumentParser(description='训练残差分段特征Logistic分类器')
    parser.add_argument('--model-dir', required=True, help='模型目录')
    parser.add_argument('--input-csv', help='带label的CSV（列名为波长，含label列）')
    parser.add_argument('--input-npz', help='带labels的NPZ（wavelengths, dvp_values, labels）')
    parser.add_argument('--output-dir', required=True, help='输出目录（保存分类器与报告）')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SpectrumDataLoader()
    std_wl, _ = loader.load_dvp_standard_curve()

    # 读数据
    if args.input_csv:
        src_wl, spectra, labels = read_labeled_csv(args.input_csv)
    elif args.input_npz:
        data = np.load(args.input_npz)
        src_wl = data['wavelengths']
        spectra = data['dvp_values']
        labels = data['labels']
    else:
        raise SystemExit('必须提供 --input-csv 或 --input-npz')

    # 对齐
    if not np.array_equal(src_wl, std_wl):
        spectra = np.vstack([loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in spectra])

    # 模型组件
    encoder, decoder, scaler, _ = load_model_components(model_dir)

    # 残差与特征
    residuals = compute_residuals_matrix(spectra, encoder, decoder, scaler)
    X = extract_segment_features(residuals, std_wl)
    y = labels.astype(int)

    # 训练
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)
    report = {
        'auc_roc': float(roc_auc_score(y_te, proba)),
        'f1': float(f1_score(y_te, pred, zero_division=0)),
        'precision': float(precision_score(y_te, pred, zero_division=0)),
        'recall': float(recall_score(y_te, pred, zero_division=0)),
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te))
    }

    # 保存
    clf_path = out_dir / 'residual_logistic.joblib'
    joblib.dump(clf, clf_path)
    with open(out_dir / 'training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('训练完成:')
    print('  分类器:', clf_path)
    print('  报告:', out_dir / 'training_report.json')
    print('  指标:', report)


if __name__ == '__main__':
    main()


