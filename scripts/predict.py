#!/usr/bin/env python3
"""
批量推断脚本 - 读取CSV/NPZ并输出Quality/Stability/Two-Stage判定

用法示例：
python scripts/predict.py \
  --model-dir models/DVP/v1.10_p50_w560_release \
  --input-csv data/DVP_train_sample.csv \
  --output-csv output/predict_results.csv

python scripts/predict.py \
  --model-dir models/DVP/v1.10_p50_w560_release \
  --input-npz data/selection_v9/export/test_subset.npz \
  --output-json output/predict_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
import joblib

# 确保可以从项目根目录导入模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 项目内导入
from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def read_csv_spectra(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """兼容：列名为波长或首行波长两种格式。"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        numeric_cols = []
        numeric_wls = []
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

    # 回退：首行波长
    rows = pd.read_csv(csv_path, header=None).values
    wavelengths = rows[0].astype(float)
    spectra = rows[1:].astype(float)
    return wavelengths, spectra


def load_model_components(model_dir: Path):
    # 兼容通配命名（evaluate.py同策略）
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
    parser = argparse.ArgumentParser(description='批量推断：Quality/Stability/Two-Stage')
    parser.add_argument('--model-dir', required=True, help='模型目录（包含encoder/decoder/scaler/weights/metadata）')
    parser.add_argument('--input-csv', help='输入CSV（列名为波长或首行波长）')
    parser.add_argument('--input-npz', help='输入NPZ（包含 wavelengths, dvp_values）')
    parser.add_argument('--output-csv', help='输出CSV路径')
    parser.add_argument('--output-json', help='输出JSON路径')
    parser.add_argument('--quality-threshold', type=float, default=None,
                        help='覆盖质量阈值(0-1或百分比>1)；不提供则读取metadata')
    parser.add_argument('--stability-threshold', type=float, default=None,
                        help='覆盖稳定性阈值(重构误差阈值)；不提供则读取metadata，若阈值<=0则按批次95分位自适应')
    parser.add_argument('--quality-pctl', type=float, default=None,
                        help='按批次分位设置质量阈值(%)，如10表示用本批质量分数的P10作为阈值(更宽松)；与quality-threshold互斥')
    parser.add_argument('--stability-pctl', type=float, default=None,
                        help='按批次分位设置稳定性阈值(%)，如97.5表示用本批稳定性分数的P97.5作为阈值(更宽松)；与stability-threshold互斥')
    parser.add_argument('--residual-clf', type=str, default=None,
                        help='可选：已训练的残差分类器joblib路径（scripts/train_residual_classifier.py 产物）')
    parser.add_argument('--residual-fuse-mode', type=str, choices=['weighted','or'], default='weighted',
                        help='残差通道与AE误差融合方式（仅提供residual-clf时生效）')
    parser.add_argument('--residual-weight', type=float, default=0.5,
                        help='融合权重，weighted模式下越大越偏向AE误差')
    args = parser.parse_args()

    if not args.input_csv and not args.input_npz:
        raise SystemExit('必须提供 --input-csv 或 --input-npz 之一')

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

    # 对齐到标准网格
    if not np.array_equal(src_wl, std_wl):
        spectra = np.vstack([loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in spectra])

    # 阈值
    # 质量阈值
    if args.quality_pctl is not None and args.quality_threshold is not None:
        raise SystemExit('quality-pctl 与 quality-threshold 互斥，请只设置其一')
    if args.quality_threshold is not None:
        q_th = float(args.quality_threshold)
    else:
        q_th = metadata.get('similarity_evaluator', {}).get('quality_threshold', 92.0)
    if q_th > 1:
        q_th = q_th / 100.0
    # 稳定性阈值（原始重构误差阈值）
    if args.stability_threshold is not None:
        s_th = float(args.stability_threshold)
    else:
        s_th = metadata.get('weighted_autoencoder', {}).get('stability_threshold', None)

    results = []
    # 先计算全量稳定性分数，便于在阈值缺失/异常时做分位自适应
    stability_scores = []
    quality_scores = []
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

    # 若用户提供按批分位阈值，覆盖默认阈值
    if args.quality_pctl is not None:
        q_th = float(np.percentile(np.array(quality_scores), args.quality_pctl))
    # 若s_th缺失或为非正（可能来自方向翻转的评估流程），回退为批次95分位
    if (s_th is None) or (s_th <= 0):
        s_th = float(np.percentile(np.array(stability_scores), 95))
    if args.stability_pctl is not None:
        s_th = float(np.percentile(np.array(stability_scores), args.stability_pctl))

    # 准备融合（可选）
    fused_scores = None
    residual_proba = None
    if args.residual_clf:
        # 计算残差特征并给出分类器概率（越大越异常）
        clf = joblib.load(args.residual_clf)
        # 构造残差
        # 需要模型组件：已经在前面加载
        residuals = []
        for sp in spectra:
            sp_scaled = scaler.transform(sp.reshape(1, -1))
            z = encoder.predict(sp_scaled)
            rec = decoder.predict(z)
            sp_orig = scaler.inverse_transform(sp_scaled)[0]
            rec = rec[0]
            residuals.append(sp_orig - rec)
        residuals = np.array(residuals)
        # 分段特征
        def seg_feats(residuals_arr: np.ndarray, wl: np.ndarray) -> np.ndarray:
            bands = [(400, 480), (480, 560), (560, 680), (680, 780)]
            feats = []
            for lo, hi in bands:
                mask = (wl >= lo) & (wl <= hi)
                seg = residuals_arr[:, mask]
                abs_seg = np.abs(seg)
                feats.append(np.mean(abs_seg, axis=1))
                feats.append(np.sqrt(np.mean(seg ** 2, axis=1)))
                feats.append(np.max(abs_seg, axis=1))
            return np.column_stack(feats)
        X_res = seg_feats(residuals, std_wl)
        residual_proba = clf.predict_proba(X_res)[:, 1]
        # 归一化AE误差至[0,1]
        s_arr = np.array(stability_scores)
        eps = 1e-12
        s_anom = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + eps)
        if args.residual_fuse_mode == 'or':
            fused_scores = np.maximum(s_anom, residual_proba)
        else:
            w = float(args.residual_weight)
            fused_scores = w * s_anom + (1.0 - w) * residual_proba

    for idx, (q, s) in enumerate(zip(quality_scores, stability_scores)):
        # Quality
        # Stability（重构误差，越大越异常）已计算
        quality_anom = (q < q_th)
        stability_anom = (s > s_th) if s_th is not None else False
        combined_anom = True if quality_anom else bool(stability_anom)
        results.append({
            'index': int(idx),
            'quality_score': float(q),
            'stability_score': float(s),
            'quality_anom': bool(quality_anom),
            'stability_anom': bool(stability_anom),
            'combined_two_stage_anom': bool(combined_anom),
            **({'residual_proba': float(residual_proba[idx])} if residual_proba is not None else {}),
            **({'fused_score': float(fused_scores[idx]), 'fused_anom': bool(fused_scores[idx] > 0.5)} if fused_scores is not None else {}),
        })

    print(f"使用阈值: quality_threshold={q_th:.6f}, stability_threshold={s_th:.6f}")

    # 输出
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f'CSV已保存: {args.output_csv}')
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'JSON已保存: {args.output_json}')
    if not args.output_csv and not args.output_json:
        # 默认打印前若干条
        print(json.dumps(results[:5], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


