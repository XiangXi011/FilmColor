#!/usr/bin/env python3
"""
产线健康度监控脚本（无二分类，仅评分与趋势）

输出：
- CSV: 包含 QHS/SHS/CHS 及 CHS_EWMA，LCL/UCL 参考线
- PNG: CHS_EWMA 控图与趋势图

示例：
python scripts/health_monitor.py \
  --model-dir models/DVP/v1.10_p50_w560_release \
  --input-csv data/line_stream.csv \
  --window 100 --alpha 0.2 \
  --output-csv output/line_health_timeseries.csv \
  --output-png output/line_health_chart.png
"""

import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目内导入
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from data.data_loader import SpectrumDataLoader
from algorithms.similarity_evaluator import SimilarityEvaluator


def read_csv_spectra(csv_path: str):
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
    ap = argparse.ArgumentParser(description='产线健康度监控（QHS/SHS/CHS与EWMA控图）')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--input-csv')
    ap.add_argument('--input-npz')
    ap.add_argument('--window', type=int, default=100, help='滚动窗口大小（用于SHS分位与基线统计）')
    ap.add_argument('--alpha', type=float, default=0.2, help='EWMA平滑系数')
    ap.add_argument('--wq', type=float, default=0.7, help='QHS权重')
    ap.add_argument('--ws', type=float, default=0.3, help='SHS权重')
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--output-png', required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_csv = Path(args.output_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.output_png); out_png.parent.mkdir(parents=True, exist_ok=True)

    loader = SpectrumDataLoader()
    std_wl, std_curve = loader.load_dvp_standard_curve()
    evaluator = SimilarityEvaluator('DVP')
    encoder, decoder, scaler, weights, _ = load_model_components(model_dir)

    if args.input_csv:
        src_wl, spectra = read_csv_spectra(args.input_csv)
    else:
        data = np.load(args.input_npz)
        src_wl = data['wavelengths']
        spectra = data['dvp_values']

    if not np.array_equal(src_wl, std_wl):
        spectra = np.vstack([loader.interpolate_to_standard_grid(src_wl, sp, std_wl) for sp in spectra])

    # 分数序列
    q_scores, s_scores = [], []
    for sp in spectra:
        q = evaluator.evaluate(sp, std_curve, std_wl, coating_name='DVP')['similarity_score']
        sp_scaled = scaler.transform(sp.reshape(1, -1))
        z = encoder.predict(sp_scaled)
        rec = decoder.predict(z)
        sp_orig = scaler.inverse_transform(sp_scaled)[0]
        rec = rec[0]
        s = float(np.mean(weights * (sp_orig - rec) ** 2)) if weights is not None else float(np.mean((sp_orig - rec) ** 2))
        q_scores.append(q); s_scores.append(s)
    q = pd.Series(q_scores)
    s = pd.Series(s_scores)

    # SHS: 基于滚动窗口的分位归一化 SHS=1-Pwindow(s)
    win = max(5, int(args.window))
    def rolling_percentile_last(x):
        # 返回序列最后一个值在窗口中的百分位（0-1）
        xr = pd.Series(x)
        return xr.rank(pct=True).iloc[-1]
    # 为避免样本数不足导致全NaN，采用 min_periods=1，并在首段窗口很小时允许估计
    s_pctl = s.rolling(win, min_periods=1).apply(rolling_percentile_last, raw=False)
    # 若仍有缺失，使用全局百分位填充
    s_pctl = s_pctl.fillna(s.rank(pct=True))
    SHS = (1.0 - s_pctl).clip(0.0, 1.0)
    QHS = q.clip(0, 1)
    CHS = args.wq * QHS + args.ws * SHS
    CHS_EWMA = CHS.ewm(alpha=args.alpha, adjust=False).mean()

    # 控制限基于首个窗口
    base = CHS_EWMA.iloc[:win].dropna()
    if len(base) >= 5:
        mu, sigma = base.mean(), base.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            LCL, UCL = mu - 0.1, mu + 0.1
        else:
            LCL, UCL = mu - 3*sigma, mu + 3*sigma
    else:
        mu = CHS_EWMA.mean(); sigma = CHS_EWMA.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            LCL, UCL = mu - 0.1, mu + 0.1
        else:
            LCL, UCL = mu - 3*sigma, mu + 3*sigma

    df = pd.DataFrame({
        'quality_score': q,
        'stability_score': s,
        'QHS': QHS,
        'SHS': SHS,
        'CHS': CHS,
        'CHS_EWMA': CHS_EWMA,
        'LCL': LCL,
        'UCL': UCL,
    })
    df.to_csv(out_csv, index=False)

    # 画图
    plt.figure(figsize=(12,6))
    plt.plot(CHS_EWMA.index, CHS_EWMA.values, label='CHS_EWMA', color='b')
    plt.axhline(LCL, color='r', linestyle='--', label=f'LCL={LCL:.3f}')
    plt.axhline(UCL, color='g', linestyle='--', label=f'UCL={UCL:.3f}')
    plt.ylim(0, 1)
    plt.xlabel('Sample Index')
    plt.ylabel('Health Score (0-1)')
    plt.title('Line Health EWMA Control Chart (CHS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'已输出: {out_csv}\n已生成图: {out_png}\nLCL={LCL:.3f}, UCL={UCL:.3f}')


if __name__ == '__main__':
    main()


