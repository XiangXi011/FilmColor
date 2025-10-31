#!/usr/bin/env python3
"""
一键流水线：筛选 → 导出(train/val/test) → 训练 → 评估

示例：
python scripts/pipeline_oneclick.py \
  --input-csv data/DVP_train_sample.csv \
  --work-dir data/selection_v4 \
  --export-dir data/selection_v4/export \
  --model-version v1.6 \
  --optimize-thresholds youden
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description='一键运行筛选→导出→训练→评估流水线')
    parser.add_argument('--input-csv', required=True, help='原始CSV（含光谱）')
    parser.add_argument('--work-dir', required=True, help='筛选输出目录，保存索引与摘要')
    parser.add_argument('--export-dir', required=True, help='导出子集输出目录（CSV+NPZ）')
    parser.add_argument('--exclude-index', nargs='*', default=None, help='历史索引文件用于剔重')
    parser.add_argument('--edge-max-ratio', type=float, default=0.2, help='边缘占比(相对高置信)')
    parser.add_argument('--edge-max-ratio-total', type=float, default=0.1, help='边缘占比(相对总量)')
    parser.add_argument('--edge-max-abs', type=int, default=5000, help='边缘绝对上限')
    parser.add_argument('--sim-min-pctl', type=float, default=35.0)
    parser.add_argument('--pearson-min-pctl', type=float, default=35.0)
    parser.add_argument('--high-sim-pctl', type=float, default=85.0)
    parser.add_argument('--high-pearson-pctl', type=float, default=90.0)
    parser.add_argument('--edge-sort-by', choices=['similarity','pearson','mix'], default='mix')

    parser.add_argument('--model-version', type=str, default='v1.6', help='模型版本用于保存路径')
    parser.add_argument('--std-agg', choices=['median','mean'], default='median', help='标准曲线聚合')
    parser.add_argument('--optimize-thresholds', choices=['none','youden','f1'], default='f1', help='评估阈值优化')
    parser.add_argument('--random-seed', type=int, default=42)

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1) 筛选
    cmd_select = [
        sys.executable, 'scripts/select_training_data.py',
        '--input', args.input_csv,
        '--output-dir', str(work_dir),
        '--edge-max-ratio', str(args.edge_max_ratio),
        '--edge-max-ratio-total', str(args.edge_max_ratio_total),
        '--edge-max-abs', str(args.edge_max_abs),
        '--sim-min-pctl', str(args.sim_min_pctl),
        '--pearson-min-pctl', str(args.pearson_min_pctl),
        '--high-sim-pctl', str(args.high_sim_pctl),
        '--high-pearson-pctl', str(args.high_pearson_pctl),
        '--edge-sort-by', args.edge_sort_by,
    ]
    if args.exclude_index:
        cmd_select += ['--exclude-index', *args.exclude_index]
    run(cmd_select)

    # 2) 导出 train/val/test 三套子集（CSV+NPZ）
    cmd_export = [
        sys.executable, 'scripts/export_training_subset.py',
        '--input', args.input_csv,
        '--index-dir', str(work_dir),
        '--output-dir', str(export_dir),
    ]
    run(cmd_export)

    # 3) 训练（使用导出的NPZ作为训练数据源，聚合标准曲线）
    # 使用 train/val/test 中的 train_subset.npz 作为样本矩阵来源，训练脚本会对矩阵进行聚合得到标准曲线
    train_npz = export_dir / 'train_subset.npz'
    cmd_train = [
        sys.executable, 'scripts/train.py',
        '--coating_name', 'DVP',
        '--version', args.model_version,
        '--data-npz', str(train_npz),
        '--std_curve_agg', args.std_agg,
    ]
    run(cmd_train)

    # 4) 评估
    cmd_eval = [
        sys.executable, 'scripts/evaluate.py',
        '--model-dir', 'models',
        '--samples', '1000',
        '--optimize-thresholds', args.optimize_thresholds,
        '--random-seed', str(args.random_seed),
    ]
    run(cmd_eval)

    print('\n✅ 流水线完成。输出:')
    print(' - 筛选:', work_dir)
    print(' - 导出:', export_dir)
    print(' - 模型: models/DVP/', args.model_version)
    print(' - 评估: evaluation/')


if __name__ == '__main__':
    main()


