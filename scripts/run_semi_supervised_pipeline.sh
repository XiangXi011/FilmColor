#!/bin/bash
#=============================================================================
# DVP光谱异常检测 - 半监督学习一键执行脚本
#=============================================================================
# 
# 功能：自动化执行半监督学习的完整流程
#   Step 1: 生成候选样本（200个最不确定的样本）
#   Step 2: 创建标注模板
#   Step 3: 提示人工标注
#   Step 4: 训练残差分类器（需标注完成后再运行）
#   Step 5: 融合评估
#
# 使用方法：
#   chmod +x scripts/run_semi_supervised_pipeline.sh
#   ./scripts/run_semi_supervised_pipeline.sh
#
#=============================================================================

set -e  # 遇到错误立即退出

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================================================================="
echo "DVP光谱异常检测 - 半监督学习流水线"
echo "=================================================================================================="
echo ""
echo "📂 项目目录: $PROJECT_ROOT"
echo "🐍 Python版本: $(python --version)"
echo ""

# 检查环境
if ! python -c "import numpy" 2>/dev/null; then
    echo "❌ 错误: numpy未安装，请先激活正确的Python环境"
    echo "   提示: conda activate dvp_py310"
    exit 1
fi

echo "✅ Python环境检查通过"
echo ""

#=============================================================================
# Step 1: 生成候选样本
#=============================================================================
echo "=================================================================================================="
echo "Step 1: 生成候选样本（主动学习）"
echo "=================================================================================================="
echo ""
echo "🎯 目标: 选择200个最不确定的样本用于人工标注"
echo "📊 方法: 基于Quality和Stability的margin（到决策边界的距离）"
echo ""

LABEL_CSV="output/label_candidates_v1.12.csv"
SPECTRA_CSV="output/label_candidates_spectra_v1.12.csv"

if [ -f "$LABEL_CSV" ] && [ -f "$SPECTRA_CSV" ]; then
    echo "⚠️  检测到已存在的候选样本文件："
    echo "   - $LABEL_CSV"
    echo "   - $SPECTRA_CSV"
    read -p "是否重新生成？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏭️  跳过候选样本生成"
        goto_step_2=true
    else
        goto_step_2=false
    fi
else
    goto_step_2=false
fi

if [ "$goto_step_2" = false ]; then
    echo "🚀 开始生成候选样本..."
    python scripts/suggest_label_candidates.py \
        --model-dir models/DVP/v1.12_varsm_21k \
        --input-npz data/export_v11/test_subset.npz \
        --top-k 200 \
        --output-csv "$LABEL_CSV" \
        --export-spectra-csv "$SPECTRA_CSV"
    
    echo ""
    echo "✅ 候选样本生成完成！"
fi

echo ""
echo "📂 输出文件："
echo "   - $LABEL_CSV"
echo "   - $SPECTRA_CSV"
echo ""

#=============================================================================
# Step 2: 创建标注模板
#=============================================================================
echo "=================================================================================================="
echo "Step 2: 准备标注模板"
echo "=================================================================================================="
echo ""

LABELED_CSV="output/label_candidates_v1.12_labeled.csv"

if [ -f "$LABELED_CSV" ]; then
    echo "✅ 检测到已标注文件: $LABELED_CSV"
    echo ""
    read -p "是否使用该文件训练残差分类器？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        goto_step_3=true
    else
        echo ""
        echo "📋 请完成标注后再运行本脚本"
        goto_step_3=false
    fi
else
    echo "📋 标注文件尚未创建"
    echo ""
    echo "下一步操作："
    echo "  1. 打开文件: $SPECTRA_CSV"
    echo "  2. 在最后添加 'label' 列"
    echo "  3. 标注每一行: label=0（正常）或 label=1（异常）"
    echo "  4. 保存为: $LABELED_CSV"
    echo ""
    echo "标注指南："
    echo "  - Top 50样本（最不确定）: 必须标注"
    echo "  - 其余150样本: 选择性标注（建议至少标注100个）"
    echo "  - 异常判断标准:"
    echo "    * 峰值位置偏移 > 5nm"
    echo "    * 峰值强度异常 > 15%"
    echo "    * 曲线出现毛刺、断裂"
    echo "    * 存在明显噪声"
    echo ""
    echo "⏸️  流水线暂停，等待人工标注完成..."
    echo ""
    echo "标注完成后，请重新运行本脚本："
    echo "  ./scripts/run_semi_supervised_pipeline.sh"
    echo ""
    goto_step_3=false
fi

if [ "$goto_step_3" = false ]; then
    exit 0
fi

#=============================================================================
# Step 3: 训练残差分类器
#=============================================================================
echo ""
echo "=================================================================================================="
echo "Step 3: 训练残差分类器"
echo "=================================================================================================="
echo ""
echo "🎯 目标: 基于标注数据训练监督分类器"
echo "🧠 模型: Logistic回归 + 12维分段残差特征"
echo ""

RESIDUAL_CLF="models/DVP/v1.12_varsm_21k/residual_clf_DVP_v1.12_varsm_21k.joblib"

if [ -f "$RESIDUAL_CLF" ]; then
    echo "⚠️  检测到已存在的残差分类器:"
    echo "   $RESIDUAL_CLF"
    read -p "是否重新训练？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏭️  使用现有残差分类器"
        goto_step_4=true
    else
        goto_step_4=false
    fi
else
    goto_step_4=false
fi

if [ "$goto_step_4" = false ]; then
    echo "🚀 开始训练残差分类器..."
    python scripts/train_residual_classifier.py \
        --model-dir models/DVP/v1.12_varsm_21k \
        --input-csv "$LABELED_CSV" \
        --output-dir models/DVP/v1.12_varsm_21k \
        --test-split 0.2
    
    echo ""
    echo "✅ 残差分类器训练完成！"
fi

echo ""
echo "📂 输出文件："
echo "   - $RESIDUAL_CLF"
echo "   - models/DVP/v1.12_varsm_21k/residual_clf_training_report_v1.12.json"
echo ""

#=============================================================================
# Step 4: 三通道融合评估
#=============================================================================
echo "=================================================================================================="
echo "Step 4: 三通道融合评估"
echo "=================================================================================================="
echo ""
echo "🎯 目标: 验证半监督学习效果"
echo "🧪 融合: Quality + Stability + Residual Classifier"
echo ""

echo "🚀 开始融合评估..."
python scripts/evaluate.py \
    --model-dir models/DVP/v1.12_varsm_21k \
    --combine-strategy weighted \
    --use-residual-clf \
    --residual-fuse-mode weighted \
    --test-data-npz data/export_v11/test_subset.npz \
    --optimize-thresholds f1 \
    --samples 1000

echo ""
echo "✅ 融合评估完成！"
echo ""
echo "📂 查看结果："
echo "   - evaluation/performance_metrics.json"
echo "   - evaluation/evaluation_report.md"
echo "   - evaluation/*.png (9张可视化图表)"
echo ""

#=============================================================================
# Step 5: 性能对比
#=============================================================================
echo "=================================================================================================="
echo "Step 5: 性能对比报告"
echo "=================================================================================================="
echo ""

if [ -f "evaluation/performance_metrics.json" ]; then
    echo "📊 当前性能指标："
    echo ""
    python -c "
import json
with open('evaluation/performance_metrics.json', 'r') as f:
    metrics = json.load(f)
    
print('Quality Score:')
print(f'  - AUC: {metrics[\"quality_score\"][\"auc_roc\"]:.3f}')
print(f'  - F1:  {metrics[\"quality_score\"][\"f1_score\"]:.3f}')
print()
print('Stability Score:')
print(f'  - AUC: {metrics[\"stability_score\"][\"auc_roc\"]:.3f}')
print(f'  - F1:  {metrics[\"stability_score\"][\"f1_score\"]:.3f}')
print()
print('Combined Model:')
print(f'  - AUC:       {metrics[\"combined_model\"][\"auc_roc\"]:.3f}')
print(f'  - F1:        {metrics[\"combined_model\"][\"f1_score\"]:.3f}')
print(f'  - Precision: {metrics[\"combined_model\"][\"precision\"]:.3f}')
print(f'  - Recall:    {metrics[\"combined_model\"][\"recall\"]:.3f}')
print(f'  - Strategy:  {metrics[\"combined_model\"][\"combine_strategy\"]}')
"
    echo ""
else
    echo "⚠️  性能指标文件不存在"
fi

echo "=================================================================================================="
echo "🎉 半监督学习流水线执行完毕！"
echo "=================================================================================================="
echo ""
echo "📈 预期效果："
echo "   - Combined F1:       0.747 → 0.80-0.85"
echo "   - Combined Precision: 0.826 → 0.75-0.85"
echo "   - Combined Recall:    0.682 → 0.80-0.90"
echo ""
echo "✅ 如果F1 ≥ 0.80，说明半监督学习成功！"
echo "📦 可以将模型部署到生产环境"
echo ""

