"""
DVP涂层光谱异常检测系统 - 决策引擎模块
基于Quality Score和Stability Score的四格决策逻辑

Author: MiniMax Agent
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionResult(Enum):
    """决策结果枚举"""
    PASS = "pass"                    # 质量良好且稳定
    REWORK = "rework"               # 质量不佳但稳定
    REVIEW = "review"               # 质量良好但不稳定
    REJECT = "reject"               # 质量不佳且不稳定

@dataclass
class DecisionThresholds:
    """决策阈值配置"""
    quality_threshold: float = 0.8   # Quality Score阈值
    stability_threshold: float = 0.5 # Stability Score阈值
    
@dataclass
class DecisionResult:
    """决策结果数据类"""
    quality_score: float
    stability_score: float
    decision: DecisionResult
    confidence: float
    reasoning: str
    recommendations: List[str]
    
class DecisionEngine:
    """决策引擎类 - 基于Quality Score和Stability Score的四格决策逻辑"""
    
    def __init__(self, thresholds: Optional[DecisionThresholds] = None):
        """
        初始化决策引擎
        
        Args:
            thresholds: 决策阈值配置，如果为None则使用默认值
        """
        self.thresholds = thresholds or DecisionThresholds()
        self.decision_matrix = self._build_decision_matrix()
        
        logger.info(f"决策引擎初始化完成，阈值配置: {self.thresholds}")
    
    def _build_decision_matrix(self) -> Dict[str, Dict]:
        """构建四格决策矩阵"""
        return {
            "high_quality_high_stability": {
                "decision": DecisionResult.PASS,
                "description": "质量良好且稳定",
                "color": "green",
                "priority": 1
            },
            "low_quality_high_stability": {
                "decision": DecisionResult.REWORK,
                "description": "质量不佳但稳定",
                "color": "yellow", 
                "priority": 2
            },
            "high_quality_low_stability": {
                "decision": DecisionResult.REVIEW,
                "description": "质量良好但不稳定",
                "color": "orange",
                "priority": 3
            },
            "low_quality_low_stability": {
                "decision": DecisionResult.REJECT,
                "description": "质量不佳且不稳定",
                "color": "red",
                "priority": 4
            }
        }
    
    def make_decision(self, 
                     quality_score: float, 
                     stability_score: float,
                     coating_name: str = "DVP") -> DecisionResult:
        """
        基于Quality Score和Stability Score做出决策
        
        Args:
            quality_score: 质量评分 (0-1)
            stability_score: 稳定性评分 (0-1)
            coating_name: 涂层类型名称
            
        Returns:
            DecisionResult: 决策结果对象
        """
        try:
            # 输入验证
            if not (0 <= quality_score <= 1):
                raise ValueError(f"Quality Score必须在0-1范围内，当前值: {quality_score}")
            if not (0 <= stability_score <= 1):
                raise ValueError(f"Stability Score必须在0-1范围内，当前值: {stability_score}")
            
            # 确定决策象限
            quality_status = "high_quality" if quality_score >= self.thresholds.quality_threshold else "low_quality"
            stability_status = "high_stability" if stability_score >= self.thresholds.stability_threshold else "low_stability"
            
            decision_key = f"{quality_status}_{stability_status}"
            decision_info = self.decision_matrix[decision_key]
            
            # 计算置信度
            confidence = self._calculate_confidence(quality_score, stability_score)
            
            # 生成推理和建议
            reasoning = self._generate_reasoning(quality_score, stability_score, decision_info)
            recommendations = self._generate_recommendations(decision_info["decision"], quality_score, stability_score)
            
            logger.info(f"决策结果: {decision_info['description']}, 置信度: {confidence:.3f}")
            
            return DecisionResult(
                quality_score=quality_score,
                stability_score=stability_score,
                decision=decision_info["decision"],
                confidence=confidence,
                reasoning=reasoning,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"决策过程中发生错误: {str(e)}")
            raise
    
    def _calculate_confidence(self, quality_score: float, stability_score: float) -> float:
        """
        计算决策置信度
        
        Args:
            quality_score: 质量评分
            stability_score: 稳定性评分
            
        Returns:
            float: 置信度 (0-1)
        """
        # 距离阈值的距离作为置信度基础
        quality_distance = abs(quality_score - self.thresholds.quality_threshold)
        stability_distance = abs(stability_score - self.thresholds.stability_threshold)
        
        # 归一化距离并计算置信度
        max_distance = max(self.thresholds.quality_threshold, 1 - self.thresholds.quality_threshold,
                          self.thresholds.stability_threshold, 1 - self.thresholds.stability_threshold)
        
        confidence = 1 - (quality_distance + stability_distance) / (2 * max_distance)
        return max(0.1, min(1.0, confidence))  # 限制在0.1-1.0范围内
    
    def _generate_reasoning(self, quality_score: float, stability_score: float, decision_info: Dict) -> str:
        """
        生成决策推理文本
        
        Args:
            quality_score: 质量评分
            stability_score: 稳定性评分
            decision_info: 决策信息
            
        Returns:
            str: 推理文本
        """
        quality_status = "良好" if quality_score >= self.thresholds.quality_threshold else "不佳"
        stability_status = "稳定" if stability_score >= self.thresholds.stability_threshold else "不稳定"
        
        reasoning = f"基于质量评分{quality_score:.3f}({quality_status})和稳定性评分{stability_score:.3f}({stability_status})，"
        reasoning += f"系统判定为'{decision_info['description']}'。"
        
        if decision_info["decision"] == DecisionResult.PASS:
            reasoning += "该产品在质量控制标准范围内，可以直接通过。"
        elif decision_info["decision"] == DecisionResult.REWORK:
            reasoning += "虽然产品稳定，但质量指标未达标，需要重新加工处理。"
        elif decision_info["decision"] == DecisionResult.REVIEW:
            reasoning += "产品质量达标，但稳定性存在波动，建议人工复检。"
        else:  # REJECT
            reasoning += "产品质量和稳定性均未达标，建议报废处理。"
            
        return reasoning
    
    def _generate_recommendations(self, decision: DecisionResult, quality_score: float, stability_score: float) -> List[str]:
        """
        生成决策建议
        
        Args:
            decision: 决策结果
            quality_score: 质量评分
            stability_score: 稳定性评分
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        if decision == DecisionResult.PASS:
            recommendations.extend([
                "产品通过质量检测",
                "建议按正常流程入库或出货",
                "可作为标准样本用于后续对比"
            ])
        elif decision == DecisionResult.REWORK:
            recommendations.extend([
                "建议重新调整工艺参数",
                "检查原材料质量",
                "重新测量光谱数据确认结果",
                "调整质量控制标准"
            ])
        elif decision == DecisionResult.REVIEW:
            recommendations.extend([
                "建议人工复检确认结果",
                "检查测量环境是否稳定",
                "确认设备校准状态",
                "考虑重新测量"
            ])
        else:  # REJECT
            recommendations.extend([
                "产品不符合质量标准",
                "建议报废处理",
                "分析根本原因",
                "检查生产流程",
                "考虑更换原材料或调整工艺"
            ])
        
        return recommendations
    
    def batch_decision(self, 
                      results: List[Dict[str, float]], 
                      coating_name: str = "DVP") -> List[DecisionResult]:
        """
        批量决策处理
        
        Args:
            results: 包含quality_score和stability_score的字典列表
            coating_name: 涂层类型名称
            
        Returns:
            List[DecisionResult]: 决策结果列表
        """
        decisions = []
        for i, result in enumerate(results):
            try:
                decision = self.make_decision(
                    quality_score=result['quality_score'],
                    stability_score=result['stability_score'],
                    coating_name=coating_name
                )
                decisions.append(decision)
            except Exception as e:
                logger.error(f"第{i+1}个样本决策失败: {str(e)}")
                # 创建错误决策结果
                error_decision = DecisionResult(
                    quality_score=result.get('quality_score', 0),
                    stability_score=result.get('stability_score', 0),
                    decision=DecisionResult.REJECT,
                    confidence=0.0,
                    reasoning=f"决策失败: {str(e)}",
                    recommendations=["检查输入数据", "重新运行分析"]
                )
                decisions.append(error_decision)
        
        return decisions
    
    def get_decision_statistics(self, decisions: List[DecisionResult]) -> Dict:
        """
        获取决策统计信息
        
        Args:
            decisions: 决策结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not decisions:
            return {}
        
        # 统计各类决策数量
        decision_counts = {}
        for decision in decisions:
            decision_type = decision.decision.value
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
        
        # 计算平均置信度
        avg_confidence = np.mean([d.confidence for d in decisions])
        
        # 计算质量评分分布
        quality_scores = [d.quality_score for d in decisions]
        stability_scores = [d.stability_score for d in decisions]
        
        return {
            "total_samples": len(decisions),
            "decision_distribution": decision_counts,
            "average_confidence": avg_confidence,
            "quality_score_stats": {
                "mean": np.mean(quality_scores),
                "std": np.std(quality_scores),
                "min": np.min(quality_scores),
                "max": np.max(quality_scores)
            },
            "stability_score_stats": {
                "mean": np.mean(stability_scores),
                "std": np.std(stability_scores),
                "min": np.min(stability_scores),
                "max": np.max(stability_scores)
            }
        }
    
    def save_thresholds(self, filepath: Union[str, Path]):
        """
        保存阈值配置到文件
        
        Args:
            filepath: 保存路径
        """
        config = {
            "quality_threshold": self.thresholds.quality_threshold,
            "stability_threshold": self.thresholds.stability_threshold,
            "decision_matrix": self.decision_matrix
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"阈值配置已保存到: {filepath}")
    
    def load_thresholds(self, filepath: Union[str, Path]):
        """
        从文件加载阈值配置
        
        Args:
            filepath: 配置文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.thresholds = DecisionThresholds(
            quality_threshold=config["quality_threshold"],
            stability_threshold=config["stability_threshold"]
        )
        
        self.decision_matrix = config.get("decision_matrix", self.decision_matrix)
        logger.info(f"阈值配置已从文件加载: {filepath}")
    
    def visualize_decision_space(self, save_path: Optional[str] = None):
        """
        可视化决策空间
        
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 创建决策空间图
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 绘制阈值线
            ax.axhline(y=self.thresholds.stability_threshold, color='black', linestyle='--', alpha=0.7, label='稳定性阈值')
            ax.axvline(x=self.thresholds.quality_threshold, color='black', linestyle='--', alpha=0.7, label='质量阈值')
            
            # 绘制四个象限
            colors = ['green', 'yellow', 'orange', 'red']
            labels = ['通过(PASS)', '返工(REWORK)', '复检(REVIEW)', '报废(REJECT)']
            
            # 象限1: 高质量高稳定性
            rect1 = patches.Rectangle((self.thresholds.quality_threshold, self.thresholds.stability_threshold),
                                    1-self.thresholds.quality_threshold, 1-self.thresholds.stability_threshold,
                                    linewidth=2, edgecolor='green', facecolor='green', alpha=0.3)
            ax.add_patch(rect1)
            
            # 象限2: 低质量高稳定性
            rect2 = patches.Rectangle((0, self.thresholds.stability_threshold),
                                    self.thresholds.quality_threshold, 1-self.thresholds.stability_threshold,
                                    linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3)
            ax.add_patch(rect2)
            
            # 象限3: 高质量低稳定性
            rect3 = patches.Rectangle((self.thresholds.quality_threshold, 0),
                                    1-self.thresholds.quality_threshold, self.thresholds.stability_threshold,
                                    linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3)
            ax.add_patch(rect3)
            
            # 象限4: 低质量低稳定性
            rect4 = patches.Rectangle((0, 0),
                                    self.thresholds.quality_threshold, self.thresholds.stability_threshold,
                                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(rect4)
            
            # 添加标签
            ax.text(0.95, 0.95, '通过\n(PASS)', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.05, 0.95, '返工\n(REWORK)', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.95, 0.05, '复检\n(REVIEW)', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.05, 0.05, '报废\n(REJECT)', ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('质量评分 (Quality Score)', fontsize=12)
            ax.set_ylabel('稳定性评分 (Stability Score)', fontsize=12)
            ax.set_title('DVP涂层光谱异常检测决策空间', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"决策空间图已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法生成决策空间可视化")
        except Exception as e:
            logger.error(f"生成决策空间可视化时发生错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 创建决策引擎
    engine = DecisionEngine()
    
    # 示例决策
    test_cases = [
        {"quality_score": 0.9, "stability_score": 0.8},
        {"quality_score": 0.6, "stability_score": 0.7},
        {"quality_score": 0.85, "stability_score": 0.3},
        {"quality_score": 0.5, "stability_score": 0.2}
    ]
    
    print("=== DVP涂层光谱异常检测决策引擎测试 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        result = engine.make_decision(**case)
        print(f"测试案例 {i}:")
        print(f"  质量评分: {result.quality_score:.3f}")
        print(f"  稳定性评分: {result.stability_score:.3f}")
        print(f"  决策结果: {result.decision.value}")
        print(f"  置信度: {result.confidence:.3f}")
        print(f"  推理: {result.reasoning}")
        print(f"  建议: {', '.join(result.recommendations)}")
        print("-" * 50)
    
    # 批量决策测试
    batch_results = engine.batch_decision(test_cases)
    stats = engine.get_decision_statistics(batch_results)
    
    print("\n=== 批量决策统计 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"决策分布: {stats['decision_distribution']}")
    print(f"平均置信度: {stats['average_confidence']:.3f}")