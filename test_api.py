"""
DVP涂层光谱异常检测系统 - API测试脚本
测试所有API端点的功能

Author: MiniMax Agent
Date: 2025-10-30
"""

import requests
import numpy as np
import pandas as pd
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITester:
    """API测试器类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化API测试器
        
        Args:
            base_url: API服务器基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # 测试结果存储
        self.test_results = []
        
        logger.info(f"API测试器初始化完成，目标服务器: {self.base_url}")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行API测试套件...")
        
        test_methods = [
            self.test_health_check,
            self.test_root_endpoint,
            self.test_single_spectrum_analysis,
            self.test_batch_spectrum_analysis,
            self.test_cache_stats,
            self.test_cache_models,
            self.test_decision_thresholds,
            self.test_decision_visualization
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"运行测试: {test_method.__name__}")
                result = test_method()
                self.test_results.append({
                    'test_name': test_method.__name__,
                    'status': 'PASS' if result else 'FAIL',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                logger.info(f"测试 {test_method.__name__} {'通过' if result else '失败'}")
            except Exception as e:
                logger.error(f"测试 {test_method.__name__} 发生异常: {str(e)}")
                self.test_results.append({
                    'test_name': test_method.__name__,
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        self.print_test_summary()
    
    def test_health_check(self) -> bool:
        """测试健康检查端点"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                logger.error(f"健康检查失败: HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # 验证响应结构
            required_fields = ['status', 'timestamp', 'version', 'components']
            if not all(field in data for field in required_fields):
                logger.error("健康检查响应缺少必要字段")
                return False
            
            # 检查组件状态
            components = data.get('components', {})
            if not components:
                logger.error("健康检查响应中组件信息为空")
                return False
            
            logger.info(f"健康检查成功: {data['status']}")
            logger.info(f"组件状态: {components}")
            
            return True
            
        except Exception as e:
            logger.error(f"健康检查测试失败: {str(e)}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """测试根端点"""
        try:
            response = self.session.get(f"{self.base_url}/")
            
            if response.status_code != 200:
                logger.error(f"根端点测试失败: HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # 验证响应内容
            if 'service' not in data or 'DVP涂层光谱异常检测API' not in data['service']:
                logger.error("根端点响应内容不正确")
                return False
            
            logger.info(f"根端点测试成功: {data['service']}")
            return True
            
        except Exception as e:
            logger.error(f"根端点测试失败: {str(e)}")
            return False
    
    def generate_test_spectrum(self, num_points: int = 81) -> Dict[str, List[float]]:
        """
        生成测试光谱数据
        
        Args:
            num_points: 数据点数量
            
        Returns:
            Dict: 包含wavelengths和spectrum的字典
        """
        # 生成波长范围 (380-780nm)
        wavelengths = np.linspace(380, 780, num_points).tolist()
        
        # 生成模拟光谱数据 (基于DVP标准光谱的变体)
        np.random.seed(42)  # 确保可重复性
        
        # 基础光谱形状
        base_spectrum = np.exp(-((np.array(wavelengths) - 550) / 100) ** 2)
        
        # 添加噪声和变化
        noise = np.random.normal(0, 0.05, num_points)
        spectrum = base_spectrum + noise
        
        # 确保所有值为正
        spectrum = np.maximum(spectrum, 0.01)
        
        return {
            'wavelengths': wavelengths,
            'spectrum': spectrum.tolist()
        }
    
    def test_single_spectrum_analysis(self) -> bool:
        """测试单个光谱分析"""
        try:
            # 生成测试数据
            test_data = self.generate_test_spectrum()
            
            # 构建请求
            request_data = {
                'wavelengths': test_data['wavelengths'],
                'spectrum': test_data['spectrum'],
                'coating_name': 'DVP'
            }
            
            logger.info("发送单个光谱分析请求...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=request_data
            )
            
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"单个光谱分析失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False
            
            data = response.json()
            
            # 验证响应结构
            required_fields = [
                'quality_score', 'stability_score', 'decision', 
                'confidence', 'reasoning', 'recommendations',
                'processing_time', 'timestamp'
            ]
            
            if not all(field in data for field in required_fields):
                logger.error("单个光谱分析响应缺少必要字段")
                return False
            
            # 验证数据范围
            if not (0 <= data['quality_score'] <= 1):
                logger.error(f"质量评分超出范围: {data['quality_score']}")
                return False
            
            if not (0 <= data['stability_score'] <= 1):
                logger.error(f"稳定性评分超出范围: {data['stability_score']}")
                return False
            
            if not (0 <= data['confidence'] <= 1):
                logger.error(f"置信度超出范围: {data['confidence']}")
                return False
            
            # 验证决策结果
            valid_decisions = ['pass', 'rework', 'review', 'reject']
            if data['decision'] not in valid_decisions:
                logger.error(f"无效的决策结果: {data['decision']}")
                return False
            
            logger.info(f"单个光谱分析成功:")
            logger.info(f"  质量评分: {data['quality_score']:.3f}")
            logger.info(f"  稳定性评分: {data['stability_score']:.3f}")
            logger.info(f"  决策: {data['decision']}")
            logger.info(f"  置信度: {data['confidence']:.3f}")
            logger.info(f"  处理时间: {data['processing_time']:.3f}s")
            logger.info(f"  响应时间: {response_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"单个光谱分析测试失败: {str(e)}")
            return False
    
    def test_batch_spectrum_analysis(self) -> bool:
        """测试批量光谱分析"""
        try:
            # 生成多个测试光谱
            num_spectra = 5
            spectra = []
            
            for i in range(num_spectra):
                test_data = self.generate_test_spectrum()
                spectra.append({
                    'wavelengths': test_data['wavelengths'],
                    'spectrum': test_data['spectrum'],
                    'coating_name': 'DVP'
                })
            
            # 构建批量请求
            request_data = {
                'spectra': spectra,
                'coating_name': 'DVP'
            }
            
            logger.info(f"发送批量光谱分析请求 (共{num_spectra}个光谱)...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/analyze/batch",
                json=request_data
            )
            
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"批量光谱分析失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False
            
            data = response.json()
            
            # 验证响应结构
            required_fields = [
                'results', 'total_processing_time', 'average_processing_time',
                'decision_summary', 'timestamp'
            ]
            
            if not all(field in data for field in required_fields):
                logger.error("批量光谱分析响应缺少必要字段")
                return False
            
            results = data['results']
            
            if len(results) != num_spectra:
                logger.error(f"结果数量不匹配: 期望{num_spectra}, 实际{len(results)}")
                return False
            
            # 验证每个结果
            for i, result in enumerate(results):
                result_fields = [
                    'quality_score', 'stability_score', 'decision',
                    'confidence', 'reasoning', 'recommendations'
                ]
                
                if not all(field in result for field in result_fields):
                    logger.error(f"第{i+1}个结果缺少必要字段")
                    return False
            
            # 验证决策统计
            decision_summary = data['decision_summary']
            if sum(decision_summary.values()) != num_spectra:
                logger.error("决策统计数量不匹配")
                return False
            
            logger.info(f"批量光谱分析成功:")
            logger.info(f"  处理光谱数量: {len(results)}")
            logger.info(f"  总处理时间: {data['total_processing_time']:.3f}s")
            logger.info(f"  平均处理时间: {data['average_processing_time']:.3f}s")
            logger.info(f"  响应时间: {response_time:.3f}s")
            logger.info(f"  决策分布: {decision_summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"批量光谱分析测试失败: {str(e)}")
            return False
    
    def test_cache_stats(self) -> bool:
        """测试缓存统计"""
        try:
            response = self.session.get(f"{self.base_url}/cache/stats")
            
            if response.status_code != 200:
                logger.error(f"缓存统计失败: HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # 验证响应结构
            required_fields = [
                'cached_models', 'max_cache_size', 'total_accesses',
                'average_load_time', 'cache_utilization'
            ]
            
            if not all(field in data for field in required_fields):
                logger.error("缓存统计响应缺少必要字段")
                return False
            
            # 验证数据范围
            if data['cached_models'] < 0:
                logger.error("缓存模型数量不能为负数")
                return False
            
            if not (0 <= data['cache_utilization'] <= 1):
                logger.error("缓存利用率超出范围")
                return False
            
            logger.info(f"缓存统计成功:")
            logger.info(f"  缓存模型数: {data['cached_models']}")
            logger.info(f"  最大缓存大小: {data['max_cache_size']}")
            logger.info(f"  总访问次数: {data['total_accesses']}")
            logger.info(f"  平均加载时间: {data['average_load_time']:.3f}s")
            logger.info(f"  缓存利用率: {data['cache_utilization']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"缓存统计测试失败: {str(e)}")
            return False
    
    def test_cache_models(self) -> bool:
        """测试缓存模型列表"""
        try:
            response = self.session.get(f"{self.base_url}/cache/models")
            
            if response.status_code != 200:
                logger.error(f"缓存模型列表失败: HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # 验证响应类型
            if not isinstance(data, dict):
                logger.error("缓存模型列表响应类型错误")
                return False
            
            logger.info(f"缓存模型列表成功: 共{len(data)}个模型")
            
            for model_name, model_info in data.items():
                logger.info(f"  模型: {model_name}")
                logger.info(f"    版本: {model_info.get('version', 'N/A')}")
                logger.info(f"    类型: {model_info.get('model_type', 'N/A')}")
                logger.info(f"    访问次数: {model_info.get('access_count', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"缓存模型列表测试失败: {str(e)}")
            return False
    
    def test_decision_thresholds(self) -> bool:
        """测试决策阈值"""
        try:
            # 获取当前阈值
            response = self.session.get(f"{self.base_url}/decision/thresholds")
            
            if response.status_code != 200:
                logger.error(f"获取决策阈值失败: HTTP {response.status_code}")
                return False
            
            current_thresholds = response.json()
            
            logger.info(f"当前决策阈值:")
            logger.info(f"  质量阈值: {current_thresholds['quality_threshold']}")
            logger.info(f"  稳定性阈值: {current_thresholds['stability_threshold']}")
            
            # 测试更新阈值
            new_thresholds = {
                'quality_threshold': 0.85,
                'stability_threshold': 0.6
            }
            
            response = self.session.post(
                f"{self.base_url}/decision/thresholds",
                json=new_thresholds
            )
            
            if response.status_code != 200:
                logger.error(f"更新决策阈值失败: HTTP {response.status_code}")
                return False
            
            # 验证更新结果
            response = self.session.get(f"{self.base_url}/decision/thresholds")
            updated_thresholds = response.json()
            
            if (updated_thresholds['quality_threshold'] != new_thresholds['quality_threshold'] or
                updated_thresholds['stability_threshold'] != new_thresholds['stability_threshold']):
                logger.error("决策阈值更新失败")
                return False
            
            logger.info("决策阈值测试成功")
            
            # 恢复原阈值
            self.session.post(
                f"{self.base_url}/decision/thresholds",
                json=current_thresholds
            )
            
            return True
            
        except Exception as e:
            logger.error(f"决策阈值测试失败: {str(e)}")
            return False
    
    def test_decision_visualization(self) -> bool:
        """测试决策空间可视化"""
        try:
            response = self.session.get(f"{self.base_url}/decision/visualize")
            
            if response.status_code != 200:
                logger.error(f"决策空间可视化失败: HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            # 验证响应
            if 'file_path' not in data:
                logger.error("可视化响应缺少文件路径")
                return False
            
            file_path = data['file_path']
            
            # 检查文件是否生成
            if not Path(file_path).exists():
                logger.error(f"可视化文件未生成: {file_path}")
                return False
            
            logger.info(f"决策空间可视化成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"决策空间可视化测试失败: {str(e)}")
            return False
    
    def print_test_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "="*50)
        logger.info("API测试摘要")
        logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        error_tests = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过: {passed_tests}")
        logger.info(f"失败: {failed_tests}")
        logger.info(f"错误: {error_tests}")
        logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        # 详细结果
        logger.info("\n详细结果:")
        for result in self.test_results:
            status_symbol = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "!"
            logger.info(f"  {status_symbol} {result['test_name']}: {result['status']}")
            if result['status'] == 'ERROR':
                logger.info(f"    错误: {result.get('error', 'Unknown error')}")
        
        # 保存测试报告
        self.save_test_report()
    
    def save_test_report(self):
        """保存测试报告"""
        report_data = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed': sum(1 for r in self.test_results if r['status'] == 'PASS'),
                'failed': sum(1 for r in self.test_results if r['status'] == 'FAIL'),
                'errors': sum(1 for r in self.test_results if r['status'] == 'ERROR'),
                'success_rate': sum(1 for r in self.test_results if r['status'] == 'PASS') / len(self.test_results) * 100
            },
            'test_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_server': self.base_url
        }
        
        report_file = f"test_results/api_test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        Path("test_results").mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试报告已保存: {report_file}")

def main():
    """主函数"""
    # 创建测试结果目录
    Path("test_results").mkdir(exist_ok=True)
    
    # 检查API服务器是否运行
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("API服务器未正常运行，请先启动服务器")
            return
    except requests.exceptions.RequestException:
        logger.error("无法连接到API服务器 (http://localhost:8000)")
        logger.error("请确保API服务器正在运行: python api_server.py")
        return
    
    # 运行测试
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()