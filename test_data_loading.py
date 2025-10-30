#!/usr/bin/env python3
"""
简化的数据加载测试脚本
"""

import sys
import os
import numpy as np
import pandas as pd

# 直接导入模块
sys.path.append('/workspace/code/spectrum_anomaly_detection')

def test_data_loading():
    """测试数据加载功能"""
    print("=" * 60)
    print("测试 DVP 数据加载功能")
    print("=" * 60)
    
    # 直接读取CSV文件
    csv_path = "/workspace/user_input_files/HunterLab DVP.csv"
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        print(f"✓ 成功读取CSV文件: {csv_path}")
        print(f"✓ 数据形状: {df.shape}")
        print(f"✓ 列名: {list(df.columns[:10])}...")  # 只显示前10列
        
        # 获取波长列（第一行是波长）
        wavelengths = df.columns[1:].astype(float)  # 跳过第一列（空列）
        
        # 获取DVP数据（第二行）
        dvp_values = df.iloc[0, 1:].astype(float)  # 跳过第一列
        
        print(f"✓ 波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
        print(f"✓ 波长点数: {len(wavelengths)}")
        print(f"✓ DVP反射率范围: {dvp_values.min():.4f} - {dvp_values.max():.4f}")
        print(f"✓ DVP均值: {dvp_values.mean():.4f}")
        print(f"✓ DVP标准差: {dvp_values.std():.4f}")
        
        # 检查数据质量
        print(f"✓ 包含NaN值: {dvp_values.isna().any()}")
        print(f"✓ 包含负值: {(dvp_values < 0).any()}")
        
        # 生成标准波长网格
        standard_wavelengths = np.arange(380, 781, 5)
        print(f"✓ 标准波长网格: {len(standard_wavelengths)} 个点 (380-780nm, 步长5nm)")
        
        # 检查是否需要插值
        if not np.array_equal(wavelengths, standard_wavelengths):
            print("⚠️  需要插值处理...")
            interpolated_values = np.interp(standard_wavelengths, wavelengths, dvp_values)
            print(f"✓ 插值完成: {len(interpolated_values)} 个数据点")
            
            # 更新数据
            wavelengths = standard_wavelengths
            dvp_values = interpolated_values
        else:
            print("✓ 数据已经是标准格式，无需插值")
        
        # 简单可视化
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(wavelengths, dvp_values, 'b-', linewidth=2, label='DVP标准曲线')
            plt.xlabel('波长 (nm)')
            plt.ylabel('反射率')
            plt.title('DVP涂层标准光谱曲线')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存图表
            output_path = "/workspace/code/spectrum_anomaly_detection/visualizations/dvp_standard_curve.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 标准曲线图已保存: {output_path}")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过可视化")
        
        # 保存处理后的数据
        output_data = {
            'wavelengths': wavelengths,
            'dvp_values': dvp_values,
            'metadata': {
                'coating_name': 'DVP',
                'source_file': csv_path,
                'wavelength_range': f"{wavelengths.min():.0f}-{wavelengths.max():.0f}nm",
                'data_points': len(wavelengths),
                'processed_at': pd.Timestamp.now().isoformat()
            }
        }
        
        np.savez_compressed("/workspace/code/spectrum_anomaly_detection/data/dvp_processed_data.npz", **output_data)
        print(f"✓ 处理后数据已保存: /workspace/code/spectrum_anomaly_detection/data/dvp_processed_data.npz")
        
        print("\n🎉 Phase 1 数据预处理完成！")
        print("数据已准备就绪，可以开始 Phase 2。")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()