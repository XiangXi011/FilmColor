#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试字体修复的简单脚本
生成一个简单的图表来验证中文字体是否正常显示
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    import warnings
except ImportError as e:
    print(f"❌ 缺少必要的依赖: {e}")
    print("请先安装: pip install matplotlib")
    exit(1)

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib for plotting with proper Chinese font configuration.
    """
    warnings.filterwarnings('default')
    
    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")
    
    # Set chart style
    plt.style.use("seaborn-v0_8")
    
    # Configure platform-appropriate fonts for cross-platform compatibility
    system = platform.system()
    if system == "Windows":
        # Windows系统优先使用微软雅黑和黑体
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong"]
    elif system == "Darwin":  # macOS
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
    else:  # Linux
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
    
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def test_chinese_fonts():
    """测试中文字体显示"""
    print("🔧 设置matplotlib字体...")
    setup_matplotlib_for_plotting()
    
    # 检查当前字体设置
    print(f"📝 当前字体设置: {plt.rcParams['font.sans-serif']}")
    print(f"🌐 操作系统: {platform.system()}")
    
    # 创建一个简单的测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试各种中文标题和标签
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    ax.plot(x, y, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('波长 (nm)', fontsize=14)
    ax.set_ylabel('光谱强度', fontsize=14)
    ax.set_title('DVP涂层光谱异常检测 - 字体测试', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(['测试数据'], loc='upper left')
    
    # 添加一些中文文本
    ax.text(3, 8, '这是中文测试文本', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图像
    output_path = "evaluation/font_test.png"
    import os
    os.makedirs("evaluation", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 测试图片已保存: {output_path}")
    print("📊 请查看图片，确认中文字体是否正常显示（不应该出现乱码）")
    
    return output_path

if __name__ == "__main__":
    try:
        test_chinese_fonts()
        print("\n🎉 字体测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

