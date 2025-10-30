#!/usr/bin/env python3
"""
DVP涂层光谱异常检测系统 - 快速启动脚本
一键启动API服务、运行测试、查看状态

Author: MiniMax Agent
Date: 2025-10-30
"""

import os
import sys
import subprocess
import time
import requests
import argparse
from pathlib import Path
import json

class QuickStarter:
    """快速启动器类"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.api_process = None
        
    def check_environment(self):
        """检查环境"""
        print("🔍 检查系统环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python版本过低，需要3.8+")
            return False
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查虚拟环境
        if not self.venv_path.exists():
            print("⚠️  虚拟环境不存在，将自动创建")
            self.create_virtual_environment()
        
        print("✅ 环境检查完成")
        return True
    
    def create_virtual_environment(self):
        """创建虚拟环境"""
        print("📦 创建虚拟环境...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            print("✅ 虚拟环境创建成功")
        except subprocess.CalledProcessError:
            print("❌ 虚拟环境创建失败")
            return False
        return True
    
    def install_dependencies(self):
        """安装依赖"""
        print("📚 安装依赖包...")
        
        # 获取pip路径
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip.exe"
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:  # Linux/macOS
            pip_path = self.venv_path / "bin" / "pip"
            python_path = self.venv_path / "bin" / "python"
        
        try:
            # 升级pip
            subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # 安装依赖
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            print("✅ 依赖安装完成")
            return True
        except subprocess.CalledProcessError:
            print("❌ 依赖安装失败")
            return False
    
    def run_tests(self):
        """运行基础测试"""
        print("🧪 运行基础功能测试...")
        
        # 获取Python路径
        if os.name == 'nt':
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
        
        test_files = [
            "test_data_loading.py",
            "test_phase2.py", 
            "test_phase3.py"
        ]
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                print(f"运行 {test_file}...")
                try:
                    result = subprocess.run([str(python_path), str(test_path)], 
                                          capture_output=True, text=True, cwd=self.project_root)
                    if result.returncode == 0:
                        print(f"✅ {test_file} 测试通过")
                    else:
                        print(f"❌ {test_file} 测试失败:")
                        print(result.stderr)
                except Exception as e:
                    print(f"❌ {test_file} 运行异常: {e}")
            else:
                print(f"⚠️  {test_file} 不存在，跳过")
    
    def start_api_server(self, port=8000, background=True):
        """启动API服务器"""
        print(f"🚀 启动API服务器 (端口: {port})...")
        
        # 获取Python路径
        if os.name == 'nt':
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
        
        try:
            if background:
                # 后台启动
                self.api_process = subprocess.Popen([
                    str(python_path), "api_server.py"
                ], cwd=self.project_root)
                print("✅ API服务器已在后台启动")
            else:
                # 前台启动
                subprocess.run([str(python_path), "api_server.py"], cwd=self.project_root)
                return
            
            # 等待服务器启动
            print("⏳ 等待服务器启动...")
            time.sleep(5)
            
            # 检查服务器状态
            if self.check_api_health():
                print("✅ API服务器启动成功")
                print(f"🌐 访问地址: http://localhost:{port}")
                print(f"📖 API文档: http://localhost:{port}/docs")
            else:
                print("❌ API服务器启动失败")
                
        except Exception as e:
            print(f"❌ API服务器启动异常: {e}")
    
    def check_api_health(self, port=8000):
        """检查API健康状态"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_api_tests(self, port=8000):
        """运行API测试"""
        print("🧪 运行API测试...")
        
        # 获取Python路径
        if os.name == 'nt':
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
        
        try:
            result = subprocess.run([
                str(python_path), "test_api.py"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ API测试通过")
                print(result.stdout)
            else:
                print("❌ API测试失败:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ API测试异常: {e}")
    
    def show_status(self):
        """显示系统状态"""
        print("📊 系统状态:")
        
        # 检查API服务器
        if self.api_process and self.api_process.poll() is None:
            print("✅ API服务器: 运行中")
            if self.check_api_health():
                print("✅ API健康检查: 正常")
            else:
                print("❌ API健康检查: 异常")
        else:
            print("❌ API服务器: 未运行")
        
        # 显示项目结构
        print("\n📁 项目结构:")
        for item in self.project_root.iterdir():
            if item.is_dir():
                print(f"  📂 {item.name}/")
            else:
                print(f"  📄 {item.name}")
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "="*50)
            print("🎯 DVP涂层光谱异常检测系统")
            print("="*50)
            print("1. 环境检查和设置")
            print("2. 安装依赖包")
            print("3. 运行基础测试")
            print("4. 启动API服务器")
            print("5. 运行API测试")
            print("6. 显示系统状态")
            print("7. 一键完整启动")
            print("0. 退出")
            print("="*50)
            
            choice = input("请选择操作 (0-7): ").strip()
            
            if choice == "1":
                self.check_environment()
            elif choice == "2":
                self.install_dependencies()
            elif choice == "3":
                self.run_tests()
            elif choice == "4":
                port = input("请输入端口号 (默认8000): ").strip() or "8000"
                self.start_api_server(int(port))
            elif choice == "5":
                self.run_api_tests()
            elif choice == "6":
                self.show_status()
            elif choice == "7":
                self.full_startup()
            elif choice == "0":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def full_startup(self):
        """一键完整启动"""
        print("🚀 开始一键完整启动...")
        
        # 1. 环境检查
        if not self.check_environment():
            return
        
        # 2. 安装依赖
        if not self.install_dependencies():
            return
        
        # 3. 运行测试
        self.run_tests()
        
        # 4. 启动API服务器
        self.start_api_server()
        
        # 5. 运行API测试
        time.sleep(2)  # 等待服务器完全启动
        self.run_api_tests()
        
        print("\n🎉 一键启动完成！")
        print("🌐 API服务地址: http://localhost:8000")
        print("📖 API文档地址: http://localhost:8000/docs")
        print("🔧 交互式菜单: python quick_start.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DVP涂层光谱异常检测系统快速启动器")
    parser.add_argument("--mode", choices=["menu", "start", "test", "status"], 
                       default="menu", help="启动模式")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    
    args = parser.parse_args()
    
    starter = QuickStarter()
    
    if args.mode == "menu":
        starter.interactive_menu()
    elif args.mode == "start":
        starter.check_environment()
        starter.install_dependencies()
        starter.start_api_server(args.port, background=False)
    elif args.mode == "test":
        starter.check_environment()
        starter.run_tests()
        starter.run_api_tests(args.port)
    elif args.mode == "status":
        starter.show_status()

if __name__ == "__main__":
    main()