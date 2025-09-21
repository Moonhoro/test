#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装和测试脚本
Installation and Test Script
"""

import subprocess
import sys
import os

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("依赖包安装成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False

def test_imports():
    """测试导入模块"""
    print("测试模块导入...")
    
    required_modules = [
        'cv2', 'numpy', 'matplotlib', 'sklearn', 
        'scipy', 'pandas', 'seaborn'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"缺少模块: {missing_modules}")
        return False
    else:
        print("所有模块导入成功!")
        return True

def check_data_structure():
    """检查数据结构"""
    print("检查数据结构...")
    
    required_paths = [
        "附件1",
        "附件2", 
        "附件3",
        "附件4",
        "围岩裂隙精准识别与三维模型重构.docx"
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path}")
            missing_paths.append(path)
    
    # 检查钻孔数据
    if os.path.exists("附件4"):
        borehole_dirs = [d for d in os.listdir("附件4") 
                        if os.path.isdir(os.path.join("附件4", d)) and d.endswith("孔")]
        print(f"找到 {len(borehole_dirs)} 个钻孔: {borehole_dirs}")
        
        for borehole in borehole_dirs:
            borehole_path = os.path.join("附件4", borehole)
            images = [f for f in os.listdir(borehole_path) if f.endswith('.jpg')]
            print(f"  {borehole}: {len(images)} 张图像")
    
    if missing_paths:
        print(f"缺少数据: {missing_paths}")
        return False
    else:
        print("数据结构检查完成!")
        return True

def run_quick_test():
    """运行快速测试"""
    print("运行快速测试...")
    
    try:
        # 导入主模块
        from fracture_analysis import FractureDetector, FractureAnalyzer
        
        # 创建检测器实例
        detector = FractureDetector()
        print("✓ 检测器创建成功")
        
        # 测试配置
        config = detector.config
        print(f"✓ 配置加载成功: {len(config)} 个参数")
        
        # 检查是否有测试图像
        test_image_path = None
        if os.path.exists("附件4"):
            for borehole in os.listdir("附件4"):
                borehole_path = os.path.join("附件4", borehole)
                if os.path.isdir(borehole_path):
                    images = [f for f in os.listdir(borehole_path) if f.endswith('.jpg')]
                    if images:
                        test_image_path = os.path.join(borehole_path, images[0])
                        break
        
        if test_image_path:
            print(f"找到测试图像: {test_image_path}")
            
            # 简单的图像读取测试
            import cv2
            image = cv2.imread(test_image_path)
            if image is not None:
                print(f"✓ 图像读取成功: {image.shape}")
                
                # 预处理测试
                processed = detector.preprocess_image(image)
                print(f"✓ 图像预处理成功: {processed.shape}")
            else:
                print("✗ 图像读取失败")
                return False
        else:
            print("⚠ 未找到测试图像，跳过图像处理测试")
        
        print("快速测试完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 围岩裂隙分析系统 - 安装与测试 ===")
    print()
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("错误: 需要Python 3.7或更高版本")
        return False
    
    # 步骤1: 安装依赖
    if not install_requirements():
        print("安装失败，请检查网络连接和权限")
        return False
    
    print()
    
    # 步骤2: 测试导入
    if not test_imports():
        print("模块导入失败，请检查安装")
        return False
    
    print()
    
    # 步骤3: 检查数据
    if not check_data_structure():
        print("数据结构不完整")
        return False
    
    print()
    
    # 步骤4: 快速测试
    if not run_quick_test():
        print("功能测试失败")
        return False
    
    print()
    print("✅ 所有测试通过!")
    print()
    print("使用说明:")
    print("1. 运行 python fracture_analysis.py 开始完整分析")
    print("2. 结果将保存在 output/ 目录中")
    print("3. 查看 SOLUTION.md 了解详细技术文档")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)