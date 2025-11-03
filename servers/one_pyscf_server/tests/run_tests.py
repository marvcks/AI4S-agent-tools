#!/usr/bin/env python3
"""
PySCF MCP Server 测试运行脚本
用于运行所有测试并生成测试报告
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """主函数"""
    
    # 获取测试目录
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    print("=" * 60)
    print("PySCF MCP Server 测试套件")
    print("=" * 60)
    
    # 检查依赖
    print("\n1. 检查测试依赖...")
    try:
        import pytest
        print("✅ pytest 已安装")
    except ImportError:
        print("❌ pytest 未安装，请运行: pip install pytest")
        return 1
    
    try:
        import pyscf
        print("✅ pyscf 已安装")
    except ImportError:
        print("❌ pyscf 未安装，请运行: pip install pyscf")
        return 1
    
    # 检查测试文件
    print("\n2. 检查测试文件...")
    test_files = [
        test_dir / "test_server.py",
        test_dir / "pyscf_codes" / "simple_dft.py",
        test_dir / "pyscf_codes" / "invalid_code.py",
        test_dir / "test_xyz" / "H2O.xyz",
        test_dir / "test_xyz" / "test_molecule.xyz",
        test_dir / "test_xyz" / "invalid.xyz"
    ]
    
    for test_file in test_files:
        if test_file.exists():
            print(f"✅ {test_file.name}")
        else:
            print(f"❌ {test_file.name} 不存在")
            return 1
    
    # 运行测试
    print("\n3. 运行测试...")
    print("-" * 40)
    
    # 方式1: 使用pytest运行
    print("使用 pytest 运行测试:")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_dir / "test_server.py"), 
            "-v", "--tb=short"
        ], cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ pytest 测试通过!")
        else:
            print(f"\n❌ pytest 测试失败 (退出码: {result.returncode})")
    
    except Exception as e:
        print(f"❌ 运行pytest时出错: {str(e)}")
    
    print("\n" + "-" * 40)
    
    # 方式2: 直接运行测试文件
    print("直接运行测试文件:")
    try:
        result = subprocess.run([
            sys.executable, str(test_dir / "test_server.py")
        ], cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ 直接运行测试通过!")
        else:
            print(f"\n❌ 直接运行测试失败 (退出码: {result.returncode})")
    
    except Exception as e:
        print(f"❌ 直接运行测试时出错: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())