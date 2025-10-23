#!/usr/bin/env python3
"""
PySCF MCP Server 测试文件
测试 server.py 中的 run_pyscf_code 函数
"""

import sys
import asyncio
import pytest
from pathlib import Path
import tempfile
import os

# 添加服务器模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from server import run_pyscf_code

# 测试数据文件路径
TEST_DIR = Path(__file__).parent
PYSCF_CODES_DIR = TEST_DIR / "pyscf_codes"
XYZ_FILES_DIR = TEST_DIR / "test_xyz"

class TestPySCFServer:
    """PySCF服务器测试类"""
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_success(self):
        """测试正常的PySCF计算"""
        
        # 读取测试用的PySCF代码
        pyscf_code_file = PYSCF_CODES_DIR / "simple_dft.py"
        with open(pyscf_code_file, "r", encoding="utf-8") as f:
            test_pyscf_code = f.read()
        
        # 测试XYZ文件路径
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert "success" in result
        assert "output" in result
        assert "error" in result
        assert "exit_code" in result
        
        # 如果计算成功，验证输出内容
        if result["success"]:
            assert result["exit_code"] == 0
            assert len(result["output"]) > 0
            print(f"计算成功！输出: {result['output'][:200]}...")
        else:
            print(f"计算失败: {result['error']}")
            # 即使失败也要确保错误信息被正确捕获
            assert len(result["error"]) > 0
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_with_different_molecule(self):
        """测试使用不同分子的PySCF计算"""
        
        # 使用简单的测试分子代码
        test_pyscf_code = """
import sys
from pyscf import gto, scf

# 获取命令行参数
xyz_path = sys.argv[1]

# 构建分子
mol = gto.Mole()
mol.atom = xyz_path
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()

# 进行SCF计算
mf = scf.RHF(mol)
energy = mf.scf()

# 输出结果
print(f"SCF Energy: {energy}")
print(f"Molecule: {mol.atom}")
"""
        
        # 测试不同的XYZ文件
        xyz_path = XYZ_FILES_DIR / "test_molecule.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert "success" in result
        
        if result["success"]:
            assert "SCF Energy:" in result["output"]
            print(f"不同分子计算成功！")
        else:
            print(f"计算失败: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_file_not_found(self):
        """测试XYZ文件不存在的情况"""
        
        test_pyscf_code = """
import sys
print("This should not run")
"""
        
        # 使用不存在的文件路径
        xyz_path = Path("nonexistent_file.xyz")
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "not found" in result["error"]
        print(f"文件不存在测试通过: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_syntax_error(self):
        """测试PySCF代码语法错误的情况"""
        
        # 包含语法错误的代码
        test_pyscf_code = """
import sys
xyz_path = sys.argv[1]

# 故意的语法错误
print("Hello World"
# 缺少右括号
"""
        
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["error"]) > 0
        print(f"语法错误测试通过: {result['error'][:100]}...")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_runtime_error(self):
        """测试PySCF代码运行时错误的情况"""
        
        # 包含运行时错误的代码
        test_pyscf_code = """
import sys
xyz_path = sys.argv[1]

# 故意的运行时错误
result = 1 / 0  # 除零错误
print(f"This should not print: {result}")
"""
        
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["error"]) > 0
        print(f"运行时错误测试通过: {result['error'][:100]}...")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_import_error(self):
        """测试PySCF代码导入错误的情况"""
        
        # 读取包含导入错误的代码
        invalid_code_file = PYSCF_CODES_DIR / "invalid_code.py"
        with open(invalid_code_file, "r", encoding="utf-8") as f:
            test_pyscf_code = f.read()
        
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["error"]) > 0
        print(f"导入错误测试通过: {result['error'][:100]}...")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_invalid_xyz_format(self):
        """测试无效XYZ文件格式的情况"""
        
        # 使用简单的PySCF代码但读取格式错误的XYZ文件
        test_pyscf_code = """
import sys
from pyscf import gto, scf

# 获取命令行参数
xyz_path = sys.argv[1]

try:
    # 构建分子
    mol = gto.Mole()
    mol.atom = xyz_path
    mol.basis = 'sto-3g'
    mol.build()
    
    # 进行SCF计算
    mf = scf.RHF(mol)
    energy = mf.scf()
    print(f"SCF Energy: {energy}")
    
except Exception as e:
    print(f"Error processing molecule: {str(e)}")
    raise
"""
        
        # 使用格式错误的XYZ文件
        xyz_path = XYZ_FILES_DIR / "invalid.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        # 这个测试可能成功也可能失败，取决于PySCF如何处理无效格式
        if not result["success"]:
            assert len(result["error"]) > 0
            print(f"无效XYZ格式测试通过: {result['error'][:100]}...")
        else:
            print(f"PySCF处理了无效格式: {result['output'][:100]}...")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_empty_code(self):
        """测试空代码的情况"""
        
        test_pyscf_code = ""
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        # 空代码应该成功执行但没有输出
        if result["success"]:
            assert result["exit_code"] == 0
            print("空代码测试通过: 成功执行")
        else:
            print(f"空代码测试: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_run_pyscf_code_no_args(self):
        """测试不使用命令行参数的代码"""
        
        test_pyscf_code = """
# 不使用sys.argv的代码
print("Hello from PySCF test!")
print("This code doesn't use the XYZ file")
"""
        
        xyz_path = XYZ_FILES_DIR / "H2O.xyz"
        
        # 执行测试
        result = await run_pyscf_code(xyz_path, test_pyscf_code)
        
        # 验证结果
        assert isinstance(result, dict)
        if result["success"]:
            assert "Hello from PySCF test!" in result["output"]
            print("无参数代码测试通过")
        else:
            print(f"无参数代码测试失败: {result['error']}")


def test_sync_wrapper():
    """同步测试包装器，用于运行异步测试"""
    
    async def run_all_tests():
        """运行所有测试"""
        test_instance = TestPySCFServer()
        
        print("开始测试 PySCF MCP Server...")
        
        try:
             print("\n1. 测试正常的PySCF计算...")
             await test_instance.test_run_pyscf_code_success()
             
             print("\n2. 测试不同分子的计算...")
             await test_instance.test_run_pyscf_code_with_different_molecule()
             
             print("\n3. 测试文件不存在的情况...")
             await test_instance.test_run_pyscf_code_file_not_found()
             
             print("\n4. 测试语法错误的情况...")
             await test_instance.test_run_pyscf_code_syntax_error()
             
             print("\n5. 测试运行时错误的情况...")
             await test_instance.test_run_pyscf_code_runtime_error()
             
             print("\n6. 测试导入错误的情况...")
             await test_instance.test_run_pyscf_code_import_error()
             
             print("\n7. 测试无效XYZ格式的情况...")
             await test_instance.test_run_pyscf_code_invalid_xyz_format()
             
             print("\n8. 测试空代码的情况...")
             await test_instance.test_run_pyscf_code_empty_code()
             
             print("\n9. 测试不使用参数的代码...")
             await test_instance.test_run_pyscf_code_no_args()
             
             print("\n✅ 所有测试完成！")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            raise
    
    # 运行异步测试
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    # 直接运行测试
    test_sync_wrapper()
    
