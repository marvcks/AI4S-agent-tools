#!/usr/bin/env python3
"""
PySCF MCP Server - 量子化学计算服务器
提供基于PySCF的量子化学计算功能，通过执行用户提供的PySCF代码进行计算
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import urllib
# 导入CalculationMCPServer
from dp.agent.server import CalculationMCPServer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MOLPILOT MCP服务器")
    parser.add_argument('--port', type=int, default=50001, help='服务器端口 (默认: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机 (默认: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("molpilot_server_pyscf", host=args.host, port=args.port)

# 配置日志
logging.basicConfig(
    level=args.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@mcp.tool()
def retrieve_pyscf_doc(keywords: str) -> str:
    """
    Retrieve documentation for PySCF functions or classes based on keywords.
    Args:
        keywords (str): Keywords to search for in the PySCF documentation.
    Returns:
        str: The relevant documentation or an error message if not found.        
    """
    #replace spaces with +
    CONTEXT7_API_KEY = "ctx7sk-21b8931d-2a53-4928-a8da-6bd2f7629979"
    keywords = urllib.parse.quote_plus(keywords)
    url = f"https://context7.com/api/v1/pyscf/pyscf.github.io?type=txt&topic={keywords}&tokens=1000"

    headers = {"Authorization": f"Bearer {CONTEXT7_API_KEY}"}

    try:
        result = requests.get(url, headers=headers, timeout=10)
        return result.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving PySCF documentation: {e}")
        return f"Error retrieving PySCF documentation: {e}"


@mcp.tool()
async def read_pyscf_output(pyscf_output: Path):
    """
    读取`run_pyscf_code`函数的输出文件.
    参数:
    - pyscf_output: 输出文件路径
    
    返回:
    - Dict[str, Any]: 包含计算结果的字典，包含以下字段：
        - success: bool - 计算是否成功
        - output: str - 计算的标准输出
        - error: str - 错误信息（如果有）
    """
    try:
        with open(pyscf_output, 'r') as f:
            content = f.read()
        return {
            "success": True,
            "output": content
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    

@mcp.tool()
async def run_pyscf_code(xyz_path: Path, pyscf_code: str) -> Dict[str, Any]:
    """
    执行用户提供的PySCF代码进行量子化学计算
    
    参数:
    - xyz_path: 分子几何结构的XYZ文件路径
    - pyscf_code: 要执行的PySCF Python代码字符串
    
    返回:
    - Dict[str, Any]: 包含计算结果的字典，包含以下字段：
        - success: bool - 计算是否成功
        - output: str - 计算的标准输出
        - error: str - 错误信息（如果有）
        - exit_code: int - 程序退出码
    
    重要说明 - 如何编写pyscf_code:
    1. 命令行参数接收:
       您的pyscf_code必须能够从命令行接收xyz_path参数.请在代码开头添加:
       ```python
       import sys
       xyz_path = sys.argv[1]  # 获取XYZ文件路径
       ```python
    2. 读取分子结构:
       使用xyz_path读取分子几何结构,例如:
       ```python
       from pyscf import gto
       mol = gto.Mole()
       mol.atom = xyz_path  # 直接使用XYZ文件路径
       mol.basis = 'sto-3g'
       mol.build()
       ```
    3. 完整示例代码结构:
        ```python       
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
       ```
    4. 输出结果:
       请使用print()函数输出您想要返回的计算结果。所有print输出都会被捕获并返回。
    5. 错误处理:
       如果计算过程中出现错误，程序会自动捕获并在返回结果中包含错误信息。
       
    注意事项:
    - 代码将在临时环境中执行，请确保所有必要的导入都包含在代码中
    - 避免使用可能造成安全风险的操作（如文件系统操作、网络访问等）

    ## 针对Skala泛函的补充
    
    Skala是一个基于神经网络的高精度交换相关泛函，专门设计用于提高DFT计算的精度。
    要使用Skala泛函，您需要在pyscf_code中按照以下方式编写：
    
    ### 基本Skala计算示例：
    ```python
    import sys
    from pyscf import gto
    from skala.pyscf import SkalaKS
    
    # 获取XYZ文件路径
    xyz_path = sys.argv[1]
    
    # 构建分子
    mol = gto.Mole()
    mol.atom = xyz_path
    mol.basis = "def2-tzvp"  # 推荐使用def2-tzvp基组
    mol.charge = 0
    mol.spin = 0
    mol.build()
    
    # 创建SkalaKS计算器
    ks = SkalaKS(mol, xc="skala")
    
    # 执行SCF计算
    energy = ks.kernel()
    
    # 输出结果
    print(f"Skala SCF Energy: {energy:.8f} Hartree")
    print(ks.dump_scf_summary())
    ```
    
    ### 高级Skala计算选项：
    
    1. **使用密度拟合加速计算**：
    ```python
    ks = SkalaKS(mol, xc="skala", with_density_fit=True)
    ```
    
    2. **使用Newton方法提高收敛性**：
    ```python
    ks = SkalaKS(mol, xc="skala", with_density_fit=True, with_newton=True)
    ```
    
    3. **禁用DFT-D3色散校正**（默认启用）：
    ```python
    ks = SkalaKS(mol, xc="skala", with_dftd3=False)
    ```
    
    4. **完整的高级配置示例**：
    ```python
    import sys
    from pyscf import gto
    from skala.pyscf import SkalaKS
    
    xyz_path = sys.argv[1]
    
    mol = gto.Mole()
    mol.atom = xyz_path
    mol.basis = "def2-tzvp"
    mol.charge = 0
    mol.spin = 0
    mol.build()
    
    # 创建高级配置的SkalaKS计算器
    ks = SkalaKS(
        mol, 
        xc="skala",
        with_density_fit=True,    # 启用密度拟合
        with_newton=True,         # 启用Newton方法
        with_dftd3=True          # 启用DFT-D3色散校正
    )
    
    energy = ks.kernel()
    
    if ks.converged:
        print(f"Skala calculation converged!")
        print(f"Total energy: {energy:.8f} Hartree")
        
        # 可以进行后续分析
        dipole = ks.dip_moment()
        print(f"Dipole moment: {dipole}")
        
        # 输出详细摘要
        print("\nDetailed SCF Summary:")
        print(ks.dump_scf_summary())
    else:
        print("WARNING: Skala calculation did not converge!")
    ```
    
    ### Skala泛函的特点：
    - **高精度**：基于神经网络训练，提供比传统泛函更高的精度
    - **自动色散校正**：默认包含DFT-D3色散校正
    - **兼容性**：完全兼容PySCF的所有功能和方法
    - **推荐基组**：建议使用def2-tzvp或更大的基组以获得最佳精度
    
    ### 注意事项：
    - Skala计算通常比传统DFT泛函需要更多计算时间
    - 对于大分子系统，强烈建议使用密度拟合（with_density_fit=True）
    - 如果遇到收敛问题，可以尝试启用Newton方法（with_newton=True）

    """
    
    logger.info(f"Starting PySCF calculation with XYZ file: {xyz_path}")
    
    try:
        # 验证XYZ文件是否存在
        if not xyz_path.exists():
            return {
                "success": False,
                "output": "",
                "error": f"XYZ file not found: {xyz_path}",
                "exit_code": 1
            }
        
        # 将PySCF代码写入临时文件
        pyscf_script_path = "run_pyscf.py"
        with open(pyscf_script_path, 'w', encoding='utf-8') as f:
            f.write(pyscf_code)
        
        logger.info(f"PySCF script written to: {pyscf_script_path}")
        logger.info(f"{str(xyz_path)}")
        # 执行PySCF脚本
        try:
            # 创建输出文件
            output_file = "pyscf_output.txt"
            with open(output_file, "w", encoding="utf-8") as out_f:
                result = subprocess.run(
                    # python xx.py xx.xyz > out.txt
                    [sys.executable, str(pyscf_script_path), str(xyz_path)],
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    text=True,
                    # timeout=300  # 5分钟超时
                )
            
            logger.info(f"PySCF calculation completed with exit code: {result.returncode}")
            
            return {
                "success": result.returncode == 0,
                "output": Path(output_file),
                "error": result.stderr,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.error("PySCF calculation timed out")
            return {
                "success": False,
                "output": "",
                "error": "Calculation timed out after 5 minutes",
                "exit_code": 124
            }
        
        except Exception as e:
                logger.error(f"Error executing PySCF script: {str(e)}")
                return {
                    "success": False,
                    "output": "",
                    "error": f"Error executing PySCF script: {str(e)}",
                    "exit_code": 1
                }
                
    except Exception as e:
        logger.error(f"Unexpected error in run_pyscf_code: {str(e)}")
        return {
            "success": False,
            "output": "",
            "error": f"Unexpected error: {str(e)}",
            "exit_code": 1
        }


# @mcp.tool()
# async def run_skala_code(xyz_path: Path, pyscf_code: str) -> Dict[str, Any]:


if __name__ == "__main__":
    logger.info(f"Starting One PySCF MCP Server on {args.host}:{args.port}")
    mcp.run(transport="sse")

 