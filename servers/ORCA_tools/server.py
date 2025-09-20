#!/usr/bin/env python3
"""
MolPilot服务器 - 整合ORCA工具、 ORCA输出文件解析工具、ORCA手册RAG工具、AutoDE功能等.
"""
import os
import re
import logging
import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import sys

# 导入MCP相关
from dp.agent.server import CalculationMCPServer

# 导入AutoDE相关
import autode as ade
import anyio

# 导入RAG相关
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from langchain_core.documents import Document

# 导入数据处理相关
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 导入分子可视化相关
import molviewspec as mvs
from ase.io import read, write

# 设置环境变量
os.environ["AUTODE_LOG_LEVEL"] = "INFO"
os.environ["AUTODE_LOG_FILE"] = "autode.log"
# 允许以 root 运行 MPI
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"


# 下面的内容注释掉了，因为如果不把任务提到Bohr上,就不需要显示的设置环境变量。
# # 初始化 PATH 和 LD_LIBRARY_PATH
# os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/opt/mamba/bin"
# os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
# os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + os.pathsep + "/usr/local/cuda/lib64"
# # OpenMPI 路径
# os.environ["PATH"] += os.pathsep + "/opt/openmpi411/bin"
# os.environ["LD_LIBRARY_PATH"] += os.pathsep + "/opt/openmpi411/lib"
# # ORCA 路径
# os.environ["PATH"] += os.pathsep + "/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411"
# os.environ["LD_LIBRARY_PATH"] += os.pathsep + "/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411"
# # packmol 路径
# os.environ["PATH"] += os.pathsep + "/root/packmol-21.1.0"
# # xtb 路径
# os.environ["PATH"] += os.pathsep + "/root/xtb-6.6.1/bin"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MOLPILOT MCP服务器")
    parser.add_argument('--port', type=int, default=50005, help='服务器端口 (默认: 50005)')
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
mcp = CalculationMCPServer("molpilot_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)

@mcp.tool()
async def get_data_from_opt_output(
    orca_output_file: Path,
    properties: Optional[Union[str, List[str]]] = "energy"
) -> Dict[str, str]:
    """
    Get data from ORCA output file with specified properties.
    
    Args:
        orca_output_file: Path to ORCA output file
        properties: List of properties to get.Possible values: 
                    'coordinates', 'energy', 'point_group', 'dipole',
                    'orbitals', 'mulliken', 'loewdin', 'hirshfeld'
    Returns:
        Dictionary with status and message.
    """
    # Define all available properties and their extraction patterns
    PROPERTY_EXTRACTORS = {
        'coordinates': {
            'pattern': r"CARTESIAN COORDINATES \(ANGSTROEM\)(.*?)(?=^CARTESIAN COORDINATES \(A\.U\.\))",
            'flags': re.DOTALL | re.MULTILINE,
            'formatter': lambda match: format_coordinates(match[-1].strip())
        },
        'energy': {
            'pattern': r"Total Energy\s*:\s*([-0-9\.]+)\s*Eh",
            'formatter': lambda match: f"Total SCF Energy: {match[-1]} Hartree\n"
        },
        'point_group': {
            'pattern': r"Point Group:\s*(\S+),",
            'formatter': lambda match: f"Point Group: {match[-1]}\n"
        },
        'dipole': {
            'pattern': r"Magnitude \(Debye\)\s*:\s*([0-9\.]+)",
            'formatter': lambda match: f"Dipole Moment: {match[-1]} Debye\n"
        },
        'orbitals': {
            'pattern': r"NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s+([\s\S]*?)\n------------------",
            'formatter': lambda match: format_orbitals(match[-1].strip())
        },
        'mulliken': {
            'pattern': r"MULLIKEN ATOMIC CHARGES\s*-+\s*\n([\s\S]*?)\nSum of atomic charges",
            'formatter': lambda match: format_charges(match[-1].strip(), "Mulliken")
        },
        'loewdin': {
            'pattern': r"LOEWDIN ATOMIC CHARGES\s*-+\s*\n([\s\S]*?)(?=\n-+|\n[A-Z])",
            'formatter': lambda match: format_charges(match[-1].strip(), "Loewdin")
        },
        'hirshfeld': {
            'pattern': r"HIRSHFELD ANALYSIS\s*-+\s*\n[\s\S]*?\n\s*ATOM\s+CHARGE\s+SPIN\s*\n([\s\S]*?)(?=\n\s*TOTAL|\n-+|\n[A-Z])",
            'formatter': lambda match: format_charges(match[-1].strip(), "Hirshfeld")
        }
    }

    # Helper functions for formatting
    def format_coordinates(coords_text: str) -> str:
        coord_lines = []
        for line in coords_text.split('\n'):
            if line.strip() and "----" not in line:
                match = line.split()
                if len(match) == 4:
                    element = match[0]
                    x, y, z = map(float, match[1:])
                    coord_lines.append(f"{element:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
        
        if not coord_lines:
            return ""
            
        output = [
            "Cartesian Coordinates (Å):",
            f"{'Element':<2} {'X':>12} {'Y':>12} {'Z':>12}",
            "-" * 42,
            "\n".join(coord_lines),
            ""
        ]
        return "\n".join(output)

    def format_orbitals(orbitals_text: str) -> str:
        orbital_energies = orbitals_text.strip().splitlines()
        orbital_data = []
        output = [
            "Orbital Energies:",
            f"{'NO':>3} {'OCC':>12} {'E (Eh)':>12}",
            "-" * 32
        ]
        
        for line in orbital_energies:
            match = re.match(r"\s*(\d+)\s+([\d\.]+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", line)
            if match:
                no, occ, energy_eh, energy_ev = int(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                orbital_data.append((no, occ, energy_eh, energy_ev))
                output.append(f"{no:>3} {occ:>12.6f} {energy_eh:>12.6f}")
        
        # Find HOMO and LUMO
        homo, lumo = None, None
        for i in range(len(orbital_data) - 1, -1, -1):
            if orbital_data[i][1] == 2.0000:
                homo = orbital_data[i]
                break
                
        for i in range(len(orbital_data)):
            if orbital_data[i][1] == 0.0000:
                lumo = orbital_data[i]
                break
        
        if homo:
            output.append(f"\nHOMO Energy (Eh): {homo[2]:.6f}, Energy (eV): {homo[3]:.4f}")
        if lumo:
            output.append(f"LUMO Energy (Eh): {lumo[2]:.6f}, Energy (eV): {lumo[3]:.4f}")
        if homo and lumo:
            gap_Eh = lumo[2] - homo[2]
            gap_eV = lumo[3] - homo[3]
            output.append(f"HOMO-LUMO Gap (Eh): {gap_Eh:.6f}")
            output.append(f"HOMO-LUMO Gap (eV): {gap_eV:.4f}")
        
        output.append("")
        return "\n".join(output)

    def format_charges(charges_text: str, charge_type: str) -> str:
        output = [
            f"{charge_type} Atomic Charges:",
            f"{'Index':>3} {'Element':<2} {'Charge':>12}",
            "-" * 30
        ]
        
        for line in charges_text.split("\n"):
            if charge_type == "Hirshfeld":
                match = re.match(r"\s*(\d+)\s+(\S+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)", line)
            else:
                match = re.match(r"\s*(\d+)\s+(\S+)\s*:\s*([-+]?\d*\.\d+)", line)
            
            if match:
                index = int(match.group(1))
                element = match.group(2)
                charge = float(match.group(3))
                output.append(f"{index:>3} {element:<2} {charge:>12.6f}")
        
        output.append("")
        return "\n".join(output)

    # Process properties parameter
    if properties is None:
        properties = list(PROPERTY_EXTRACTORS.keys())
    elif isinstance(properties, str):
        properties = [properties]
    
    # Validate requested properties
    invalid_props = [p for p in properties if p not in PROPERTY_EXTRACTORS]
    if invalid_props:
        return {
            "status": "error",
            "message": f"Invalid properties requested: {', '.join(invalid_props)}. Valid options are: {', '.join(PROPERTY_EXTRACTORS.keys())}"
        }
    report_file_name = "orca_opt_report.txt"
    report_output_file = Path(report_file_name)
    
    try:
        with open(orca_output_file, 'r', encoding='utf-8') as file:
            file_content = file.read()

        if "Number of atoms                             ...      1" in file_content:
            post_convergence = file_content.split("TOTAL SCF ENERGY")[-1]
            
            with open(report_output_file, 'w') as report:
                report.write("Single Atom Report\n")
                report.write("====================\n\n")
                report.write(f"Single Atom Report for: {report_output_file}\n\n")
                
                for prop in properties:
                    extractor = PROPERTY_EXTRACTORS[prop]
                    flags = extractor.get('flags', 0)
                    matches = re.findall(extractor['pattern'], post_convergence, flags)
                    
                    if matches:
                        formatted = extractor['formatter'](matches)
                        if formatted:
                            report.write(formatted + "\n")
            
            return {
                "status": "success",
                # "message": f"Single Atom Report written to {str(report_output_file)}",
                # "report_file": report_output_file,
                "data": report_output_file.read_text()
            }

        if "***        THE OPTIMIZATION HAS CONVERGED     ***" not in file_content:
            with open(report_output_file, 'w') as report:
                report.write(f"Optimization did not converge for: {report_output_file}\n")
            return {
                "status": "warning",
                "message": "Optimization did not converge",
                # "report_file": report_output_file
                "data": report_output_file.read_text()
            }
        
        post_convergence = file_content.split("***        THE OPTIMIZATION HAS CONVERGED     ***")[-1]
        
        with open(report_output_file, 'w') as report:
            report.write("Optimization Report\n")
            report.write("====================\n\n")
            report.write(f"Optimization converged for: {report_output_file}\n\n")
            
            for prop in properties:
                extractor = PROPERTY_EXTRACTORS[prop]
                flags = extractor.get('flags', 0)
                matches = re.findall(extractor['pattern'], post_convergence, flags)
                
                if matches:
                    formatted = extractor['formatter'](matches)
                    if formatted:
                        report.write(formatted + "\n")
        
        return {
            "status": "success",
            # "message": f"Optimization report written to {str(report_output_file)}",
            # "report_file": report_output_file,
            "data": report_output_file.read_text()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def extract_thermochemistry(text, properties=None):
    # 定义所有可用的性质及其提取模式
    all_properties = {
        'basic_info': {
            'temperature': r'Temperature\s+\.\.\.\s+([\d.]+)\s+K',
            'pressure': r'Pressure\s+\.\.\.\s+([\d.]+)\s+atm',
            'total_mass': r'Total Mass\s+\.\.\.\s+([\d.]+)\s+AMU',
            'point_group': r'Point Group:\s+(\w+)',
            'symmetry_number': r'Symmetry Number:\s+(\d+)'
        },
        'vibrational_frequencies': r'freq\.\s+([\d.]+)\s+E\(vib\)\s+\.\.\.\s+([\d.]+)',
        'rotational_constants': r'Rotational constants in cm-1:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        'electronic_energy': r'Electronic energy\s+\.\.\.\s+([-\d.]+)\s+Eh',
        'zero_point_energy': r'Zero point energy\s+\.\.\.\s+([\d.]+)\s+Eh\s+([\d.]+)\s+kcal/mol',
        'thermal_corrections': {
            'vibrational': r'Thermal vibrational correction\s+\.\.\.\s+([\d.]+)\s+Eh',
            'rotational': r'Thermal rotational correction\s+\.\.\.\s+([\d.]+)\s+Eh',
            'translational': r'Thermal translational correction\s+\.\.\.\s+([\d.]+)\s+Eh'
        },
        'total_thermal_energy': r'Total thermal energy\s+([-\d.]+)\s+Eh',
        'enthalpy': r'Total Enthalpy\s+\.\.\.\s+([-\d.]+)\s+Eh',
        'entropy_contributions': {
            'vibrational': r'Vibrational entropy\s+\.\.\.\s+([\d.]+)\s+Eh',
            'rotational': r'Rotational entropy\s+\.\.\.\s+([\d.]+)\s+Eh',
            'translational': r'Translational entropy\s+\.\.\.\s+([\d.]+)\s+Eh',
            'total': r'Final entropy term\s+\.\.\.\s+([\d.]+)\s+Eh'
        },
        'gibbs_free_energy': r'Final Gibbs free energy\s+\.\.\.\s+([-\d.]+)\s+Eh'
    }
    
    # 如果没有指定properties，则使用所有性质
    if properties is None:
        properties = list(all_properties.keys())
    # 如果输入的是单个字符串，转换为列表
    elif isinstance(properties, str):
        properties = [properties]
    
    # 初始化结果字典
    results = {}
    
    # 提取请求的性质
    for prop in properties:
        if prop not in all_properties:
            continue  # 跳过无效的性质名称
            
        if prop == 'basic_info':
            results[prop] = {}
            for key, pattern in all_properties[prop].items():
                match = re.search(pattern, text)
                if match:
                    results[prop][key] = match.group(1)
        
        elif prop == 'vibrational_frequencies':
            matches = re.findall(all_properties[prop], text)
            results[prop] = [{'frequency': m[0], 'energy': m[1]} for m in matches]
        
        elif prop == 'rotational_constants':
            match = re.search(all_properties[prop], text)
            if match:
                results[prop] = list(match.groups())
        
        elif prop in ['thermal_corrections', 'entropy_contributions']:
            results[prop] = {}
            for key, pattern in all_properties[prop].items():
                match = re.search(pattern, text)
                if match:
                    results[prop][key] = match.group(1)
        
        else:
            match = re.search(all_properties[prop], text)
            if match:
                results[prop] = match.group(1)
    
    return results

def format_thermochemistry_report(data, properties=None):
    # 如果没有指定properties，则使用数据中的所有性质
    if properties is None:
        properties = list(data.keys())
    # 如果输入的是单个字符串，转换为列表
    elif isinstance(properties, str):
        properties = [properties]
    
    report_lines = ["THERMOCHEMISTRY SUMMARY REPORT"]
    
    if 'basic_info' in properties and 'basic_info' in data:
        report_lines.extend([
            "",
            "Basic Information:",
            f"Temperature: {data['basic_info']['temperature']} K",
            f"Pressure: {data['basic_info']['pressure']} atm",
            f"Total Mass: {data['basic_info']['total_mass']} AMU",
            f"Point Group: {data['basic_info']['point_group']}",
            f"Symmetry Number: {data['basic_info']['symmetry_number']}"
        ])
    
    if 'vibrational_frequencies' in properties and 'vibrational_frequencies' in data:
        report_lines.extend([
            "",
            "Vibrational Frequencies (cm⁻¹):",
        ])
        for freq in data['vibrational_frequencies']:
            report_lines.append(f"- {freq['frequency']} cm⁻¹")
            # report_lines.append(f"  • {freq['frequency']} cm⁻¹ (Energy: {freq['energy']} Eh)")
    
    if 'rotational_constants' in properties and 'rotational_constants' in data:
        report_lines.extend([
            "",
            "Rotational Constants (cm⁻¹):",
        ])
        for i, const in enumerate(data['rotational_constants']):
            report_lines.append(f"  {const}")
    
    if 'electronic_energy' in properties and 'electronic_energy' in data:
        report_lines.extend([
            "",
            "Energy Components:",
            f"Electronic Energy: {data['electronic_energy']} Eh"
        ])
    
    if 'zero_point_energy' in properties and 'zero_point_energy' in data:
        if 'Energy Components:' not in report_lines:
            report_lines.extend(["", "Energy Components:"])
        report_lines.append(f"Zero Point Energy: {data['zero_point_energy']} Eh")
    
    if 'thermal_corrections' in properties and 'thermal_corrections' in data:
        report_lines.extend([
            "",
            "Thermal Corrections:",
            f"Vibrational: {data['thermal_corrections']['vibrational']} Eh",
            f"Rotational: {data['thermal_corrections']['rotational']} Eh",
            f"Translational: {data['thermal_corrections']['translational']} Eh"
        ])
    
    if 'total_thermal_energy' in properties and 'total_thermal_energy' in data:
        report_lines.extend([
            "",
            f"Total Thermal Energy: {data['total_thermal_energy']} Eh"
        ])
    
    if 'enthalpy' in properties and 'enthalpy' in data:
        report_lines.extend([
            "",
            "Thermodynamic Properties:",
            f"Enthalpy (H): {data['enthalpy']} Eh"
        ])
    
    if 'entropy_contributions' in properties and 'entropy_contributions' in data:
        report_lines.extend([
            "",
            "Entropy Contributions (T*S):",
            f"Vibrational: {data['entropy_contributions']['vibrational']} Eh",
            f"Rotational: {data['entropy_contributions']['rotational']} Eh",
            f"Translational: {data['entropy_contributions']['translational']} Eh",
            f"Total Entropy: {data['entropy_contributions']['total']} Eh"
        ])
    
    if 'gibbs_free_energy' in properties and 'gibbs_free_energy' in data:
        report_lines.extend([
            "",
            "Gibbs Free Energy (G):",
            f"Final Gibbs Free Energy: {data['gibbs_free_energy']} Eh"
        ])
    
    return "\n".join(report_lines)


@mcp.tool()
async def get_data_from_freq_output(
    orca_output_file: Path,
    properties: Optional[Union[str, List[str]]] = [
        "vibrational_frequencies",
        "total_thermal_energy",
        "enthalpy",
        "entropy_contributions",
        "gibbs_free_energy",
        ]
    ):
    """
    Get thermochemistry data from ORCA frequency output file and write a formatted report.

    Args:
        orca_output_file (Path): The path to the ORCA output file containing thermochemistry data.
        properties (list[str], optional): A list of specific properties to include in the report.
        Accepts 'basic_info', 'vibrational_frequencies', 'rotational_constants', 
                "electronic_energy", "zero_point_energy", "thermal_corrections", 
                "total_thermal_energy", "enthalpy", "entropy_contributions", 
                "gibbs_free_energy"

    Returns:
        dict: A dictionary containing the status of the operation and a message. 
    """
    report_file_name = "thermochemistry.txt"
    report_output_file = Path(report_file_name)
    try:
        # 打开ORCA输出文件并读取内容
        with open(orca_output_file, 'r', encoding='utf-8') as file:
            thermochemistry_text = file.read()

        # 提取热化学数据
        data = extract_thermochemistry(thermochemistry_text)

        # 格式化报告
        formatted_report = format_thermochemistry_report(data, properties)

        # 写入报告文件
        with open(report_output_file, 'w', encoding='utf-8') as report_file:
            report_file.write(formatted_report)

        return {
            "status": "success",
            # "message": f"Thermochemistry report written to {str(report_output_file)}",
            # "report_file": report_output_file,
            "report_content": formatted_report,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def parse_orca_ir(out_file):
    """从 ORCA 输出文件中解析 IR 频率和强度"""
    freqs, intensities = [], []
    in_block = False
    with open(out_file, 'r') as f:
        for line in f:
            if line.strip().startswith("IR SPECTRUM"):
                in_block = True
                continue
            if in_block and line.strip().startswith("* The epsilon"):
                break
            if in_block:
                match = re.match(
                    r"\s*\d+:\s+([0-9.]+)\s+[0-9.Ee+-]+\s+([0-9.Ee+-]+)",
                    line,
                )
                if match:
                    freqs.append(float(match.group(1)))
                    intensities.append(float(match.group(2)))
    return np.array(freqs), np.array(intensities)

def gaussian_broadening(freqs, intensities, x_range, resolution=1.0, width=10.0):
    """对 IR 数据进行高斯展宽，生成传统透射率谱样式"""
    x = np.arange(x_range[0], x_range[1], resolution)
    y = np.zeros_like(x)
    for f, I in zip(freqs, intensities):
        if x_range[0] <= f <= x_range[1]:
            y += I * np.exp(-(x - f)**2 / (2 * width**2))
    
    # 转换为透射率谱样式：将强度转换为透射率百分比
    # 归一化强度到0-1范围，然后转换为透射率百分比
    if np.max(y) > 0:
        normalized_y = y / np.max(y)
        # 转换为透射率：T = 100 * (1 - normalized_intensity)
        # 这样强吸收峰会显示为低透射率（向下），弱吸收峰显示为高透射率
        y = 100 * (1 - normalized_y)
    else:
        y = np.full_like(y, 100)  # 如果没有强度，设为100%透射率
    
    return x, y

EMPTY_PATH = Path("_EMPTY_")

@mcp.tool()
def plot_three_ir_spectra(
    out_file1: Path,
    out_file2: Path,
    out_file3: Path,
    x_range: List[float],
    save_path: Optional[str] = "ir_spectrum.png"
    ):
    """
    Plots the IR spectra from the provided 3 output files.

    This function reads IR data from the ORCA output files, applies Gaussian broadening,
    and generates a plot comparing the IR spectra. The plot is saved to the specified path.

    Parameters:

    out_file1 (Path): The path to the first ORCA output file.
    out_file2 (Path): The path to the second ORCA output file.
    out_file3 (Path): The path to the third ORCA output file.
    x_range (List[float]): A list containing the minimum and maximum wavenumber (cm^-1) for the x-axis.
    save_path (str): The file path where the plot will be saved. Default is "ir_spectrum.png".

    Returns:
    dict: A dictionary containing the status of the operation and a message. 
          If successful, it includes the path to the saved plot.
    """
    try:
        out_files = [out_file1, out_file2, out_file3]
        labels = [Path(f).stem for f in out_files]  # 使用文件名(不含扩展名)作为标签
        colors = list(mcolors.TABLEAU_COLORS.values())  # 使用预设颜色
        
        plt.figure(figsize=(10, 6), facecolor='white')
        
        for i, out_file in enumerate(out_files):
            try:
                freqs, intensities = parse_orca_ir(out_file)
                if len(freqs) == 0:
                    logging.warning(f"No IR data found in {out_file}. Skipping.")
                
                # 绘制高斯展宽曲线
                x, y = gaussian_broadening(freqs, intensities, x_range)
                plt.plot(x, y, color=colors[i % len(colors)], 
                        linewidth=2.0, label=labels[i], alpha=0.8)
                
                # 绘制棒图(自动调整透明度) - 透射率样式
                mask = (freqs >= x_range[0]) & (freqs <= x_range[1])
                if len(intensities[mask]) > 0:
                    # 对棒图也应用透射率转换
                    normalized_intensities = intensities[mask] / np.max(intensities[mask]) if np.max(intensities[mask]) > 0 else intensities[mask]
                    # 转换为透射率：T = 100 * (1 - normalized_intensity)
                    transmission_values = 100 * (1 - normalized_intensities)
                    # 棒图从100%透射率向下延伸
                    plt.vlines(freqs[mask], transmission_values, 100, 
                            color=colors[i % len(colors)], 
                            linestyle='-', linewidth=1.2, alpha=0.7)
                
            except Exception as e:
                print(f"处理文件 {out_file} 时出错: {e}")
                continue

        plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12, fontweight='bold')
        plt.ylabel("Transmission (%)", fontsize=12, fontweight='bold')
        plt.title("IR Spectra Comparison", fontsize=14, fontweight='bold', pad=20)
        plt.xlim(x_range[::-1])  # 反转x轴并设置范围
        plt.ylim(0, 100)  # 设置Y轴范围为0-100%
        
        # 美化图例
        plt.legend(frameon=True, fancybox=True, shadow=True, 
                  loc='upper right', fontsize=10, framealpha=0.9)
        
        # 美化网格
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3)
        
        # 美化坐标轴
        plt.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=6)
        plt.tick_params(axis='both', which='minor', width=0.8, length=3)
        
        # 设置背景色和边框
        ax = plt.gca()
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        return {
            "status": "success",
            "message": f"IR spectrum plot saved.",
            "plot_file": Path(save_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def plot_two_ir_spectra(
    out_file1: Path,
    out_file2: Path,
    x_range: List[float],
    save_path: Optional[str] = "ir_spectrum.png"
    ):
    """
    Plots the IR spectra from the provided 2 output files.

    This function reads IR data from the ORCA output files, applies Gaussian broadening,
    and generates a plot comparing the IR spectra. The plot is saved to the specified path.

    Parameters:

    out_file1 (Path): The path to the first ORCA output file.
    out_file2 ([Path]): The path to the second ORCA output file.
    x_range (List[float]): A list containing the minimum and maximum wavenumber (cm^-1) for the x-axis.
    save_path (str): The file path where the plot will be saved. Default is "ir_spectrum.png".

    Returns:

    dict: A dictionary containing the status of the operation and a message. 
          If successful, it includes the path to the saved plot.
    """
    try:
        # 自动生成标签和颜色
        out_files = [out_file1, out_file2]
        labels = [Path(f).stem for f in out_files]  # 使用文件名(不含扩展名)作为标签
        colors = list(mcolors.TABLEAU_COLORS.values())  # 使用预设颜色
        
        plt.figure(figsize=(10, 6), facecolor='white')
        
        for i, out_file in enumerate(out_files):
            try:
                freqs, intensities = parse_orca_ir(out_file)
                if len(freqs) == 0:
                    logging.warning(f"No IR data found in {out_file}. Skipping.")
                
                # 绘制高斯展宽曲线
                x, y = gaussian_broadening(freqs, intensities, x_range)
                plt.plot(x, y, color=colors[i % len(colors)], 
                        linewidth=2.0, label=labels[i], alpha=0.8)
                
                # 绘制棒图(自动调整透明度) - 透射率样式
                mask = (freqs >= x_range[0]) & (freqs <= x_range[1])
                if len(intensities[mask]) > 0:
                    # 对棒图也应用透射率转换
                    normalized_intensities = intensities[mask] / np.max(intensities[mask]) if np.max(intensities[mask]) > 0 else intensities[mask]
                    # 转换为透射率：T = 100 * (1 - normalized_intensity)
                    transmission_values = 100 * (1 - normalized_intensities)
                    # 棒图从100%透射率向下延伸
                    plt.vlines(freqs[mask], transmission_values, 100, 
                            color=colors[i % len(colors)], 
                            linestyle='-', linewidth=1.2, alpha=0.7)
                
            except Exception as e:
                print(f"处理文件 {out_file} 时出错: {e}")
                continue

        plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12, fontweight='bold')
        plt.ylabel("Transmission (%)", fontsize=12, fontweight='bold')
        plt.title("IR Spectra Comparison", fontsize=14, fontweight='bold', pad=20)
        plt.xlim(x_range[::-1])  # 反转x轴并设置范围
        plt.ylim(0, 100)  # 设置Y轴范围为0-100%
        
        # 美化图例
        plt.legend(frameon=True, fancybox=True, shadow=True, 
                  loc='upper right', fontsize=10, framealpha=0.9)
        
        # 美化网格
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3)
        
        # 美化坐标轴
        plt.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=6)
        plt.tick_params(axis='both', which='minor', width=0.8, length=3)
        
        # 设置背景色和边框
        ax = plt.gca()
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        return {
            "status": "success",
            "message": f"IR spectrum plot saved.",
            "plot_file": Path(save_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def plot_one_ir_spectra(
    out_file1: Path,
    x_range: List[float],
    save_path: Optional[str] = "ir_spectrum.png"
):
    """
    Plots the IR spectrum from a single output file.

    Parameters:
    out_file1 (Path): The path to the ORCA output file.
    x_range (List[float]): A list containing the minimum and maximum wavenumber (cm^-1) for the x-axis.
    save_path (str): The file path where the plot will be saved. Default is "ir_spectrum.png".

    Returns:
    dict: A dictionary containing the status of the operation and a message. 
          If successful, it includes the path to the saved plot.
    """
    try:
        out_files = [out_file1]
        labels = [Path(out_file1).stem]  # 使用文件名(不含扩展名)作为标签
        colors = list(mcolors.TABLEAU_COLORS.values())  # 使用预设颜色
        
        plt.figure(figsize=(10, 6), facecolor='white')
        
        for i, out_file in enumerate(out_files):
            try:
                freqs, intensities = parse_orca_ir(out_file)
                if len(freqs) == 0:
                    logging.warning(f"No IR data found in {out_file}. Skipping.")
                
                # 绘制高斯展宽曲线
                x, y = gaussian_broadening(freqs, intensities, x_range)
                plt.plot(x, y, color=colors[i % len(colors)], 
                        linewidth=2.0, label=labels[i], alpha=0.8)
                
                # 绘制棒图(自动调整透明度) - 透射率样式
                mask = (freqs >= x_range[0]) & (freqs <= x_range[1])
                if len(intensities[mask]) > 0:
                    # 对棒图也应用透射率转换
                    normalized_intensities = intensities[mask] / np.max(intensities[mask]) if np.max(intensities[mask]) > 0 else intensities[mask]
                    # 转换为透射率：T = 100 * (1 - normalized_intensity)
                    transmission_values = 100 * (1 - normalized_intensities)
                    # 棒图从100%透射率向下延伸
                    plt.vlines(freqs[mask], transmission_values, 100, 
                            color=colors[i % len(colors)], 
                            linestyle='-', linewidth=1.2, alpha=0.7)
                
            except Exception as e:
                print(f"处理文件 {out_file} 时出错: {e}")
                continue

        plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12, fontweight='bold')
        plt.ylabel("Transmission (%)", fontsize=12, fontweight='bold')
        plt.title("IR Spectrum", fontsize=14, fontweight='bold', pad=20)
        plt.xlim(x_range[::-1])  # 反转x轴并设置范围
        plt.ylim(0, 100)  # 设置Y轴范围为0-100%
        
        # 美化图例
        plt.legend(frameon=True, fancybox=True, shadow=True, 
                  loc='upper right', fontsize=10, framealpha=0.9)
        
        # 美化网格
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3)
        
        # 美化坐标轴
        plt.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=6)
        plt.tick_params(axis='both', which='minor', width=0.8, length=3)
        
        # 设置背景色和边框
        ax = plt.gca()
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        return {
            "status": "success",
            "message": f"IR spectrum plot saved.",
            "plot_file": Path(save_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def convert_xyz_to_molstar_html(input_xyz_file: Path):
    """
    读取 .xyz 文件，将其转换为 Mol* HTML 格式.。

    参数:
    input_xyz_file (Path): 输入的 .xyz 文件路径。

    返回:
    Path: 生成的 HTML 文件路径。如果转换失败，则返回 None。
    """
    try:
        atoms = read(input_xyz_file)

        # 为了生成 Mol* HTML，我们需要一个 PDB 格式的数据。
        # 这里我们使用一个临时文件来存储 PDB 数据，这样就不需要在磁盘上留下多余的文件。
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_pdb_file:
            temp_pdb_path = temp_pdb_file.name
            # 使用 ase.io.write() 将 Atoms 对象写入临时 PDB 文件
            write(temp_pdb_path, atoms)
            temp_pdb_file.seek(0)
            pdb_data = temp_pdb_file.read().decode('utf-8')

        builder = mvs.create_builder()
        (
            builder.download(url='data.pdb')
            .parse(format='pdb')
            .model_structure()
            .component()
            .representation()
        )
        html_content = builder.molstar_html(data={'data.pdb': pdb_data})

        output_html_file = os.path.splitext(input_xyz_file)[0] + ".html"
        with open(output_html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return {
            "status": "success",
            "html_file": Path(output_html_file)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to convert XYZ to Mol* HTML: {str(e)}"
        }


@mcp.tool()
async def write_xyz_file(
    xyz_content: str,
    mol_name: str,
    ) -> Tuple[Path, str]:
    """
    Write an XYZ file from the given content.

    Parameters:
    ----------
    xyz_content : str
        The content of the XYZ file. For example,
        ```ch4.xyz example
        4
        comment line
        C   -4.27036      0.369      -0.190699
        H   -3.23418      0.131       0.070996
        H   -4.78845     -0.212      -0.959517
        H   -4.78845      1.1904      0.316394

        ```
    mol_name : str
        The name of the molecule (used for naming the output file).
    Returns:
    -------
    tuple
        A tuple containing the path to the generated XYZ file and a message indicating success or failure.
    """
    try:
        xyz_path = Path(f"{mol_name}.xyz")
        with open(xyz_path, "w") as f:
            f.write(xyz_content.strip())
            f.write("\n\n")
        return {
            "xyz_file": xyz_path,
            "message": f"Successfully wrote XYZ file: {xyz_path}"
        }
    except Exception as e:
        logging.error(f"Error writing XYZ file: {e}")
        return {
            "xyz_file": None,
            "message": f"Error writing XYZ file: {str(e)}"
        }


@mcp.tool()
async def smiles_to_xyz(
    smiles: str,
    mol_name: str,
    charge: int = 0,
    multiplicity: int = 1,
    n_cores: int = 4,
    ) -> Tuple[Path, str]:
    """
    Convert a SMILES string to an XYZ file.

    Parameters:
    ----------
    smiles : str
        The SMILES string to convert.
    mol_name : str
        The name of the molecule (used for naming the output file).
    charge : int, optional
        The charge of the molecule (default is 0).
    multiplicity : int, optional
        The multiplicity of the molecule (default is 1).
    n_cores : int
        The number of CPU cores to use for the calculation (default is 4).

    Returns:
    -------
    tuple
        A tuple containing the path to the generated XYZ file and a message indicating success or failure.
    """
    ade.Config.n_cores = n_cores
    
    try:
        mol = ade.Molecule(smiles=smiles, charge=charge, mult=multiplicity)
        mol.optimise(method=ade.methods.XTB())
        xyz_path = Path(f"{mol.name}.xyz")
        mol.print_xyz_file(filename=str(xyz_path))
        return {
            "xyz_file": xyz_path,
            "message": f"Successfully converted SMILES to XYZ."
        }
    except Exception as e:
        logging.error(f"Error converting SMILES to XYZ: {e}")
        return {
            "xyz_file": None,
            "message": f"Error converting SMILES to XYZ: {str(e)}"
        }


class OrcaResult(TypedDict):
    """Result structure for ORCA calculation"""
    output_file: Path
    gbw_file: Path
    mol_file: Path
    report_file: Path
    message: str

add_keywords = f"""%output
Print[ P_Basis ] 2
Print[ P_MOs ] 1
Print[ P_Mulliken ] 1
Print[ P_Loewdin ] 1
Print[ P_Hirshfeld ] 1
end\n"""


@mcp.tool()
async def run_orca_calculation(
    keywords: str,
    mol_xyz: Path,
    additional_keywords: Optional[str] = "\n",
    charge: int = 0,
    multiplicity: int = 1,
    memory: int = 4000,
    nprocs: int = 4
) -> OrcaResult:
    """
    Run an ORCA calculation.

    This function prepares the input for an ORCA calculation, executes the calculation,
    and retrieves the results.

    Args:
        keywords (str): The content of the ORCA input file. 
            Examples:
                "!B3LYP D4 DEF2-TZVP OPT FREQ"
                "!B97M-V DEF2-SVP CPCM(WATER)"
                "!PBE0 D4 X2C X2C-TZVPALL"
                "!B3LYP DEF2-TZVP D4 OPT FREQ CPCM(WATER)"

        mol_xyz (Path): The path to the XYZ coordinates file of the molecular structure.
                        必须使用oss格式的URI进行传递(格式形如https://xxx),不能使用文件名.
                        可能需要根据上一步的任务来确定URL.
        additional_keywords (Optional[str]): Additional keywords to be added to the ORCA input file.
            Examples:
            When need print charge data in ORCA output.
            "%output
                Print[ P_Basis ] 2
                Print[ P_MOs ] 1
                Print[ P_Mulliken ] 1
                Print[ P_Loewdin ] 1
                Print[ P_Hirshfeld ] 1
            end"
            When need run tddft calculation.
            "%tddft
            NRoots 20      # Calculate 20 excited states (adjust as needed)
            TDA true       # Use Tamm-Dancoff approximation for stability
            Triplets true  # Calculate triplet states in addition to singlets
            MaxDim 100     # Set Davidson expansion space (5*NRoots)
            ETol 1e-6      # Tight energy convergence
            RTol 1e-6      # Tight residual convergence
            DoNTO true     # Generate Natural Transition Orbitals for analysis
            NTOThresh 1e-4 # Threshold for printing NTO occupation numbers
            end"

            When need run solvent calculation.
            "%cpcm
                epsilon 78.39
                rsolv 1.4
                surfacetype vdw_gaussian
            end"

        charge (int): The charge of the molecule, default is 0.
        multiplicity (int): The multiplicity of the molecule, default is 1.
        memory (int): The amount of memory (in MB) allocated for ORCA, default is 1000 MB.
        nprocs (int): The number of processors to be used by ORCA, default is 4.
                     **注意**: 计算单原子时必须设置为1.

    Returns:
        OrcaResult: A dictionary containing:
            - output_dir (Path): The path to the ORCA calculation output directory.
            - output_file (Path): The path to the ORCA output file.
            - gbw_file (Path): The path to the ORCA GBW file.
            - mol_file (Path): The path to the molecular structure file.
            - report_file (Path): The path to the ORCA report file.
            - message (str): Success or error message.
    """
    try:
        work_dir = Path(f"orca_calc_{int(os.path.getctime('.'))}")
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        orca_input = work_dir / "calc.inp"
        with open(orca_input, "w") as f:
            f.write(f"{keywords}\n")
            f.write(f"%maxcore {memory}\n")
            f.write(f"%pal nprocs {nprocs} end\n")
            f.write(additional_keywords)
            # f.write(add_keywords)
            f.write(f"* XYZFILE {charge} {multiplicity} ../{mol_xyz}\n")

        cmd = "/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411/orca calc.inp > calc.out"
        logging.info(f"Running ORCA: {cmd} (cwd={work_dir})")

        process = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            env=os.environ.copy(),  # 显式传递当前环境变量
        )

        if process.returncode != 0:
            raise RuntimeError(f"ORCA calculation failed: {process.stderr}")



        return OrcaResult(
            output_dir=work_dir,
            output_file=work_dir / "calc.out",
            gbw_file=work_dir / "calc.gbw",
            mol_file=work_dir / "mol.xyz",
            message="ORCA calculation completed successfully."
        )
        
    except Exception as e:
        logging.error(f"Error running ORCA calculation: {e}")
        return OrcaResult(
            output_file="",
            gbw_file="",
            mol_file="",
            output_dir=work_dir,
            message=f"ORCA calculation failed: {e}"
        )


@mcp.tool()
async def packmol_merge(
    solute_file_path: Path,
    solvent_file_path: Path,
    num_solvent_molecules: int,
    output_file_name: str = "mol_with_solvent.xyz",
    tolerance: float = 2.0,
    sphere_radius: float = 8.0,
) -> dict:
    """
    Merge solute and solvent molecules using Packmol.

    Parameters:
    ----------
    solute_file_path : Path
        Path to the solute molecule XYZ file.
    solvent_file_path : Path
        Path to the solvent molecule XYZ file.
    num_solvent_molecules : int
        Number of solvent molecules to add.
    output_file_name : str, optional
        Name of the output merged XYZ file (default is "mol_with_solvent.xyz").
    tolerance : float, optional
        Packmol tolerance (default is 2.0).
    sphere_radius : float, optional
        Radius of the sphere around the solute for solvent placement (default is 8.0).

    Returns:
    -------
    Path
        Path to the merged XYZ file.
    """
    try:
        # Create packmol.inp content
        packmol_inp_content = f"""tolerance {tolerance}
filetype xyz
output {output_file_name}

# Solute molecule, placed at the center of the box
structure {solute_file_path}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

# Solvent molecules, randomly placed in a spherical shell around the solute
structure {solvent_file_path}
  number {num_solvent_molecules}
  inside sphere 0. 0. 0. {sphere_radius}
end structure
"""
        # Write packmol.inp file
        packmol_inp_path = Path("packmol.inp")
        with open(packmol_inp_path, "w") as f:
            f.write(packmol_inp_content)

        # Copy solute and solvent files to current directory for packmol to find them
        # This assumes the files are accessible in the current working directory or a path relative to it.
        # For a more robust solution, consider handling absolute paths or a temporary directory.
        # For now, we'll assume the input paths are relative to the current working directory
        # or that packmol can find them if they are absolute.
        # If the files are not in the current directory, packmol will fail.
        # A better approach would be to copy them to a temporary directory where packmol.inp is created.
        # However, for this example, we'll assume they are accessible.

        # Execute packmol command
        cmd = f"packmol < {packmol_inp_path}"
        logging.info(f"Running Packmol: {cmd}")

        # Ensure solute and solvent files are in the current directory or accessible by packmol
        # For simplicity, we'll assume they are already in the current directory or their paths are absolute.
        # In a real scenario, you might want to copy them to a temporary directory.
        
        # Check if solute and solvent files exist in the current directory
        if not solute_file_path.exists():
            raise FileNotFoundError(f"Solute file not found: {solute_file_path}")
        if not solvent_file_path.exists():
            raise FileNotFoundError(f"Solvent file not found: {solvent_file_path}")

        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        # opt by xtb
        cmd = f"xtb {output_file_name} --opt"
        logging.info(f"Running XTB optimization: {cmd}")
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        output_file_name = "xtbopt.xyz"


        if process.returncode != 0:
            raise RuntimeError(f"Packmol failed: {process.stderr}")

        output_path = Path(output_file_name)
        if not output_path.exists():
            raise FileNotFoundError(f"Packmol output file not found: {output_file_name}")

        return {
            "merged_file": output_path,
            "message": f"Successfully merged."
        }

    except Exception as e:
        logging.error(f"Error merging molecules with Packmol: {e}")
        return {
            "merged_file": None,
            "message": f"Error merging molecules with Packmol: {str(e)}"
        }



class ReactionProfileResult(TypedDict):
    reaction_profile: Path
    energies_csv: Path
    methods: Path
    reactants: List[Path]
    products: List[Path]
    transition_states: List[Path]
    imaginary_modes: List[Path]
    message: str


def _run_calculation_sync(
    reactant_smiles: list[str],
    product_smiles: list[str],
    solvent_name: Optional[str],
    n_cores: int,
    energy_type: str,
    calculation_level: str,
    single_point_refinement: bool,
    with_complexes: bool,
) -> ReactionProfileResult:
    """
    Synchronous implementation of reaction profile calculation.
    
    Args:
        reactant_smiles: List of SMILES strings for reactants
        product_smiles: List of SMILES strings for products
        solvent_name: Name of solvent to use
        n_cores: Number of CPU cores to use
        energy_type: Type of energy to calculate (potential, enthalpy, free_energy)
        calculation_level: Level of theory (low, medium, high)
        single_point_refinement: Whether to perform single point refinement
        with_complexes: Whether to include reactant/product complexes
    
    Returns:
        Dictionary with calculation results
    """
    # Set configuration based on calculation level
    ade.Config.n_cores = n_cores
    
    # Configure calculation methods
    if calculation_level == "low":
        # Use xtb for low level calculations
        ade.Config.lcode = "xtb"
    elif calculation_level == "medium":
        # Use DFT with medium basis set
        ade.Config.hcode = "ORCA"
        ade.Config.ORCA.keywords.set_opt_basis_set('def2-SVP')
        ade.Config.ORCA.keywords.sp.basis_set = 'def2-TZVP'
    else:  # high
        # Use DFT with high level basis set
        ade.Config.hcode = "ORCA"
        ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
        ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'
    
    # Parse reactants and products
    reacts = [ade.Reactant(smiles=smile) for smile in reactant_smiles]
    prods = [ade.Product(smiles=smile) for smile in product_smiles]
    
    # Create reaction
    rxn = ade.Reaction(
        *reacts, *prods, 
        name="reaction", 
        solvent_name=solvent_name
    )
    
    # Determine energy calculation parameters
    free_energy = energy_type == "free_energy"
    enthalpy = energy_type == "enthalpy"
    
    # Calculate reaction profile
    rxn.calculate_reaction_profile(
        units="kcal mol-1",
        single_point_refinement=single_point_refinement,
        with_complexes=with_complexes,
        free_energy=free_energy,
        enthalpy=enthalpy
    )
    
    # Collect output files
    output_dir = Path("reaction/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all transition state files
    ts_files = list(output_dir.glob("TS_*.xyz"))
    # Filter out imaginary mode file from TS files
    imag_mode_files = [f for f in ts_files if "imag_mode" in f.name]
    ts_files = [f for f in ts_files if f not in imag_mode_files]
    
    # # If not returning all TS, keep only the lowest energy one
    # if not return_all_transition_states and ts_files:
    #     # Sort by energy (assuming filenames indicate energy)
    #     ts_files = [min(ts_files, key=lambda x: float(x.stem.split('_')[1]))]
    
    return {
        "reaction_profile": Path("reaction_profile.png"),
        "energies_csv": output_dir / "energies.csv",
        "methods": output_dir / "methods.txt",
        # "reactants": [Path(f"{r.name}.xyz") for r in reacts],
        # "products": [Path(f"{p.name}.xyz") for p in prods],
        "transition_states": Path(ts_files[0]),
        "imaginary_modes": Path(imag_mode_files[0]),
        "message": "Reaction profile calculated successfully."
    }


@mcp.tool()
async def calculate_reaction_profile(
    reactant_smiles: list[str] = ["C=CC=C", "C=C"], 
    product_smiles: list[str] = ["C1=CCCCC1"],
    solvent_name: Optional[str] = None,
    n_cores: int = 2,
    energy_type: Literal["potential", "enthalpy", "free_energy"] = "free_energy",
    calculation_level: Literal["low", "medium", "high"] = "medium",
    single_point_refinement: bool = True,
    with_complexes: bool = False,
) -> ReactionProfileResult:
    """
    Calculate the reaction profile for a given set of reactants and products with enhanced control over calculation parameters.
    
    This implementation extends the basic reaction profile calculation with:
    - Different energy types (potential, enthalpy, free energy)
    - Configurable calculation levels (low, medium, high)
    - Option to include reactant/product complexes
    - Option to return all transition states or just the lowest energy one
    
    Parameters:
    ----------
    reactant_smiles : list[str]
        A list of SMILES strings representing the reactants.
    product_smiles : list[str]
        A list of SMILES strings representing the products.
    solvent_name : Optional[str], optional
        The name of the solvent to be used in the reaction (default is None).
    n_cores : int
        The number of CPU cores to use for the calculation (default is 4).
    energy_type : Literal["potential", "enthalpy", "free_energy"]
        Type of energy to calculate for the profile (default is "potential").
    calculation_level : Literal["low", "medium", "high"]
        Level of theory for the calculation (default is "medium").
    single_point_refinement : bool
        Whether to perform single-point energy calculations at a higher level (default is True).
    with_complexes : bool
        Whether to include reactant and product complexes in the profile (default is False).

    
    Returns:
    -------
    ReactionProfileResult
        Dictionary containing paths to generated files and calculation status.
        Includes:
        - reaction_profile: Path to the reaction profile visualization
        - energies_csv: Path to energy data in CSV format
        - methods: Path to file describing computational methods used
        - transition_states: Path to transition state XYZ files
        - imaginary_modes: Path to imaginary mode XYZ files
        - message: Status message of the calculation
    """
    try:
        # Run the calculation in a separate thread to avoid blocking the event loop
        result = await anyio.to_thread.run_sync(
            _run_calculation_sync,
            reactant_smiles,
            product_smiles,
            solvent_name,
            n_cores,
            energy_type,
            calculation_level,
            single_point_refinement,
            with_complexes,
        )
        return result
    except Exception as e:
        logging.error(f"Error calculating reaction profile: {e}")
        return {
            "reaction_profile": None,
            "energies_csv": None,
            "methods": None,
            "transition_states": None,
            "imaginary_modes": None,
            "message": f"Error calculating reaction profile: {str(e)}"
        }


top_n = 3
max_assembly_length = 2000
vector_stores = ["orca_manual"]

embeddings = DashScopeEmbeddings(model="text-embedding-v4")

orca_vector_store = Chroma(
    persist_directory="./vector_db_orca_manual_qwen",
    embedding_function=embeddings
)


class RetrieveContentResult(TypedDict):
    """Retrieve content result."""
    status: str
    retrieved_content: Optional[list[dict[str, Any]]] = None


@mcp.tool()
async def retrieve_content_from_docs(
    query: str,
    vector_store_name: str = "orca_manual",
) -> RetrieveContentResult:
    """
    Retrieve relevant content from documents based on a query.
    
    Args:
        query: The search query
        vector_store_name: Name of the vector store to search in.
         Available options: "orca_manual",
    
    Returns:
        Retrieved content with metadata
    """
    try:

        if vector_store_name not in vector_stores:
            raise ValueError(
                f"Unsupported vector store: {vector_store_name}. Available options: {vector_stores}"
                )


        if vector_store_name == "orca_manual":
            vector_store = orca_vector_store

        final_docs = vector_store.similarity_search(query, k=top_n)
        
        retrieved_content = []
        total_length = 0
        
        for i, doc in enumerate(final_docs):
            content = doc.page_content


            retrieved_content.append({
                "content": content,
                "metadata": doc.metadata,
            })
            
            total_length += len(content)
        
        return RetrieveContentResult(
            status="success",
            retrieved_content=retrieved_content
        )
    except Exception as e:
        logging.error(f"Error retrieving content: {e}")
        return RetrieveContentResult(
            status="error",
            retrieved_content=None
        )


if __name__ == "__main__":
    logging.info("Starting MOLPILOT MCP Server with all tools...")
    mcp.run(transport="sse")
 