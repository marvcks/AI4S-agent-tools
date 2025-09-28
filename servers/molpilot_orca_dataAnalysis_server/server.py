#!/usr/bin/env python3
"""
Data Analysis Server for ORCA
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

from dp.agent.server import CalculationMCPServer

import anyio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import molviewspec as mvs
from ase.io import read, write


def parse_args():
    parser = argparse.ArgumentParser(description="Data Analysis Server")
    parser.add_argument('--port', type=int, default=50007, help='Server port (default: 50007)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50007
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("dataAnalysis_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)

@mcp.tool()
async def get_data_from_orca_opt_output(
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
async def get_data_from_orca_freq_output(
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


if __name__ == "__main__":
    logging.info("Starting Data Analysis Server with all tools...")
    mcp.run(transport="sse")
 