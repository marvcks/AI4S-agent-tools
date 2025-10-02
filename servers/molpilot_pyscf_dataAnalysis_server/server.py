#!/usr/bin/env python3
"""
PySCF MCP服务器
"""
import os
import sys
import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import json
import molviewspec as mvs
from ase.io import read, write
import requests
import numpy as np
import urllib.parse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# 导入MCP相关
from dp.agent.server import CalculationMCPServer

# 导入AutoDE相关
import autode as ade
import anyio

# Logging module
import loguru

from safety_check import is_safe

import matplotlib.pyplot as plt

import nanoid

# 设置环境变量
os.environ["AUTODE_LOG_LEVEL"] = "INFO"
os.environ["AUTODE_LOG_FILE"] = "autode.log"
os.environ["MCP_SCRATCH"] = "/tmp"
os.environ['OMP_NUM_THREADS'] = "2" 


CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY", "")
MCP_SCRATCH = os.getenv("MCP_SCRATCH", "/home/zhouoh/scratch")

# --- 1. 配置您的 R2 凭证和信息 ---
# 替换成您自己的信息
ACCOUNT_ID = ''
ACCESS_KEY_ID = ''
SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'mcp'

# 构建 Endpoint URL
ENDPOINT_URL = f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com'

# --- 2. 创建 S3 客户端 ---
# 我们使用 boto3 的 S3 客户端，但将其指向 R2 的 endpoint
s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    region_name='auto'  # 对于 R2，region 通常设置为 'auto'
)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MOLPILOT PySCF Data analysis MCP服务器")
    parser.add_argument('--port', type=int, default=50012, help='服务器端口 (默认: 50012)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机 (默认: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50012
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("molpilot_pyscf_dataAnalysis_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_pyscf_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"Molpilt PySCF Data Analysis MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")

    
def Gaussian_Expansion(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def Lorentzian_Expansion(x, mu, gamma):
    return (gamma / np.pi) / ((x - mu) ** 2 + gamma ** 2)
    

def _plot_spectrum(npz_file: Path, spectrum_type: Literal["IR", "UV-Vis", "NMR"]) :
    """
    Plot and save the IR or UV-Vis spectrum from a .npz file.

    Parameters:
    ----------
    npz_file : Path
        The path to the .npz file containing the spectrum data.
    spectrum_type : str
        The type of spectrum to plot. Must be either "IR", "UV-Vis", or "NMR".
    output_file : Optional[Path]
        The path to save the output plot image. If None, defaults to 'spectrum.png'.

    Returns:
    -------
    dict
        A dictionary containing the status and the path to the saved plot image.
    """

    data = np.load(npz_file)
    x = data['x']
    y = data['y']
    
    if spectrum_type == "IR":
        num_peaks = len(x)
        intensities = np.zeros(1000)
        frequencies = np.linspace(0, max(x) * 1.2, 1000)
        for i in range(num_peaks):
            intensities += Lorentzian_Expansion(frequencies, x[i], 10) * y[i]
        plt.plot(frequencies, intensities)
    
    elif spectrum_type == "UV-Vis":
        x = 1240 / x  # Convert eV to nm
        wavelengths = np.linspace(0, max(x) * 1.2, 1000)
        oscillator_strengths = np.zeros(1000)
        num_peaks = len(x)
        for i in range(num_peaks):
            oscillator_strengths += Gaussian_Expansion(wavelengths, x[i], 10) * y[i]
        plt.plot(wavelengths, oscillator_strengths)
    elif spectrum_type == "NMR":
        num_peaks = len(x)
        intensities = np.zeros(1000)
        shifts = np.linspace(0, max(x) * 1.2, 1000)
        for i in range(num_peaks):
            intensities += Lorentzian_Expansion(shifts, x[i], 0.5) * y[i]
        plt.plot(shifts, intensities)
    return

        
@mcp.tool()
def plot_spectrum(npz_file: List[Path], spectrum_type: Literal["IR", "UV-Vis", "NMR"], output_file: str) -> Dict[str, Any]:
    """
    Plot and save the IR or UV-Vis spectrum from a .npz file.

    Parameters:
    ----------
    npz_file : List[Path]
        The path to the .npz file containing the spectrum data. Could be frequency_file or tddft_file or nmr_file. Could be a list of files for plotting multiple spectra together.
    spectrum_type : str
        The type of spectrum to plot. Must be either "IR", "UV-Vis", or "NMR".
    output_file : str
        The path to save the output plot image. If None, defaults to 'spectrum.png'.

    Returns:
    -------
    dict
        A dictionary containing the status and the path to the saved plot image.
    """
    if spectrum_type not in ["IR", "UV-Vis", "NMR"]:
        return {
            "status": "error",
            "message": f"Invalid spectrum type: {spectrum_type}. Must be 'IR', 'UV-Vis', or 'NMR'."
        }

    try:
        plt.figure(figsize=(8, 6))
        for file in npz_file:
            data = np.load(file)
            
            if spectrum_type == "IR":
                _plot_spectrum(file, "IR")
            elif spectrum_type == "UV-Vis":
                _plot_spectrum(file, "UV-Vis")
            elif spectrum_type == "NMR":
                _plot_spectrum(file, "NMR")

        plt.xlabel({
            "IR": "Wavenumber (cm$^{-1}$)",
            "UV-Vis": "Wavelength (nm)",
            "NMR": "Chemical Shift (ppm)"
        }[spectrum_type])
        plt.ylabel({
            "IR": "Intensity (a.u.)",
            "UV-Vis": "Oscillator Strength (a.u.)",
            "NMR": "Intensity (a.u.)"
        }[spectrum_type])
        plt.title({
            "IR": "Infrared (IR) Spectrum",
            "UV-Vis": "UV-Visible Spectrum",
            "NMR": "NMR Spectrum"
        }[spectrum_type])
        if spectrum_type == "NMR":
            plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        if output_file is None:
            output_file = f"{MCP_SCRATCH}/spectrum.png"
        output_file = f"{MCP_SCRATCH}/{output_file}"
        plt.savefig(output_file, dpi=300)
        plt.close()
        return {
            "status": "success",
            "message": f"Spectrum plot saved to {output_file}",
            "plot_file": output_file
        }
    except Exception as e:
        logger.error(f"Error plotting spectrum: {e}")
        return {
            "status": "error",
            "message": f"Failed to plot spectrum: {str(e)}"
        }

@mcp.tool()
def execute_python(code: str) -> Dict[str, Any]:
    """
    Execute arbitrary Python code in a restricted environment.

    Parameters:
    ----------
    code : str
        The Python code to execute. The stdout and stderr will be captured and returned.
        Example:
        "a = 5\nb = 10\nprint('Sum:', a + b)"

    Returns:
    -------
    dict
        A dictionary containing the execution status and any output or error messages.
    """
    try:
        # 1. 静态代码分析
        if not is_safe(code, logger):
            logger.error("Generated code failed the safety check.")
            return {"error": "Generated code failed the safety check. Please ensure it does not contain unsafe operations like file I/O, network access, or execution of arbitrary system commands."}

        id = nanoid.generate(size=10)

        with open(f"exec_{id}.py", "w") as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, f"exec_{id}.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        os.remove(f"exec_{id}.py")
        if result.returncode != 0:
            logger.error(f"Error executing code: {result.stderr}")
            return {
                "status": "error",
                "message": f"Error executing code: {result.stderr}"
            }
        else:
            logger.info(f"Code executed successfully: {result.stdout}")
            return {
                "status": "success",
                "output": result.stdout
            }

    except Exception as e:
        logger.error(f"Error executing Python code: {e}")
        return {
            "status": "error",
            "message": f"Error executing code: {str(e)}"
        }
    
@mcp.tool()
def local_file_to_r2_url(local_file: str) -> Dict[str, str]:
    """
    Upload a local file to Cloudflare R2 and return the public URL.

    Parameters:
    ----------
    local_file : str
        The path to the local file to upload.

    Returns:
    -------
    dict
        A dictionary containing the status and the public URL of the uploaded file or an error message.
    """
    try:
        if not os.path.isfile(local_file):
            return {
                "status": "error",
                "message": f"Local file does not exist: {local_file}"
            }
        
        if "file://" in local_file:
            local_file = local_file.replace("file://", "")
        if "local://" in local_file:
            local_file = local_file.replace("local://", "")

        file_name = os.path.basename(local_file)
        s3_client.upload_file(local_file, BUCKET_NAME, file_name, ExtraArgs={'ACL': 'public-read'})
        #https://pyscftoolmcp.cc/1-A.inp
        public_url = f"https://pyscftoolmcp.cc/{file_name}"

        return {
            "status": "success",
            "url": public_url
        }
    except (NoCredentialsError, ClientError) as e:
        logger.error(f"Error uploading file to R2: {e}")
        return {
            "status": "error",
            "message": f"Error uploading file to R2: {str(e)}"
        }


if __name__ == "__main__":
    logger.info("Starting the MCP server...")
    mcp.run(transport="sse")
 