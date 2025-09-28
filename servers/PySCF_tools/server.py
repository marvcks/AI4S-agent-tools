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

from handler import (
    SinglePointHandler,
    GeometryOptimizationHandler,
    FrequencyAnalysisHandler,
    PropAnalysisHandler,
    NMRHandler,
)

from safety_check import is_safe

import matplotlib.pyplot as plt

import nanoid

# 设置环境变量
os.environ["AUTODE_LOG_LEVEL"] = "INFO"
os.environ["AUTODE_LOG_FILE"] = "autode.log"
os.environ["MCP_SCRATCH"] = "/home/zhouoh/scratch"
os.environ['OMP_NUM_THREADS'] = "12" 


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
mcp = CalculationMCPServer("molpilot_server_pyscf", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_pyscf_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"PySCF_tools MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")

JOB_LIST_REGISTRY = {
    "single_point": SinglePointHandler,
    "geometry_optimization": GeometryOptimizationHandler,
    "frequency_analysis": FrequencyAnalysisHandler,
    "population_analysis": PropAnalysisHandler,
    "NMR": NMRHandler,
}


@mcp.tool()
def process_job(job_data: str) -> str:
    """
    Run a quantum chemistry job using PySCF based on the provided job data.
    Args:
        job_data (str): JSON string containing job specifications.
            Examples of job specifications:
            {
                "job_type": "single_point", // MUST BE ONE OF "single_point", "geometry_optimization", "frequency_analysis", "population_analysis", "NMR".
                
                "molecule": "/path/to/molecule/file", //Path to the molecule XYZ file.
                "parameters": {
                    "method": "RHF", // e.g., "B3LYP", "PBE0", "MP2", "HF", etc. Can be None for default (b3lyp-d3bj)
                    "basis": "ccpvdz", // e.g., "sto-3g", "6-31g", "ccpvdz", etc. Can be None for default (def2-svp)
                    "solvent": "water", // e.g., "water", "ethanol", etc. Can be None for gas phase.
                    "solvent_model": "PCM", // e.g., "PCM", "SMD", etc. Can be None for default (PCM)
                    "solvent_dielectric": 78.4, // Dielectric constant for the solvent. required if PCM is used.
                    "tddft": false, // Whether to perform TDDFT calculation.
                    "charge": 0, // Molecular charge. Default is 0.
                    "multiplicity": 1, // Spin multiplicity. Default is 1 (singlet).
                    "max_iterations": 100, // Maximum SCF iterations. Default is 50
                    "convergence": 1e-6, // SCF convergence threshold. Default is 1e-6
                    "optimization_threshold": "normal", // e.g., "loose", "normal", "tight". Only for geometry_optimization jobs. Default is "normal".
                    "max_optimization_steps": 100, // Maximum optimization steps. Only for geometry_optimization jobs. Default is 100.
                    "population_analysis_method": "Mulliken", // e.g., "Mulliken", or "Hirshfeld". Only for population_analysis jobs. Default is "Mulliken".
                    "population_properties": ["charges", "dipole", "orbitals"], // List of properties to calculate. Options include "charges", "dipole", "orbitals".
                    "n_states": 5, // Number of excited states to calculate. Only for TDDFT jobs. Can be None for default (5).
                    "i_state": 1, // Interested excited state index (1-based). Only for TDDFT jobs. Can be None for default (1).
                    "triplet": false, // Whether to calculate triplet states. Only for TDDFT jobs. Default is false.
                    "Temperature": 298.15, // Temperature in Kelvin for thermochemical analysis. Only for frequency_analysis jobs.
                    "Pressure": 101325, // Pressure in Pa for thermochemical analysis. Only for
                }
            }
    Returns:
        str: JSON string containing job results or error messages.
            Example of successful result:
            {
                "energy": -76.026760737428, // Total energy in Hartree (if applicable)
                "optimized_geometry": "/path/to/optimized_geometry.xyz", // Path to optimized geometry file (if applicable)
                "frequencies": [500.0, 1500.0, ...], // List of vibrational frequencies in cm^-1 (if applicable)
                "charges": [0.1, -0.1, ...], // List of atomic charges (if applicable)
                "excitation_energies": [3.5, 4.2, ...], //List of excitation energies in eV (if applicable)
                "excitation_wavelengths": [350, 295, ...], // List of excitation wavelengths in nm (if applicable)  
                "oscillator_strengths": [0.1, 0.05, ...], // List of oscillator strengths (if applicable)
                "nmr_shifts": [10.5, 20.3, ...], // List of NMR chemical shifts in ppm (if applicable)
                "dipole_moment": 1.85, // Dipole moment in Debye (if applicable)
                "orbitals": [...], // List of molecular orbital energies in eV (if applicable)
                "G_corr": 0.01234, // Gibbs free energy correction in Hartree from Freq (if applicable)
                "H_corr": 0.02345, // Enthalpy correction in Hartree from Freq (if applicable)
                "frequency_file": "/path/to/frequencies_and_intensities.npz", // Path to npz file containing frequencies and intensities (if applicable)
                "tddft_file": "/path/to/excitation_energies_and_oscillator_strengths.npz" // Path to npz file containing excitation energies and oscillator strengths (if applicable)
            }
            Example of error result:
            {"error": "Unsupported job type: XXX"} // If the job type is not supported.
            {"error": "Unsupported method: XXX"} // If the specified method is not supported.
            {"error": "Unsupported basis_set: XXX"} // If the specified method is not supported.
            {"error": "Failed to read molecule file"} // If the molecule file cannot be read.
            {"error": "SCF did not converge"} // If the SCF calculation fails to converge.
            {"error": "Geometry optimization failed"} // If the geometry optimization fails.
    """
    result = {}
    try:
        job = json.loads(job_data)
    except Exception as e:
        logger.error(f"Failed to parse job data: {e}")
        return {"error": "Invalid job data format", "details": str(e)}
    try:
        job_type = job.get("job_type")
        if job_type not in JOB_LIST_REGISTRY:
            logger.error(f"Unsupported job type: {job_type}")
            raise ValueError(f"Unsupported job type: {job_type}")
        handler = JOB_LIST_REGISTRY[job_type]
        logger.info(f"Processing job type: {job_type} with handler {handler.__name__}")
        result = handler(job, logger)
        logger.info(f"Job completed successfully")
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        result = {"error": str(e)}
    return result


def _generate_safe_prompt(task_description, params):
    # 1. 设定角色和目标
    prompt = """
    You are an expert quantum chemistry programmer specializing in the PySCF library.
    Your task is to write a single PySCF script that accomplishes the user's goal.
    The script will take 'molecule.xyz' at the same directory as input.
    The results should be printed to standard output or saved to files in the current directory but the saved files' paths must be returned in stdout.
    The script MUST NOT import any libraries other than 'pyscf', 'numpy', and 'json'.
    It MUST NOT perform any file I/O (except reading the input molecule) or network operations.
    DO NOT give any explanations or comments outside the function definition.
    DO NOT include any code outside the function definition.
    Focus on clarity, correctness, and efficiency.
    """

    # 3. 提供上下文和文档
    prompt += f"""
    The user wants to perform the following task: "{task_description}"
    The molecule geometry is provided. The user-supplied parameters are: {params}.
    You can use standard PySCF functions like scf.RHF(mol).run(), cc.CCSD(mf).run(), etc.
    Focus on calculating the primary objective and returning the key results in a dictionary.
    """
    return prompt

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
def run_dynamic_job(code: str) -> dict:
    """
    Run dynamically generated Python code for quantum chemistry tasks using PySCF.

    Args:
        code (str): The Python code string generated by the Agent.
                     You should write a complete script that can be executed directly.
                     It reads the xyz file, executes the calculation, and prints or saves the results.
                     The stdout and stderr will be captured and returned.

    Returns:
        dict: A dictionary containing the results of the calculation or error messages.
    """
    result = {}
    try:
        # 1. 静态代码分析
        if not is_safe(code, logger):
            logger.error("Generated code failed the safety check.")
            return {"error": "Generated code failed the safety check. Please ensure it does not contain unsafe operations like file I/O, network access, or execution of arbitrary system commands."}

        # 2. 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            with open("dynamic_job.py", "w") as f:
                f.write(code)

            exec_result = subprocess.run(
                [sys.executable, "dynamic_job.py"],
                capture_output=True,
                text=True
            )
            logger.info("Dynamic job executed successfully.")
            result = { "output": exec_result.stdout, "error": exec_result.stderr }
    except Exception as e:
        logger.error(f"Error executing dynamic job: {e}")
        result = {"error": str(e)}
    return result
                    
@mcp.tool()
def convert_xyz_to_molstar_html(input_xyz_file: str):
    """
    读取 .xyz 文件，将其转换为 Mol* HTML 格式.。

    参数:
    input_xyz_file (str): 输入的 .xyz 文件路径。

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
            "html_file": output_html_file
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
    ) -> Tuple[str, str]:
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
        xyz_path = f"{MCP_SCRATCH}/{mol_name}.xyz"
        with open(xyz_path, "w") as f:
            f.write(xyz_content.strip())
            f.write("\n\n")
        return {
            "xyz_file": xyz_path,
            "message": f"Successfully wrote XYZ file: {xyz_path}"
        }
    except Exception as e:
        logger.error(f"Error writing XYZ file: {e}")
        return {
            "xyz_file": None,
            "message": f"Error writing XYZ file: {str(e)}"
        }

@mcp.tool()
def read_xyz_file(xyz_file: str) -> str:
    """
    Read an XYZ file and return its content as a string. Can be used to get the coordinates and atom types for further processing and analysis.

    Parameters:
    ----------
    xyz_file : Path
        The path to the XYZ file.

    Returns:
    -------
    str
        The content of the XYZ file as a string.
    """
    try:
        with open(xyz_file, "r") as f:
            content = f.read()
        return {
            "status": "success",
            "xyz_content": content
        }
    except Exception as e:
        logger.error(f"Error reading XYZ file: {e}")
        return {
            "status": "error",
            "message": f"Error reading XYZ file: {str(e)}"
        }

@mcp.tool()
async def smiles_to_xyz(
    smiles: str,
    mol_name: str,
    charge: int = 0,
    multiplicity: int = 1,
    n_cores: int = 2,
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
        xyz_path = f"{MCP_SCRATCH}/{mol_name}.xyz"
        logger.info(f"Converting SMILES to XYZ for molecule: {mol.name}")
        logger.debug(f"SMILES: {smiles}, Charge: {charge}, Multiplicity: {multiplicity}")
        logger.info(f"saving to {xyz_path}")
        mol.print_xyz_file(filename=str(xyz_path))
        return {
            "xyz_file": xyz_path,
            "message": f"Successfully converted SMILES to XYZ."
        }
    except Exception as e:
        logger.error(f"Error converting SMILES to XYZ: {e}")
        return {
            "xyz_file": None,
            "message": f"Error converting SMILES to XYZ: {str(e)}"
        }
    
@mcp.tool()
def get_smiles_from_pubchem(mol_name: str) -> Dict[str, str]:
    """
    Retrieves the Canonical SMILES string for a molecule from the PubChem database.

    Args:
        mol_name: The common or IUPAC name of the molecule.

    Returns:
        A dictionary containing the SMILES string under the "smiles" key on success,
        or an error message under the "error" key on failure.
    """
    if not mol_name or not isinstance(mol_name, str):
        return {"error": "Invalid input: Molecule name must be a non-empty string."}

    # Use the more specific API endpoint for SMILES
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{mol_name}/JSON"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()

        #go through the json to get the smiles
        for item in data.get('PC_Compounds', []):
            for prop in item.get('props', []):
                if prop.get('urn', {}).get('label') == 'SMILES' and prop.get('urn', {}).get('name') == 'Absolute':
                    smiles = prop.get('value', {}).get('sval')
                    if smiles:
                        return {"smiles": smiles}


    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return {"error": f"Molecule '{mol_name}' not found in PubChem."}
        else:
            return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}
    except (KeyError, IndexError, json.JSONDecodeError):
        return {"error": "Failed to parse a valid SMILES string from the API response."}
    
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
def plot_spectrum(npz_file: List[Path], spectrum_type: Literal["IR", "UV-Vis", "NMR"], output_file: str) -> Dict[str, Union[str, Path]]:
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
            output_file = Path(f"{MCP_SCRATCH}/spectrum.png")
        output_file = Path(f"{MCP_SCRATCH}/{output_file}")
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
 