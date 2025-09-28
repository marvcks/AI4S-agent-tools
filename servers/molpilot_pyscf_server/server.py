#!/usr/bin/env python3
"""
PySCF MCP服务器
"""
import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import json


# 导入MCP相关
from dp.agent.server import CalculationMCPServer


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

os.environ['OMP_NUM_THREADS'] = "2" 

MCP_SCRATCH = os.getenv("MCP_SCRATCH", "/tmp")




def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MOLPILOT PySCF MCP服务器")
    parser.add_argument('--port', type=int, default=50011, help='服务器端口 (默认: 50011)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机 (默认: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50011
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("molpilot_pyscf_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_pyscf_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"Molpilot PySCF MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")

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
                    

if __name__ == "__main__":
    logger.info("Starting the MCP server...")
    mcp.run(transport="sse")
 