#!/usr/bin/env python3
"""
ORCA Server
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

from ase.io import read, write


os.environ["AUTODE_LOG_LEVEL"] = "INFO"
os.environ["AUTODE_LOG_FILE"] = "autode.log"

# 初始化 PATH 和 LD_LIBRARY_PATH
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/opt/mamba/bin"
os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + os.pathsep + "/usr/local/cuda/lib64"

# OpenMPI 路径
os.environ["PATH"] += os.pathsep + "/opt/openmpi411/bin"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + "/opt/openmpi411/lib"

# ORCA 路径
os.environ["PATH"] += os.pathsep + "/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + "/opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411"

# packmol 路径
os.environ["PATH"] += os.pathsep + "/root/packmol-21.1.0"

# 允许以 root 运行 MPI
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

# xtb 路径
os.environ["PATH"] += os.pathsep + "/root/xtb-6.6.1/bin"


def parse_args():
    parser = argparse.ArgumentParser(description="ORCA Server")
    parser.add_argument('--port', type=int, default=50009, help='Server port (default: 50009)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50009
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("orca_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)


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
            env=os.environ.copy(),
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


if __name__ == "__main__":
    logging.info("Starting ORCA Server with all tools...")
    mcp.run(transport="sse")
 