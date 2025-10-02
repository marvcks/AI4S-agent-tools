#!/usr/bin/env python3
"""
Structure Generator Server
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
import requests
from dp.agent.server import CalculationMCPServer

import autode as ade
import anyio
import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


import molviewspec as mvs
from ase.io import read, write


os.environ["AUTODE_LOG_LEVEL"] = "INFO"
os.environ["AUTODE_LOG_FILE"] = "autode.log"


# packmol 路径
os.environ["PATH"] += os.pathsep + "/root/packmol-21.1.0"

# 允许以 root 运行 MPI
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

# xtb 路径
os.environ["PATH"] += os.pathsep + "/root/xtb-6.6.1/bin"


def parse_args():
    parser = argparse.ArgumentParser(description="Structure Generator Server")
    parser.add_argument('--port', type=int, default=50010, help='Server port (default: 50010)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50010
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("structure_generator_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)

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
                        return {
                            "smiles": smiles, 
                            "status": "success"
                            }


    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return {
                "status": "error",
                "message": f"Molecule '{mol_name}' not found in PubChem."
                }
        else:
            return {
                "status": "error",
                "message": f"HTTP error occurred: {http_err}"
                }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "error",
            "message": f"A network error occurred: {req_err}"
            }
    except (KeyError, IndexError, json.JSONDecodeError):
        return {
            "status": "error",
            "message": "Failed to parse a valid SMILES string from the API response."
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
        logging.error(f"Error uploading file to R2: {e}")
        return {
            "status": "error",
            "message": f"Error uploading file to R2: {str(e)}"
        }

if __name__ == "__main__":
    logging.info("Starting Structure Generator Server with all tools...")
    mcp.run(transport="sse")
 