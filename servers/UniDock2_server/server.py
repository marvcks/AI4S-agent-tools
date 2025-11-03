#!/usr/bin/env python3
"""
Example MCP Server using the new simplified pattern.
This demonstrates how to create a new AI4S tool with tools defined at module level.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import argparse
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import requests
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import loguru
from pathlib import Path

# 导入MCP相关
from dp.agent.server import CalculationMCPServer

import MDAnalysis as mda
import re
import nanoid



def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server for Uni-Dock2")
    parser.add_argument('--port', type=int, default=50004, help='Server port (default: 50004)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50004
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = CalculationMCPServer("ud2_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_ud2_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"Uni-Dock2 MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")


MCP_SCRATCH = os.getenv("MCP_SCRATCH", "/tmp")
MCP_SCRATCH_PATH = Path(MCP_SCRATCH)


@mcp.tool()
def extract_template_ligand_from_holo_pdb(holo_pdb: Path, ligand_resname: str="LIG") -> TypedDict("results",{"status": str, "receptor_pdb": Path, "ligand_sdf": Path}): 
    """
    Extract the native ligand from a receptor-ligand complex pdb file.
    Will save a protein-only pdb file and a ligand-only sdf file.
    Args:
        holo_pdb (Path): Path to the holo PDB file containing both receptor and ligand.
        ligand_resname (str): Residue name of the ligand in the PDB file. Default is "LIG".
    Returns:
        Dict[str, Any]: Dictionary containing status, paths to the receptor PDB file and ligand SDF file.
    Example Input/Output:
        Input:
            holo_pdb: https://XXX/3HTB.pdb
            ligand_resname: "JZ4"
        Output:
        {
            "status": "success",
            "receptor_pdb": Path("/path/to/receptor_XXXXXX.pdb"),
            "ligand_sdf": Path("/path/to/ligand_XXXXXX.sdf")
        }
        
    """
    try:
        u = mda.Universe(holo_pdb)
        protein = u.select_atoms("protein")
        ligand = u.select_atoms(f"resname {ligand_resname}")
    except Exception as e:
        logger.error(f"Error loading PDB file {holo_pdb}: {e}")
        return {"status": "error", "message": f"Error loading PDB file: {e}"}
    if len(ligand) == 0:
        logger.error(f"No ligand found with residue name '{ligand_resname}' in {holo_pdb}")
        return {"status": "error", "message": f"No ligand found with residue name '{ligand_resname}'"}
    
    id = nanoid.generate(size=6)
    receptor_pdb_path = Path(f"receptor_{id}.pdb")
    ligand_pdb_path = Path(f"ligand_{id}.pdb")
    try:
        protein.write(receptor_pdb_path)
        ligand.write(ligand_pdb_path)
    except Exception as e:
        logger.error(f"Error writing output files: {e}")
        return {"status": "error", "message": f"Error writing output files: {e}"}
    # Convert ligand PDB to SDF using Open Babel
    ligand_sdf_path = Path(f"ligand_{id}.sdf")
    try:
        subprocess.run(["obabel", "-ipdb", ligand_pdb_path, "-osdf", "-O", ligand_sdf_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting ligand PDB to SDF: {e}")
        return {"status": "error", "message": f"Error converting ligand PDB to SDF: {e}"}
    logger.info(f"Extracted receptor to {receptor_pdb_path} and ligand to {ligand_sdf_path}")
    return {"status": "success", "receptor_pdb": receptor_pdb_path, "ligand_sdf": ligand_sdf_path}

    
@mcp.tool()
def convert_ligand_file_to_sdf(input_file: Path) -> TypedDict("results",{"status": str, "sdf_file": Path}):
    """
    Convert a molecular structure file (PDB, MOL2, SDF) to SDF format using Open Babel.
    Args:
        input_file (Path): Path to the input structure file.
    Returns:
        Dict[str, Any]: Dictionary containing status and path to the converted SDF file.
    Example Input/Output:
        Input:
            input_file: https://XXX/molecule.pdb
        Output:
        {
            "status": "success",
            "sdf_file": Path("/path/to/converted_XXXXXX.sdf")
        }
    """
    if not input_file.is_file():
        logger.error(f"Input file {input_file} does not exist.")
        return {"status": "error", "message": f"Input file {input_file} does not exist."}
    
    id = nanoid.generate(size=6)
    sdf_path = Path(f"converted_{id}.sdf")
    try:
        subprocess.run(["obabel", "-i", input_file.suffix[1:], input_file, "-osdf", "-O", sdf_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting file to SDF: {e}")
        return {"status": "error", "message": f"Error converting file to SDF: {e}"}
    
    logger.info(f"Converted {input_file} to {sdf_path}")
    return {"status": "success", "sdf_file": sdf_path}


@mcp.tool()
def calculate_docking_box_center(receptor_pdb: Path, selection: str) -> TypedDict("results",{"status": str, "center": Tuple[float, float, float]}):
    """
    Calculate the geometric center of a selection in a PDB file.
    Args:
        receptor_pdb (Path): Path to the receptor PDB file.
        selection (str): MDAnalysis selection string to define the region of interest. e.g. resid 100 
    Returns:
        Dict[str, Any]: Dictionary containing status and the (x, y, z) coordinates of the center.
    Example Input/Output:
        Input:
            receptor_pdb: https://XXX/receptor.pdb
            selection: "resid 100"
        Output:
        {
            "status": "success",
            "center": (12.345, -6.789, 0.123)
        }
    """
    if not receptor_pdb.is_file():
        logger.error(f"Receptor PDB file {receptor_pdb} does not exist.")
        return {"status": "error", "message": f"Receptor PDB file {receptor_pdb} does not exist."}
    
    try:
        u = mda.Universe(receptor_pdb)
        selected_atoms = u.select_atoms(selection)
        if len(selected_atoms) == 0:
            logger.error(f"No atoms found for selection '{selection}' in {receptor_pdb}")
            return {"status": "error", "message": f"No atoms found for selection '{selection}'"}
        center = selected_atoms.center_of_geometry()
    except Exception as e:
        logger.error(f"Error processing PDB file {receptor_pdb}: {e}")
        return {"status": "error", "message": f"Error processing PDB file: {e}"}
    
    logger.info(f"Calculated center for selection '{selection}' in {receptor_pdb}: {center}")
    return {"status": "success", "center": (float(center[0]), float(center[1]), float(center[2]))}

@mcp.tool()
def combine_protein_ligand(receptor_pdb: Path, ligand_sdf: Path, ligand_resname: str="LIG") -> TypedDict("results",{"status": str, "complex_pdb": Path}):
    """
    Combine a receptor PDB file and a ligand SDF file into a single PDB file for further MD.
    Args:
        receptor_pdb (Path): Path to the receptor PDB file.
        ligand_sdf (Path): Path to the ligand SDF file.
        ligand_resname (str): Residue name to assign to the ligand in the combined PDB. Default is "LIG".
    Returns:
        Dict[str, Any]: Dictionary containing status and path to the combined PDB file.

    Example Input/Output:
        Input:
            receptor_pdb: https://XXX/receptor.pdb
            ligand_sdf: https://XXX/ligand.sdf
            ligand_resname: "LIG"
        Output:
        {
            "status": "success",
            "complex_pdb": Path("/path/to/complex_XXXXXX.pdb")
        }
    """
    if not receptor_pdb.is_file():
        logger.error(f"Receptor PDB file {receptor_pdb} does not exist.")
        return {"status": "error", "message": f"Receptor PDB file {receptor_pdb} does not exist."}
    if not ligand_sdf.is_file():
        logger.error(f"Ligand SDF file {ligand_sdf} does not exist.")
        return {"status": "error", "message": f"Ligand SDF file {ligand_sdf} does not exist."}
    
    id = nanoid.generate(size=6)
    combined_pdb_path = Path(f"complex_{id}.pdb")
    ligand_path = Path(f"ligand_{id}.pdb")
    try:
        subprocess.run(["obabel", "-isdf", ligand_sdf, "-opdb", "-O", ligand_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting ligand SDF to PDB: {e}")
        return {"status": "error", "message": f"Error converting ligand SDF to PDB: {e}"}
    u_receptor = mda.Universe(receptor_pdb)
    u_ligand = mda.Universe(ligand_path)
    u_ligand.trajectory[0]
    #rename ligand residue to LIG
    for res in u_ligand.residues:
        res.resname = ligand_resname

    combined = mda.Merge(u_receptor.atoms, u_ligand.atoms)
    try:
        combined.atoms.write(combined_pdb_path)
    except Exception as e:
        logger.error(f"Error writing combined PDB file: {e}")
        return {"status": "error", "message": f"Error writing combined PDB file: {e}"}
    logger.info(f"Combined receptor and ligand into {combined_pdb_path}")
    return {"status": "success", "complex_pdb": combined_pdb_path}
    

# Define tools at module level
@mcp.tool()
def run_unidock2(
    receptor_pdb: Path,
    ligand_sdf: Path,
    center_x: float,
    center_y: float,
    center_z: float,
    box_size_x: float = 30.0,
    box_size_y: float = 30.0,
    box_size_z: float = 30.0,
    template_sdf: Optional[Path] = None,
) -> TypedDict("results",{"status": str, "results_sdf": Path, "affinity": Optional[List[float]]}):
    """
    Run Uni-Dock2 docking simulation.

    Args:
        receptor_pdb (Path): Path to the receptor PDB file.
        ligand_sdf (Path): Path to the ligand SDF file.
        center_x (float): X coordinate of the box center.
        center_y (float): Y coordinate of the box center.
        center_z (float): Z coordinate of the box center.
        box_size_x (float): Size of the box in X dimension. Default is 30.0.
        box_size_y (float): Size of the box in Y dimension. Default is 30.0.
        box_size_z (float): Size of the box in Z dimension. Default is 30.0.
        template_sdf (Optional[Path]): Path to a template ligand SDF file for guided docking. Default is None.

    Returns:
        Dict[str, Any]: Dictionary containing status and path to the docking results file.

    Example Input/Output:
        Input:
            receptor_pdb: https://XXX/receptor.pdb
            ligand_sdf: https://XXX/ligand.sdf
            center_x: 0.0
            center_y: 0.0
            center_z: 0.0
            box_size_x: 30.0
            box_size_y: 30.0
            box_size_z: 30.0
            template_sdf: https://XXX/template.sdf
        Output:
        {
            "status": "success",
            "results_sdf": Path("/path/to/ud2_results_XXXXXX.sdf"),
            "affinity": [-7.5, -6.8, -6.2]  # List of docking affinities in kcal/mol
        }
    """
    logger.info(f"Running Uni-Dock2 with receptor: {receptor_pdb}, ligand: {ligand_sdf}, center: ({center_x}, {center_y}, {center_z}), box size: ({box_size_x}, {box_size_y}, {box_size_z}), template: {template_sdf}")
    id = nanoid.generate(size=6)
    input_file = os.path.join(MCP_SCRATCH, f"ud2_{id}.yaml")
    with open(input_file, 'w') as f:
        f.write("Required:\n")
        f.write(f"  receptor: {receptor_pdb}\n")
        f.write(f"  ligand: {ligand_sdf}\n")
        f.write(f"  center: [{center_x}, {center_y}, {center_z}]\n")
        f.write("Settings:\n")
        f.write(f"  box_size: [{box_size_x}, {box_size_y}, {box_size_z}]\n")
        if template_sdf:
            f.write("Preprocessing:\n")
            f.write(f"  template_docking: true\n")
            f.write(f"  reference_sdf_file_name: {template_sdf}\n")

    output_file = Path(f"ud2_results_{id}.sdf")
    try:
        result = subprocess.run(["unidock2", "docking" , "-cf", input_file, "-o", output_file], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Uni-Dock2: {e}")
        return {"status": "error", "message": f"Error running Uni-Dock2: {e}"}
    #check every line in result.stdout for affinity
    regex = r"1\s+(-?\d+\.\d+)"
    match_all = re.findall(regex, result.stdout)
    if match_all:
        affinity = [float(x) for x in match_all]
        logger.info(f"Uni-Dock2 completed. Best affinity: {affinity} kcal/mol. Results saved to {output_file}")
    else:
        logger.warning(f"Uni-Dock2 completed but no affinity found in output. Results saved to {output_file}")

    logger.info(f"Uni-Dock2 completed. Results saved to {output_file}")
    return {"status": "success", "results_sdf": output_file, "affinity": affinity if 'affinity' in locals() else None}


@mcp.tool()
def get_unidock2_workflow_instructions() -> str:  
    """Provides instructions for the agent on how to use the Uni-Dock2 tools effectively.
    Please refer to the detailed guidelines below before performing Uni-Dock molecular docking tasks.
    Returns:
        str: Instructions for the agent.
    """
    instruction="""
    # Role: Molecular Docking Specialist

    ## Primary Objective:
    To perform molecular docking of small molecule ligands with protein receptors, predicting binding modes and affinities.

    ## Core Protocol for docking:
    1. **Docking**: You can use `run_unidock2` tool to perform molecular docking if needed. 
    2. Note that you need to call `convert_file_to_sdf` to convert ligand file to sdf format before docking 
    3. You can use `calculate_box_center` to determine the center of the docking box based on the receptor structure and key binding site residues.
    3. You can use `combine_protein_ligand` to combine the receptor and ligand into a single pdb file for further preparation and MD simulation.
    4. Always ensure the input files are in the correct format and paths are valid.

    ## Additional Guidelines for Templated Docking:
    - If a template ligand is provided, utilize it to guide the docking process.
    - If user provides a protein-ligand complex structure, use it to identify the binding site and set up the docking box accordingly.
    - You can use `extract_template_ligand` to extract the ligand from the provided complex structure for use as a template and get a empty receptor pdb for docking.
    - Ensure that the docking box is appropriately sized. You can use `calculate_box_center` to determine optimal box center based on the template ligand.

    ## Important Notes:
    All the file paths input to the tools should be a url like `https://...` or a local path like `local://<path>`.
    """
    return instruction


if __name__ == "__main__":
    logger.info("Starting ProteinPreP MCP Server with all tools...")
    #fetch_rcsb("3HTB")
    #Protein_Prep("3HTB.pdb", toDeleteRes=["PO4", "BME"])
    #parametrize_ligand("3HTB_fixer_amber.pdb", "JZ4", 0)
    #extract_template_ligand("3HTB.pdb", "JZ4")
    #run_unidock2(
    #    receptor_pdb="receptor_YPwmV7.pdb",
    #    ligand_sdf="ligand_YPwmV7.sdf",
    #    center_x=25.0,
    #    center_y=-25.0,
    #    center_z=0.0,
    #)
    #combine_protein_ligand(
    #    receptor_pdb="receptor_YPwmV7.pdb",
    #    ligand_sdf="ud2_results_IPj2j2.sdf",
    #)
    #run_unidock2(
    #    receptor_pdb="receptor_YPwmV7.pdb",
    #    ligand_sdf="phenol.sdf",
    #    center_x=0.0,
    #    center_y=0.0,
    #    center_z=0.0,
    #    template_sdf="ligand_YPwmV7.sdf",
    #)

    mcp.run(transport="sse")
    