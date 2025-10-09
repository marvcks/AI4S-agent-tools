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

@mcp.tool()
def extract_template_ligand(holo_pdb: str, ligand_resname: str="LIG") -> Dict[str, Any]:
    """
    Extract the native ligand from a receptor-ligand complex pdb file.
    Will save a protein-only pdb file and a ligand-only sdf file.
    Args:
        holo_pdb (str): Path to the holo PDB file containing both receptor and ligand.
        ligand_resname (str): Residue name of the ligand in the PDB file. Default is "LIG".
    Returns:
        Dict[str, Any]: Dictionary containing status, paths to the receptor PDB file and ligand SDF file.
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
    receptor_pdb_path = os.path.join(MCP_SCRATCH, f"receptor_{id}.pdb")
    ligand_pdb_path = os.path.join(MCP_SCRATCH, f"ligand_{id}.pdb")
    try:
        protein.write(receptor_pdb_path)
        ligand.write(ligand_pdb_path)
    except Exception as e:
        logger.error(f"Error writing output files: {e}")
        return {"status": "error", "message": f"Error writing output files: {e}"}
    # Convert ligand PDB to SDF using Open Babel
    ligand_sdf_path = os.path.join(MCP_SCRATCH, f"ligand_{id}.sdf")
    try:
        subprocess.run(["obabel", "-ipdb", ligand_pdb_path, "-osdf", "-O", ligand_sdf_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting ligand PDB to SDF: {e}")
        return {"status": "error", "message": f"Error converting ligand PDB to SDF: {e}"}
    logger.info(f"Extracted receptor to {receptor_pdb_path} and ligand to {ligand_sdf_path}")
    return {"status": "success", "receptor_pdb": receptor_pdb_path, "ligand_sdf": ligand_sdf_path}

    
@mcp.tool()
def convert_file_to_sdf(input_file: str) -> Dict[str, Any]:
    """
    Convert a molecular structure file (PDB, MOL2, SDF) to SDF format using Open Babel.
    Args:
        input_file (str): Path to the input structure file.
    Returns:
        Dict[str, Any]: Dictionary containing status and path to the converted SDF file.
    """
    if not os.path.isfile(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return {"status": "error", "message": f"Input file {input_file} does not exist."}
    
    id = nanoid.generate(size=6)
    sdf_path = os.path.join(MCP_SCRATCH, f"converted_{id}.sdf")
    try:
        subprocess.run(["obabel", "-i", input_file.split('.')[-1], input_file, "-osdf", "-O", sdf_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting file to SDF: {e}")
        return {"status": "error", "message": f"Error converting file to SDF: {e}"}
    
    logger.info(f"Converted {input_file} to {sdf_path}")
    return {"status": "success", "sdf_file": sdf_path}

@mcp.tool()
def combine_protein_ligand(receptor_pdb: str, ligand_sdf: str, ligand_resname: str="LIG") -> Dict[str, Any]:
    """
    Combine a receptor PDB file and a ligand SDF file into a single PDB file for further MD.
    Args:
        receptor_pdb (str): Path to the receptor PDB file.
        ligand_sdf (str): Path to the ligand SDF file.
        ligand_resname (str): Residue name to assign to the ligand in the combined PDB. Default is "LIG".
    Returns:
        Dict[str, Any]: Dictionary containing status and path to the combined PDB file.
    """
    if not os.path.isfile(receptor_pdb):
        logger.error(f"Receptor PDB file {receptor_pdb} does not exist.")
        return {"status": "error", "message": f"Receptor PDB file {receptor_pdb} does not exist."}
    if not os.path.isfile(ligand_sdf):
        logger.error(f"Ligand SDF file {ligand_sdf} does not exist.")
        return {"status": "error", "message": f"Ligand SDF file {ligand_sdf} does not exist."}
    
    id = nanoid.generate(size=6)
    combined_pdb_path = os.path.join(MCP_SCRATCH, f"complex_{id}.pdb")
    try:
        subprocess.run(["obabel", "-isdf", ligand_sdf, "-opdb", "-O", f"{MCP_SCRATCH}/ligand_{id}.pdb"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting ligand SDF to PDB: {e}")
        return {"status": "error", "message": f"Error converting ligand SDF to PDB: {e}"}
    u_receptor = mda.Universe(receptor_pdb)
    u_ligand = mda.Universe(f"{MCP_SCRATCH}/ligand_{id}.pdb")
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
    receptor_pdb: str,
    ligand_sdf: str,
    center_x: float,
    center_y:float,
    center_z:float,
    box_size_x: float = 30.0,
    box_size_y: float = 30.0,
    box_size_z: float = 30.0,
    template_sdf: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Uni-Dock2 docking simulation.

    Args:
        receptor_pdb (str): Path to the receptor PDB file.
        ligand_sdf (str): Path to the ligand SDF file.
        center_x (float): X coordinate of the box center.
        center_y (float): Y coordinate of the box center.
        center_z (float): Z coordinate of the box center.
        box_size_x (float): Size of the box in X dimension. Default is 30.0.
        box_size_y (float): Size of the box in Y dimension. Default is 30.0.
        box_size_z (float): Size of the box in Z dimension. Default is 30.0.
        template_sdf (Optional[str]): Path to a template ligand SDF file for guided docking. Default is None.

    Returns:
        Dict[str, Any]: Dictionary containing status and path to the docking results file.
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
    
    output_file = os.path.join(MCP_SCRATCH, f"ud2_results_{id}.sdf")
    try:
        result = subprocess.run(["unidock2", "docking" , "-cf", input_file, "-o", output_file], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Uni-Dock2: {e}")
        return {"status": "error", "message": f"Error running Uni-Dock2: {e}"}
    #check every line in result.stdout for affinity
    regex = r"1\s+(-?\d+\.\d+)"
    for line in result.stdout.splitlines():
        #[2025-10-04 17:13:56.868] [info] 1         -7.2813826 
        match = re.search(regex, line)
        if match:
            affinity = float(match.group(1))
            logger.info(f"Docking completed with best affinity: {affinity} kcal/mol")
            break
    else:
        logger.warning("No affinity found in Uni-Dock2 output.")

    logger.info(f"Uni-Dock2 completed. Results saved to {output_file}")
    return {"status": "success", "results_sdf": output_file, "affinity": affinity if 'affinity' in locals() else None}

        

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
    