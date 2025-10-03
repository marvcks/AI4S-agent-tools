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
from mcp.server.fastmcp import FastMCP

import pdbfixer
from openmm.app import PDBFile, Modeller
import re
import nanoid



def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server for MD Protein Preparation")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
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
mcp = FastMCP("proteinprep_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_pyscf_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"ProteinPreP MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")

ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
MCP_SCRATCH = os.getenv("MCP_SCRATCH", "/tmp")


# Define tools at module level
@mcp.tool()
def fetch_rcsb(pdb_id: str) -> dict:
    """
    Fetches the RCSB PDB file for a given PDB ID.
    
    Args:
        pdb_id: The PDB ID of the protein structure to fetch.
    Returns:
        dict: A dictionary containing the status and path to the downloaded PDB file or error message.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{MCP_SCRATCH}/{pdb_id}.pdb", "w") as f:
            f.write(response.text)
        return {"status": "success", "pdb_path": f"{MCP_SCRATCH}/{pdb_id}.pdb"}
    else:
        return {"status": "error", "message": f"Failed to fetch PDB ID {pdb_id}. HTTP Status: {response.status_code}"}

@mcp.tool()
def Protein_Prep(pdb_path: str, ph: float = 7.0, toDeleteRes: Optional[List[str]] = None) -> dict:
    """
    Prepares a protein structure for molecular dynamics simulations.
    Will add the missing residues, add hydrogens, and adjust protonation states.
    It won't add hydrogens to ligands or other non-protein residues.
    
    Args:
        pdb_path: Path to the input PDB file.
        ph: pH value for protonation state adjustment (default: 7.0).
        toDeleteRes: List of residue names to delete from the structure (e.g., PO4, BEM).
    Returns:
        dict: A dictionary containing the status and path to the prepared PDB file or error message
    """
    try:
        fixer = pdbfixer.PDBFixer(filename=pdb_path)
        #remove all hydrogens first
        for atom in list(fixer.topology.atoms()):
            if atom.element == "H":
                logger.debug(f"Removing hydrogen atom: {atom}")
                fixer.topology.deleteAtom(atom)
                fixer.positions = np.delete(fixer.positions, atom.index, axis=0)

        fixer.findMissingResidues()

        #only add missing residues that are not at the termini
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        missingResidues = dict()
        for key in keys:
            chain = chains[key[0]]
            if not (key[1] == 0 or key[1] == len(list(chain.residues()))):
                missingResidues[key] = fixer.missingResidues[key]
        fixer.missingResidues = missingResidues

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        delete_list = []
        if toDeleteRes:
            for residue in fixer.topology.residues():
                if residue.name in toDeleteRes:
                    logger.info(f"Deleting residue: {residue}")
                    delete_list.append(residue)
        modifier = Modeller(fixer.topology, fixer.positions)
        modifier.delete(delete_list)
        fixer.topology = modifier.topology
        fixer.positions = modifier.positions

        output_path = pdb_path.replace(".pdb", "_fixer.pdb")
        logger.info(f"Prepared PDB will be saved to {output_path}")
        with open(output_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        command = ["pdb4amber", "-i", output_path, "-o", output_path.replace(".pdb", "_amber.pdb")]
        output_path = output_path.replace(".pdb", "_amber.pdb")
        logger.info(f"Running pdb4amber: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"pdb4amber failed: {result.stderr}")
            return {"status": "error", "message": f"pdb4amber failed: {result.stderr}"}

        return {"status": "success", "prepared_pdb_path": output_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@mcp.tool()
def get_protein_sequence(pdb_path: str) -> dict:
    """
    Extracts the amino acid sequence from a PDB file.
    
    Args:
        pdb_path: Path to the input PDB file.
    Returns:
        dict: A dictionary containing the status and the amino acid sequence or error message.
    """
    try:
        fixer = pdbfixer.PDBFixer(filename=pdb_path)
        sequence = ""
        for chain in fixer.topology.chains():
            for residue in chain.residues():
                sequence += residue.name
                sequence += " "
            sequence += ":"  # Chain separator
        sequence = sequence.rstrip(":")  # Remove trailing separator
        return {"status": "success", "sequence": sequence}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@mcp.tool()
def parametrize_ligand(pdb_path: str, ligand_resname: str = "LIG", charge: int = 0) -> dict:
    """
    Parametrizes a ligand using antechamber and parmchk2 from AmberTools.
    
    Args:
        pdb_path: Path to the input PDB file of the protein-ligand complex.
        ligand_resname: The residue name of the ligand in the PDB file (default: "LIG").
        charge: The net charge of the ligand (default: 0).
    Returns:
        dict: A dictionary containing the status and a ligand id that can be used to retrieve the generated files or error message.
    """
    try:
        ligand_id = nanoid.generate(size=6)
        logger.info(f"Parametrizing ligand {ligand_resname} with ID {ligand_id}")
        #extract ligand to a separate pdb file
        cmd = f"grep {ligand_resname} {pdb_path} > {MCP_SCRATCH}/{ligand_id}.pdb"
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Ligand PDB saved to {MCP_SCRATCH}/{ligand_id}.pdb")
        ligand_pdb_path = f"{MCP_SCRATCH}/{ligand_id}.pdb"

        # Run antechamber and parmchk2
        mol2_path = f"{MCP_SCRATCH}/{ligand_id}.mol2"
        frcmod_path = f"{MCP_SCRATCH}/{ligand_id}.frcmod"

        command = ["antechamber", "-i", ligand_pdb_path, "-fi", "pdb", "-o", mol2_path, "-fo", "mol2",
                   "-c", "bcc", "-rn", ligand_resname, "-nc", str(charge)]

        logger.info(f"Running antechamber: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Antechamber failed: {result.stderr}")
            return {"status": "error", "message": f"Antechamber failed: {result.stderr}"}

        command = ["parmchk2", "-i", mol2_path, "-f", "mol2", "-o", frcmod_path]
        logger.info(f"Running parmchk2: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Parmchk2 failed: {result.stderr}")
            return {"status": "error", "message": f"Parmchk2 failed: {result.stderr}"}

        return {"status": "success", "ligand_id": ligand_id, "mol2_path": mol2_path, "frcmod_path": frcmod_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def HMR(prmtop_path: str, inpcrd_path: str) -> dict:
    """
    Applies Hydrogen Mass Repartition
    Args:
        prmtop_path: Path to the input prmtop file.
        inpcrd_path: Path to the input inpcrd file.
    Returns:
        dict: A dictionary containing the status and paths to the modified files or error message.
    """
    prefix = prmtop_path.replace(".prmtop", "")
    hmr_template =f"""HMassRepartition
    outparm {prefix}_hmr.parm7 {prefix}_hmr.rst7
    """
    try:
        with open(f"{MCP_SCRATCH}/hmr.in", "w") as f:
            f.write(hmr_template)
        cmd = f"parmed -O -p {prmtop_path} -c {inpcrd_path} -i {MCP_SCRATCH}/hmr.in"
        logger.info(f"Running HMR: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"HMR failed: {result.stderr}")
            logger.error(f"HMR failed: {result.stdout}")
        
        return {"status": "success", "prmtop_path": f"{prefix}_hmr.parm7", "inpcrd_path": f"{prefix}_hmr.rst7"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def run_tleap(prepared_pdb_path: str, ligand_res_name: str, ligand_id: str) -> dict:
    """
    Runs tleap to generate topology and coordinate files for the protein-ligand complex.
    
    Args:
        prepared_pdb_path: Path to the prepared PDB file of the protein (output from Protein_Prep).
        ligand_res_name: The residue name of the ligand in the PDB file. Can be empty if no ligand.
        ligand_id: The ligand id returned from parametrize_ligand. Can be empty if no ligand.
    Returns:
        dict: A dictionary containing the status and paths to the generated files or error message.
    """
    try:
        prefix = prepared_pdb_path.replace(".pdb", "")
        if ligand_res_name != "" and ligand_id == "":
            return {"status": "error", "message": "ligand_id must be provided if ligand_res_name is provided"}
        if ligand_res_name == "" and ligand_id != "":
            return {"status": "error", "message": "ligand_res_name must be provided if ligand_id is provided"}
        if ligand_res_name != "" and ligand_id != "":
            tleap_input = f"""
    source leaprc.protein.ff14SB
    source leaprc.gaff
    source leaprc.water.tip3p

    {ligand_res_name} = loadmol2 {MCP_SCRATCH}/{ligand_id}.mol2
    loadamberparams {MCP_SCRATCH}/{ligand_id}.frcmod
    mol = loadpdb {prepared_pdb_path}
    center mol
    alignAxes mol
    savepdb mol {MCP_SCRATCH}/{prefix}_dry.pdb
    solvateBox mol TIP3PBOX 10.0
    addions mol Na+ 0
    addions mol Cl- 0
    savepdb mol {prefix}_solv.pdb
    saveamberparm mol {prefix}_solv.prmtop {prefix}_solv.inpcrd
    quit
    """
        else:
            tleap_input = f"""
    source leaprc.protein.ff14SB
    source leaprc.water.tip3p

    mol = loadpdb {prepared_pdb_path}
    center mol
    alignAxes mol
    savepdb mol {MCP_SCRATCH}/{prefix}_dry.pdb
    solvateBox mol TIP3PBOX 10.0
    addions mol Na+ 0
    addions mol Cl- 0
    savepdb mol {prefix}_solv.pdb
    saveamberparm mol {prefix}_solv.prmtop {prefix}_solv.inpcrd
    quit
    """

        with open(f"{MCP_SCRATCH}/tleap.in", "w") as f:
            f.write(tleap_input)

        cmd = f"tleap -f {MCP_SCRATCH}/tleap.in"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            #check if the error is due to missing ligand parameters
            if "does not have a type" in result.stdout:
                logger.warning(f"tleap failed due to atom mismatch, will retry")
            else:
                logger.error(f"tleap failed")
                for line in result.stdout.split("\n"):
                    if line.startswith("FATAL"):
                        logger.error(f"tleap error: {line}")
                        return {"status": "error", "message": f"tleap failed: {line}"}
            
            with open(prepared_pdb_path, "r") as f:
                pdb_lines = f.readlines()
            #check if the error is due to atom type, e.g. FATAL:  Atom .R<NSER 544>.A<H 14> does not have a type
            #split the stdout by lines and look for lines starting with FATAL
            for line in result.stdout.split("\n"):
                if line.startswith("FATAL"):
                    logger.error(f"tleap error: {line}")
                    #check if the error is due to atom type, e.g. FATAL:  Atom .R<NSER 544>.A<H 14> does not have a type
                    match = re.search(r"FATAL:  Atom .R<(\w+) (\d+)>\.A<(\w+) (\d+)> does not have a type", line)
                    if match:

                        residue_index = match.group(2)
                        atom_name = match.group(3)


                        logger.error(f"tleap error: Atom {atom_name} in residue {residue_index} does not have a type")
                        #remove the atom from the pdb file
                        new_pdb_lines = []
                        for pdb_line in pdb_lines:
                            if pdb_line.startswith("HETATM") or pdb_line.startswith("ATOM"):
                                if (pdb_line[22:26].strip() == residue_index and 
                                    pdb_line[12:16].strip() == atom_name):
                                    logger.info(f"Removing atom line: {pdb_line.strip()}")
                                    continue 
                            new_pdb_lines.append(pdb_line)
                        pdb_lines = new_pdb_lines

            #write the new pdb file
            with open(prepared_pdb_path, "w") as f:
                f.writelines(pdb_lines)
            #retry tleap
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"tleap failed again: {result.stdout}")
                for line in result.stdout.split("\n"):
                    if line.startswith("FATAL"):
                        logger.error(f"tleap error: {line}")
                        return {"status": "error", "message": f"tleap failed: {line}"}
        # If we reach here, it means tleap succeeded
        logger.info(f"tleap succeeded")
        HMR_result = HMR(f"{prefix}_solv.prmtop", f"{prefix}_solv.inpcrd")
        if HMR_result["status"] != "success":
            return HMR_result
        return {"status": "success", 
                "prmtop_path": HMR_result["prmtop_path"], 
                "inpcrd_path": HMR_result["inpcrd_path"]}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@mcp.tool()
def upload_to_r2(file_path: str, object_key: str) -> dict:
    """
    Uploads a file to Cloudflare R2 storage.
    
    Args:
        file_path: Path to the local file to upload.
        object_key: The key (path) under which to store the file in the bucket.
    Returns:
        dict: A dictionary containing the status and message about the upload result.
    """
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY,
        )

        s3_client.upload_file(file_path, BUCKET_NAME, object_key)
        return {"status": "success", "message": f"File uploaded to {BUCKET_NAME}/{object_key}"}
    except FileNotFoundError:
        return {"status": "error", "message": f"The file {file_path} was not found."}
    except NoCredentialsError:
        return {"status": "error", "message": "Credentials not available."}
    except ClientError as e:
        return {"status": "error", "message": f"Client error: {e}"}


if __name__ == "__main__":
    logger.info("Starting ProteinPreP MCP Server with all tools...")
    #fetch_rcsb("3HTB")
    #Protein_Prep("3HTB.pdb", toDeleteRes=["PO4", "BME"])
    #parametrize_ligand("3HTB_fixer_amber.pdb", "JZ4", 0)
    run_tleap("3HTB_fixer_amber.pdb", "JZ4", "9WvrHU")
    #mcp.run()
    