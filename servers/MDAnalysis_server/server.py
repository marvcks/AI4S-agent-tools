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
from pathlib import Path
import subprocess
import numpy as np
import loguru
import matplotlib.pyplot as plt

# 导入MCP相关
from dp.agent.server import CalculationMCPServer

from openmm.app import *
from openmm import *
from openmm.unit import *
import nanoid
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms
import MDAnalysis.transformations as trans
import boto3

import molviewspec as mvs

from safety_check import is_safe


def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server for MDAnalysis")
    parser.add_argument('--port', type=int, default=50003, help='Server port (default: 50003)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50003
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = CalculationMCPServer("md_analysis_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_mdanalysis_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"MDAnalysis MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")


MCP_SCRATCH = os.getenv("MCP_SCRATCH", "/tmp")


# Define tools at module level
@mcp.tool()
def prepare_trajectories(
    prmtop_path: Path,   
    trajectory_path: Path,
    selection: str = "backbone") -> Dict[str, Any]:
    """
    Prepare and align trajectories using MDAnalysis.
    It will unwrap and align the trajectory to the first frame based on the given selection.

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        selection (str): Atom selection string for alignment using MDAnalysis syntax. Default is "backbone".

    Returns:
        Dict[str, Any]: Dictionary containing status and path to the aligned trajectory file in XTC format.
    """
    logger.info(f"Aligning trajectory using topology '{prmtop_path}' and trajectory '{trajectory_path}' with selection '{selection}'")
    
    # Load the universe
    u = mda.Universe(prmtop_path, trajectory_path)
    logger.info(f"Loaded universe with {len(u.atoms)} atoms and {len(u.trajectory)} frames")

    protein_atoms = u.select_atoms("protein")
    not_protein = u.select_atoms('not protein')
    transforms = [trans.unwrap(protein_atoms),
                  trans.center_in_box(protein_atoms, wrap=True),
                  trans.wrap(not_protein)]
    
    u.trajectory.add_transformations(*transforms)

    u.trajectory[0]
    u_ref = u.select_atoms(selection)
    logger.info(f"Reference selection has {len(u_ref)} atoms")
    u_ref.translate(-u_ref.center_of_mass())

    # Prepare output path
    id = nanoid.generate(size=6)
    aligned_trajectory_path = Path(MCP_SCRATCH) / f"aligned_trajectory_{id}.xtc"
    logger.info(f"Saving aligned trajectory to '{aligned_trajectory_path}'")
    try:
        align.AlignTraj(u, u_ref, select=selection, filename=aligned_trajectory_path).run()
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "aligned_trajectory_path": aligned_trajectory_path}


@mcp.tool()
def calculate_rmsd(
    prmtop_path: Path,   
    trajectory_path: Path,
    selection: str = "backbone") -> Dict[str, Any]:
    """
    Calculate RMSD of a selection over a trajectory to the first frame using MDAnalysis.

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        selection (str): Atom selection string for RMSD calculation using MDAnalysis syntax. Default is "backbone".

    Returns:
        Dict[str, Any]: Dictionary containing time and the path of a npz file with x: time and y: RMSD arrays in Angstrom.
    """
    logger.info(f"Calculating RMSD for selection '{selection}' using topology '{prmtop_path}' and trajectory '{trajectory_path}'")
    
    # Load the universe
    try:
        u = mda.Universe(prmtop_path, trajectory_path)
        
        logger.info(f"Loaded universe with {len(u.atoms)} atoms and {len(u.trajectory)} frames")

        R = rms.RMSD(u, select=selection, ref_frame=0)
        R.run()

        rmsd = R.results.rmsd.T
        time = rmsd[1]
        rmsd_values = rmsd[2]
        logger.info(f"Calculated RMSD for {len(rmsd_values)} frames")
        id = nanoid.generate(size=6)
        rmsd_output_path = Path(MCP_SCRATCH) / f"rmsd_{id}.npz"
        logger.debug(rmsd_values)
        np.savez(rmsd_output_path, x=time, y=rmsd_values)
        logger.info(f"Saved RMSD data to '{rmsd_output_path}'")
    except Exception as e:
        logger.error(f"RMSD calculation failed: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "rmsd_output_path": rmsd_output_path}
    
@mcp.tool()
def calculate_rmsf(
    prmtop_path: Path,   
    trajectory_path: Path) -> Dict[str, Any]:
    """
    Calculate RMSF of protein over a trajectory using MDAnalysis.

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).

    Returns:
        Dict[str, Any]: Dictionary containing atom indices and the path of a npz file with x: residue indices and y: RMSF values in Angstrom.
    """
    selection = "protein"
    logger.info(f"Calculating RMSF for selection '{selection}' using topology '{prmtop_path}' and trajectory '{trajectory_path}'")
    
    # Load the universe
    try:
        u = mda.Universe(prmtop_path, trajectory_path, in_memory=True)
        atoms = u.select_atoms("protein")
        
        logger.info(f"Loaded universe with {len(u.atoms)} atoms and {len(u.trajectory)} frames")

        protein_atoms = u.select_atoms(selection)
        logger.info(f"Selection has {len(protein_atoms)} atoms")

        ref_coordinates = u.trajectory.timeseries(asel=atoms).mean(axis=1)
        reference = mda.Merge(atoms).load_new(ref_coordinates[:, None, :], order='afc')
        aligner = align.AlignTraj(u, reference, select=selection, in_memory=True).run()

        calphas = u.select_atoms("name CA")
        R = rms.RMSF(calphas, verbose=True).run()

        resume = calphas.resnums
        rmsf_values = R.results.rmsf
        logger.info(f"Calculated RMSF for {len(rmsf_values)} atoms")
        id = nanoid.generate(size=6)
        rmsf_output_path = Path(MCP_SCRATCH) / f"rmsf_{id}.npz"
        logger.debug(rmsf_values)
        np.savez(rmsf_output_path, x=resume, y=rmsf_values)
        logger.info(f"Saved RMSF data to '{rmsf_output_path}'")
    except Exception as e:
        logger.error(f"RMSF calculation failed: {e}")
        return {"status": "error", "message": str(e)}   
    return {"status": "success", "rmsf_output_path": rmsf_output_path}

@mcp.tool()
def calculate_Rg(
    prmtop_path: Path,
    trajectory_path: Path,
    selection: str = "protein") -> Dict[str, Any]:
    """
    Calculate Radius of Gyration (Rg) of a selection over a trajectory using MDAnalysis.

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        selection (str): Atom selection string for Rg calculation using MDAnalysis syntax. Default is "backbone".

    Returns:
        Dict[str, Any]: Dictionary containing time and the path of a npz file with x: time and y: Rg arrays in Angstrom.
    """
    logger.info(f"Calculating Rg for selection '{selection}' using topology '{prmtop_path}' and trajectory '{trajectory_path}'")
    
    # Load the universe
    try:
        u = mda.Universe(prmtop_path, trajectory_path)
        
        logger.info(f"Loaded universe with {len(u.atoms)} atoms and {len(u.trajectory)} frames")

        Rg_values = []
        times = []
        selection_atoms = u.select_atoms(selection)
        for ts in u.trajectory:
            rg = selection_atoms.radius_of_gyration()
            Rg_values.append(rg)
            times.append(u.trajectory.time)
        
        Rg_values = np.array(Rg_values)
        times = np.array(times)
        logger.info(f"Calculated Rg for {len(Rg_values)} frames")
        id = nanoid.generate(size=6)
        rg_output_path = Path(MCP_SCRATCH) / f"rg_{id}.npz"
        logger.debug(Rg_values)
        np.savez(rg_output_path, x=times, y=Rg_values)
        logger.info(f"Saved Rg data to '{rg_output_path}'")
    except Exception as e:
        logger.error(f"Rg calculation failed: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success",  "rg_output_path": rg_output_path}

@mcp.tool()
def calculate_distance(
    prmtop_path: Path,
    trajectory_path: Path,
    selection1: str,
    selection2: str) -> Dict[str, Any]:
    """
    Calculate distance between two selections over a trajectory using MDAnalysis.

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        selection1 (str): Atom selection string for the first group using MDAnalysis syntax.
        selection2 (str): Atom selection string for the second group using MDAnalysis syntax.

    Returns:
        Dict[str, Any]: Dictionary containing time and the path of a npz file with x: time and y: distance arrays in Angstrom.
    """
    logger.info(f"Calculating distance between selections '{selection1}' and '{selection2}' using topology '{prmtop_path}' and trajectory '{trajectory_path}'")
    
    # Load the universe
    try:
        u = mda.Universe(prmtop_path, trajectory_path)
        
        logger.info(f"Loaded universe with {len(u.atoms)} atoms and {len(u.trajectory)} frames")

        distances = []
        times = []
        sel1_atoms = u.select_atoms(selection1)
        sel2_atoms = u.select_atoms(selection2)
        for ts in u.trajectory:
            dist = np.linalg.norm(sel1_atoms.center_of_mass() - sel2_atoms.center_of_mass())
            distances.append(dist)
            times.append(u.trajectory.time)
        
        distances = np.array(distances)
        times = np.array(times)
        logger.info(f"Calculated distances for {len(distances)} frames")
        id = nanoid.generate(size=6)
        distance_output_path = Path(MCP_SCRATCH) / f"distance_{id}.npz"
        logger.debug(distances)
        np.savez(distance_output_path, x=times, y=distances)
        logger.info(f"Saved distance data to '{distance_output_path}'")
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success",  "distance_output_path": distance_output_path}


@mcp.tool()
def calculate_mm_gbsa(
    prmtop_path: Path,
    trajectory_path: Path,
    ligand_resname: str = "LIG",
    interval: int = 1,
    igb: int = 5,
    salt_conc: float = 0.1) -> Dict[str, Any]:
    """
    Calculate MM-GBSA free energy over a trajectory using MMPBSA.py

    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        ligand_resname (str): Residue name of the ligand in the topology. Default is "LIG".
        interval (int): Interval to sample frames from the trajectory. Default is 1 (use all frames). Please make sure the trajectory is not too large to avoid long computation time.
        igb (int): GB model to use. Default is 5.
        salt_conc (float): Salt concentration in mol/L. Default is 0.1.

    Returns:
        Dict[str, Any]: Dictionary containing status, mmgbsa_energy in kcal/mol, stddev in kcal/mol.
    """
    logger.info(f"Calculating MM-GBSA using topology '{prmtop_path}' and trajectory '{trajectory_path}' with igb={igb} and salt_conc={salt_conc}")
    
    # ante-MMPBSA.py -p prmtop_path -c prmtop_complex_path -r prmtop_receptor_path -l prmtop_ligand_path -s ":WAT,Na+,Cl-" -n ligand_resname
    with open(f"{MCP_SCRATCH}/mmgbsa.in", 'w') as f:
        f.write(f"""&general
endframe=9999, interval={interval}, verbose=1, keep_files=0,
/
&gb
igb={igb}, saltcon={salt_conc},
/
""")
    try:
        id = nanoid.generate(size=6)
        complex_prmtop_path = os.path.join(MCP_SCRATCH, f"complex_{id}.prmtop")
        receptor_prmtop_path = os.path.join(MCP_SCRATCH, f"receptor_{id}.prmtop")
        ligand_prmtop_path = os.path.join(MCP_SCRATCH, f"ligand_{id}.prmtop")
        logger.info("Generating prmtop files for complex, receptor, and ligand")
        ante_cmd = [
            "ante-MMPBSA.py",
            "-p", prmtop_path,
            "-c", complex_prmtop_path,
            "-r", receptor_prmtop_path,
            "-l", ligand_prmtop_path,
            "-s", ":WAT,Na+,Cl-",
            "-n", f":{ligand_resname}"
        ]
        logger.debug(f"Running command: {' '.join(ante_cmd)}")
        subprocess.run(ante_cmd, check=True)

        mmgbsa_output_path = os.path.join(MCP_SCRATCH, f"mmgbsa_{id}.dat")
        logger.info(f"Running MMPBSA.py and saving output to '{mmgbsa_output_path}'")
        mmpbsa_cmd = [
            "MMPBSA.py",
            "-O",
            "-prefix", f"_mmgbsa_{id}",
            "-i", f"{MCP_SCRATCH}/mmgbsa.in",
            "-cp", complex_prmtop_path,
            "-rp", receptor_prmtop_path,
            "-lp", ligand_prmtop_path,
            "-sp", prmtop_path,
            "-y", trajectory_path,
            "-o", mmgbsa_output_path
        ]
        logger.debug(f"Running command: {' '.join(mmpbsa_cmd)}")
        subprocess.run(mmpbsa_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"MMPBSA calculation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"MMPBSA calculation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    # Parse the output file to extract energies
    try: 
        with open(mmgbsa_output_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("DELTA TOTAL   "):
                    energy = float(line.split()[2])
                    stddev = float(line.split()[3])
                    logger.info(f"Extracted MM-GBSA energy: {energy} ± {stddev} kcal/mol")
                    break
        return {"status": "success", "mmgbsa_energy_kcal_per_mol": energy, "stddev_kcal_per_mol": stddev}
    except Exception as e:
        logger.error(f"Failed to parse MMPBSA output: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    
        
@mcp.tool()
def export_molstar_html(
    prmtop_path: Path,
    trajectory_path: Path,
    time: float,
    selection: str = "not (resname HOH or resname WAT)",) -> Dict[str, Any]:
    """
    Export an interactive Mol* HTML visualization of a trajectory using molviewspec.
    Args:
        prmtop_path (Path): Path to the topology file (e.g., Amber prmtop).
        trajectory_path (Path): Path to the trajectory file (e.g., DCD, XTC).
        time (float): Time in picoseconds to extract the frame for visualization.
        selection (str): Atom selection string for visualization using MDAnalysis syntax. Default is "not (resname HOH or resname WAT)".
    Returns:
        Dict[str, Any]: Dictionary containing status and path to the generated Mol* HTML file.
    """
    logger.info(f"Exporting Mol* HTML visualization for selection '{selection}'...")
    
    try:
        u = mda.Universe(prmtop_path, trajectory_path)
        
        times = [ts.time for ts in u.trajectory]
        closest_frame_idx = np.argmin(np.abs(np.array(times) - time))
        u.trajectory[closest_frame_idx] # Move to the target frame
        
        selected_atoms = u.select_atoms(selection)
        logger.info(f"Selected {len(selected_atoms)} atoms at frame {closest_frame_idx}")
        
        # Correctly write selected atoms to an in-memory string
        selected_atoms.write("temp.pdb")
        with open("temp.pdb", 'r') as f:
            pdb_content = f.read()
        # Build the Mol* viewer specification
        builder = mvs.create_builder()
        structure =   builder.download(url="temp.pdb").parse(format="pdb").model_structure()
        structure.component(selector="protein").representation(type="cartoon").color()
        structure.component(selector="ligand").representation(type="ball_and_stick", size_factor=0.5).color(custom={"molstar_color_theme_name": "element-symbol"})
  
        html_content = builder.molstar_html(data={"temp.pdb": pdb_content})
        
        id = nanoid.generate(size=6)
        html_output_path = Path(MCP_SCRATCH) / f"molstar_{id}.html"
        with open(html_output_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved Mol* HTML visualization to '{html_output_path}'")
        
    except Exception as e:
        logger.error(f"Mol* HTML export failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

    return {"status": "success", "html_output_path": html_output_path}


@mcp.tool()
def plot_picture(
    npz_path: List[Path],
    Legend: List[str] = [],
    x_label: str = "Time (ps)",
    y_label: str = "Value",
    title: str = "Plot",
) -> Dict[str, Any]:
    """
    Generate a plot from one or more npz files containing x and y data arrays.

    Args:
        npz_path (List[Path]): List of paths to npz files. Each file should contain 'x' and 'y' arrays.
        Legend (List[str]): List of legend labels for each dataset. If None, legends will not be shown.
        x_label (str): Label for the x-axis. Default is "Time (ps)".
        y_label (str): Label for the y-axis. Default is "Value".
        title (str): Title of the plot. Default is "Plot".

    Returns:
        Dict[str, Any]: Dictionary containing status and path to the generated plot image in PNG format.
    """
    logger.info(f"Generating plot from npz files: {npz_path}")
    legend_flag = len(Legend) == len(npz_path)
    try:
        plt.figure(figsize=(10, 6))
        for i, path in enumerate(npz_path):
            data = np.load(path)
            x = data['x']
            y = data['y']
            if legend_flag:
                plt.plot(x, y, label=Legend[i])
            else:
                plt.plot(x, y)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if legend_flag:
            plt.legend()

        plt.grid(True)

        id = nanoid.generate(size=6)
        plot_output_path = Path(MCP_SCRATCH) / f"plot_{id}.png"
        plt.savefig(plot_output_path, dpi=200)
        plt.close()
        logger.info(f"Saved plot to '{plot_output_path}'")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

    return {"status": "success", "plot_output_path": plot_output_path}

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



if __name__ == "__main__":
    logger.info("Starting OpenMM MCP Server with all tools...")
    #prepare_trajectories("3HTB_fixer_amber_solv_hmr.parm7", "md_trajectory_RwCjbL.xtc")
    #calculate_rmsd("3HTB_fixer_amber_solv_hmr.parm7", "md_trajectory_RwCjbL.xtc")
    #calculate_rmsf("3HTB_fixer_amber_solv_hmr.parm7", "aligned_trajectory_P1FMFL.xtc")
    #calculate_Rg("3HTB_fixer_amber_solv_hmr.parm7", "aligned_trajectory_P1FMFL.xtc")
    #export_molstar_html("3HTB_fixer_amber_solv_hmr.parm7", "aligned_trajectory_P1FMFL.xtc", time=0.0)
    #calculate_mm_gbsa("3HTB_fixer_amber_solv_hmr.parm7", "md_trajectory_RwCjbL.xtc", ligand_resname="JZ4", interval=10)
    #plot_picture(["rmsd_F4la-I.npz"])
    mcp.run(transport="sse")
    
