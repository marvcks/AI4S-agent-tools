#!/usr/bin/env python3
"""
autodE Server
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

import autode as ade
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
    parser = argparse.ArgumentParser(description="autodE Server")
    parser.add_argument('--port', type=int, default=50006, help='Server port (default: 50006)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50006
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("autodE_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)


class ReactionProfileResult(TypedDict):
    reaction_profile: Path
    energies_csv: Path
    methods: Path
    reactants: List[Path]
    products: List[Path]
    transition_states: List[Path]
    imaginary_modes: List[Path]
    message: str


def _run_calculation_sync(
    reactant_smiles: list[str],
    product_smiles: list[str],
    solvent_name: Optional[str],
    n_cores: int,
    energy_type: str,
    calculation_level: str,
    single_point_refinement: bool,
    with_complexes: bool,
) -> ReactionProfileResult:
    """
    Synchronous implementation of reaction profile calculation.
    
    Args:
        reactant_smiles: List of SMILES strings for reactants
        product_smiles: List of SMILES strings for products
        solvent_name: Name of solvent to use
        n_cores: Number of CPU cores to use
        energy_type: Type of energy to calculate (potential, enthalpy, free_energy)
        calculation_level: Level of theory (low, medium, high)
        single_point_refinement: Whether to perform single point refinement
        with_complexes: Whether to include reactant/product complexes
    
    Returns:
        Dictionary with calculation results
    """
    # Set configuration based on calculation level
    ade.Config.n_cores = n_cores
    
    # Configure calculation methods
    if calculation_level == "low":
        # Use xtb for low level calculations
        ade.Config.lcode = "xtb"
    elif calculation_level == "medium":
        # Use DFT with medium basis set
        ade.Config.hcode = "ORCA"
        ade.Config.ORCA.keywords.set_opt_basis_set('def2-SVP')
        ade.Config.ORCA.keywords.sp.basis_set = 'def2-TZVP'
    else:  # high
        # Use DFT with high level basis set
        ade.Config.hcode = "ORCA"
        ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
        ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'
    
    # Parse reactants and products
    reacts = [ade.Reactant(smiles=smile) for smile in reactant_smiles]
    prods = [ade.Product(smiles=smile) for smile in product_smiles]
    
    # Create reaction
    rxn = ade.Reaction(
        *reacts, *prods, 
        name="reaction", 
        solvent_name=solvent_name
    )
    
    # Determine energy calculation parameters
    free_energy = energy_type == "free_energy"
    enthalpy = energy_type == "enthalpy"
    
    # Calculate reaction profile
    rxn.calculate_reaction_profile(
        units="kcal mol-1",
        single_point_refinement=single_point_refinement,
        with_complexes=with_complexes,
        free_energy=free_energy,
        enthalpy=enthalpy
    )
    
    # Collect output files
    output_dir = Path("reaction/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all transition state files
    ts_files = list(output_dir.glob("TS_*.xyz"))
    # Filter out imaginary mode file from TS files
    imag_mode_files = [f for f in ts_files if "imag_mode" in f.name]
    ts_files = [f for f in ts_files if f not in imag_mode_files]
    
    # # If not returning all TS, keep only the lowest energy one
    # if not return_all_transition_states and ts_files:
    #     # Sort by energy (assuming filenames indicate energy)
    #     ts_files = [min(ts_files, key=lambda x: float(x.stem.split('_')[1]))]
    
    return {
        "reaction_profile": Path("reaction_profile.png"),
        "energies_csv": output_dir / "energies.csv",
        "methods": output_dir / "methods.txt",
        # "reactants": [Path(f"{r.name}.xyz") for r in reacts],
        # "products": [Path(f"{p.name}.xyz") for p in prods],
        "transition_states": Path(ts_files[0]),
        "imaginary_modes": Path(imag_mode_files[0]),
        "message": "Reaction profile calculated successfully."
    }


@mcp.tool()
async def calculate_reaction_profile(
    reactant_smiles: list[str] = ["C=CC=C", "C=C"], 
    product_smiles: list[str] = ["C1=CCCCC1"],
    solvent_name: Optional[str] = None,
    n_cores: int = 8,
    energy_type: Literal["potential", "enthalpy", "free_energy"] = "free_energy",
    calculation_level: Literal["low", "medium", "high"] = "medium",
    single_point_refinement: bool = True,
    with_complexes: bool = False,
) -> ReactionProfileResult:
    """
    Calculate the reaction profile for a given set of reactants and products with enhanced control over calculation parameters.
    
    This implementation extends the basic reaction profile calculation with:
    - Different energy types (potential, enthalpy, free energy)
    - Configurable calculation levels (low, medium, high)
    - Option to include reactant/product complexes
    - Option to return all transition states or just the lowest energy one
    
    Parameters:
    ----------
    reactant_smiles : list[str]
        A list of SMILES strings representing the reactants.
    product_smiles : list[str]
        A list of SMILES strings representing the products.
    solvent_name : Optional[str], optional
        The name of the solvent to be used in the reaction (default is None).
    n_cores : int
        The number of CPU cores to use for the calculation (default is 4).
    energy_type : Literal["potential", "enthalpy", "free_energy"]
        Type of energy to calculate for the profile (default is "potential").
    calculation_level : Literal["low", "medium", "high"]
        Level of theory for the calculation (default is "medium").
    single_point_refinement : bool
        Whether to perform single-point energy calculations at a higher level (default is True).
    with_complexes : bool
        Whether to include reactant and product complexes in the profile (default is False).

    
    Returns:
    -------
    ReactionProfileResult
        Dictionary containing paths to generated files and calculation status.
        Includes:
        - reaction_profile: Path to the reaction profile visualization
        - energies_csv: Path to energy data in CSV format
        - methods: Path to file describing computational methods used
        - transition_states: Path to transition state XYZ files
        - imaginary_modes: Path to imaginary mode XYZ files
        - message: Status message of the calculation
    """
    try:
        # Run the calculation in a separate thread to avoid blocking the event loop
        result = await anyio.to_thread.run_sync(
            _run_calculation_sync,
            reactant_smiles,
            product_smiles,
            solvent_name,
            n_cores,
            energy_type,
            calculation_level,
            single_point_refinement,
            with_complexes,
        )
        return result
    except Exception as e:
        logging.error(f"Error calculating reaction profile: {e}")
        return {
            "reaction_profile": None,
            "energies_csv": None,
            "methods": None,
            "transition_states": None,
            "imaginary_modes": None,
            "message": f"Error calculating reaction profile: {str(e)}"
        }


if __name__ == "__main__":
    logging.info("Starting autodE Server with all tools...")
    mcp.run(transport="sse")
 