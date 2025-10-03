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


# 导入MCP相关
from mcp.server.fastmcp import FastMCP

from openmm.app import *
from openmm import *
from openmm.unit import *
import nanoid




def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server for OpenMM Simulation")
    parser.add_argument('--port', type=int, default=50002, help='Server port (default: 50002)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50002
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = FastMCP("openmm_server", host=args.host, port=args.port)

logger = loguru.logger
logger.add("logs/mcp_openmm_{time}.log", level="DEBUG", retention="1 days")
logger.info(f"OpenMM MCP Server initialized on {args.host}:{args.port} with log level {args.log_level}")

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
def Creat_system(prmtop_path:str, inpcrd_path:str, temperature:float=300.0, step_size:float=0.004) -> dict:
    """
    Create OpenMM System Chk from AMBER prmtop and inpcrd files.
    Will do energy minimization and save the state to a chk file as well as a pdb file.
    
    Args:
        prmtop_path: Path to the AMBER prmtop file.
        inpcrd_path: Path to the AMBER inpcrd file.
        temperature: Temperature in Kelvin for the Langevin integrator (default: 300K).
        step_size: Time step in picoseconds for the integrator (default: 0.004 ps).
    Returns:
        dict: A dictionary containing the status and paths to the created system chk files or error message.
    """
    try:
        inpcrd = AmberInpcrdFile(inpcrd_path)
        prmtop = AmberPrmtopFile(prmtop_path)
    except Exception as e:
        logger.error(f"Failed to read prmtop or inpcrd file: {str(e)}")
        return {"status": "error", "message": f"Failed to read prmtop or inpcrd file: {str(e)}"}
    # Create OpenMM System
    try:
        system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        integrator = LangevinIntegrator(temperature*kelvin, 1.0/picosecond, 0.004*picoseconds)
        simulation = Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
    except Exception as e:
        logger.error(f"Failed to create OpenMM System: {str(e)}")
        return {"status": "error", "message": f"Failed to create OpenMM System: {str(e)}"}
    
    # Minimize energy
    try:
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy()
    except Exception as e:
        logger.error(f"Energy minimization failed: {str(e)}")
        return {"status": "error", "message": f"Energy minimization failed: {str(e)}"}
    # Save state
    unique_id = nanoid.generate(size=6)
    state_file = f"{MCP_SCRATCH}/system_{unique_id}.xml"

    simulation.saveState(state_file)

    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    PDBFile.writeFile(simulation.topology, positions, open(f"{MCP_SCRATCH}/system_{unique_id}.pdb", 'w'))
    logger.info(f"System created and saved to {state_file} and {MCP_SCRATCH}/system_{unique_id}.pdb")

    return {"status": "success", "state_file": state_file, "pdb_file": f"{MCP_SCRATCH}/system_{unique_id}.pdb"}

def load_coords_vel(simulation: simulation, state_file: str):
    """Load coordinates and velocities from an OpenMM state XML file."""
    with open(state_file, 'r') as f:
        state = XmlSerializer.deserialize(f.read())
    positions = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True) if state.getVelocities() is not None else None
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    return positions, velocities

@mcp.tool()
def heating_equilibration(prmtop_path: str, state_file:str, temperature:float=300.0, pressure=1.0, step_size:float=0.004, heating_time:float=0.5, eq_time:float=1.0) -> dict:
    """
    Perform heating and equilibration on an OpenMM System from a xml file and AMBER prmtop file.
    All the CA atoms will be restrained during the simulation.
    Will gradually heat the system to the target temperature and then equilibrate it.
    
    Args:
        prmtop_path: Path to the AMBER prmtop file.
        state_file: Path to the OpenMM xml file.
        temperature: Target temperature in Kelvin (default: 300K).
        pressure: Target pressure in bar (default: 1.0 bar).
        step_size: Time step in picoseconds for the integrator (default: 0.004 ps).
        heating_time: Time in nanoseconds for the heating phase (default: 1.0 ns).
        eq_time: Time in nanoseconds for the equilibration phase (default: 1.0 ns).

    Returns:
        dict: A dictionary containing the status and path to the trajectory file or error message.
    """
    try:
        prmtop = AmberPrmtopFile(prmtop_path)
    except Exception as e:
        logger.error(f"Failed to read prmtop file: {str(e)}")
        return {"status": "error", "message": f"Failed to read prmtop file: {str(e)}"}
    # Load OpenMM System from xml file
    try:
        system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        integrator = LangevinIntegrator(0*kelvin, 1.0/picosecond, step_size*picoseconds)
        simulation = Simulation(prmtop.topology, system, integrator)
        load_coords_vel(simulation, state_file)
    except Exception as e:
        logger.error(f"Failed to load OpenMM System from xml file: {str(e)}")
        return {"status": "error", "message": f"Failed to load OpenMM System from xml file: {str(e)}"}
    
    # Restrain CA atoms
    try:
        logger.info("Applying restraints to CA atoms...")
        restraint_force = CustomExternalForce('k * periodicdistance(x, y, z, x0, y0, z0)^2')
        restraint_force.addGlobalParameter('k', 20.0*kilocalories_per_mole/nanometers**2)
        system.addForce(restraint_force)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')
        for atom in prmtop.topology.atoms():
            if atom.name == 'CA':
                pos = simulation.context.getState(getPositions=True).getPositions()[atom.index]
                restraint_force.addParticle(atom.index, [pos.x, pos.y, pos.z])
    except Exception as e:
        logger.error(f"Failed to apply restraints: {str(e)}")
        return {"status": "error", "message": f"Failed to apply restraints: {str(e)}"}
    
    # Heating phase
    total_heating_steps = int((heating_time * 1000) / (step_size ))  # Convert ns to ps
    step_per_window = total_heating_steps // 20
    try:
        for i in range(20):
            current_temp = (i + 1) * (temperature / 20)
            integrator.setTemperature(current_temp*kelvin)
            logger.info(f"Heating to {current_temp} K")
            simulation.step(step_per_window)
    except Exception as e:
        logger.error(f"Heating phase failed: {str(e)}")
        return {"status": "error", "message": f"Heating phase failed: {str(e)}"}
    
    # Equilibration phase
    total_eq_steps = int((eq_time * 1000) / (step_size ))  # Convert ns to ps
    steps_per_window = total_eq_steps // 20
    try:
        integrator.setTemperature(temperature*kelvin)
        system.addForce(MonteCarloBarostat(pressure*bar, temperature*kelvin))
        simulation.context.reinitialize(preserveState=True)
        logger.info(f"Equilibrating at {temperature} K")
        for i in range(20):
            logger.info(f"Equilibration window {i+1}/20")
            simulation.step(steps_per_window)
            simulation.context.setParameter('k', 20.0*(1 - (i + 1)/20)*kilocalories_per_mole/nanometers**2)
    except Exception as e:
        logger.error(f"Equilibration phase failed: {str(e)}")
        return {"status": "error", "message": f"Equilibration phase failed: {str(e)}"}
    
    # Save xml state and pdb
    unique_id = nanoid.generate(size=6)
    final_state_file = f"{MCP_SCRATCH}/equilibrated_{unique_id}.xml"
    simulation.saveState(final_state_file)
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(f"{MCP_SCRATCH}/equilibrated_{unique_id}.pdb", 'w'))
    logger.info(f"Equilibration completed and saved to {final_state_file} and {MCP_SCRATCH}/equilibrated_{unique_id}.pdb")
    return {"status": "success", "state_file": final_state_file, "pdb_file": f"{MCP_SCRATCH}/equilibrated_{unique_id}.pdb"}


@mcp.tool()
def run_production_md(prmtop_path: str, state_file:str, temperature:float=300.0, pressure:float=1.0, step_size:float=0.004, md_time:float=10.0, report_interval:float=2) -> dict:
    """
    Run production MD on an OpenMM System from a xml file and AMBER prmtop file.
    
    Args:
        prmtop_path: Path to the AMBER prmtop file.
        state_file: Path to the OpenMM xml file.
        temperature: Target temperature in Kelvin (default: 300K).
        pressure: Target pressure in bar (default: 1.0 bar).
        step_size: Time step in picoseconds for the integrator (default: 0.004 ps).
        md_time: Time in nanoseconds for the production MD phase (default: 10.0 ns).
        report_interval: Interval in ps to report progress (default: 2 ps).

    Returns:
        dict: A dictionary containing the status and path to the trajectory file or error message.
    """
    try:
        prmtop = AmberPrmtopFile(prmtop_path)
    except Exception as e:
        logger.error(f"Failed to read prmtop file: {str(e)}")
        return {"status": "error", "message": f"Failed to read prmtop file: {str(e)}"}
    # Load OpenMM System from xml file
    try:
        system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        system.addForce(MonteCarloBarostat(pressure*bar, temperature*kelvin))
        integrator = LangevinIntegrator(temperature*kelvin, 1.0/picosecond, step_size*picoseconds)
        simulation = Simulation(prmtop.topology, system, integrator)
        load_coords_vel(simulation, state_file)
        
    except Exception as e:
        logger.error(f"Failed to load OpenMM System from xml file: {str(e)}")
        return {"status": "error", "message": f"Failed to load OpenMM System from xml file: {str(e)}"}
    
    # Run production MD
    total_md_steps = int((md_time * 1000) / (step_size))  # Convert ns to ps
    report_interval_steps = int(report_interval / step_size)
    unique_id = nanoid.generate(size=6)
    traj_file = f"{MCP_SCRATCH}/md_trajectory_{unique_id}.xtc"
    log_file = f"{MCP_SCRATCH}/md_log_{unique_id}.log"
    try:
        logger.info(f"Starting production MD for {md_time} ns...")
        xtc_reporter = XTCReporter(traj_file, report_interval_steps)
        state_reporter = StateDataReporter(log_file, report_interval_steps, step=True,
                                           potentialEnergy=True, temperature=True, density=True, speed=True, remainingTime=True, totalSteps=total_md_steps)
        simulation.reporters.append(xtc_reporter)
        simulation.reporters.append(state_reporter)
        simulation.step(total_md_steps)
        logger.info(f"Production MD completed. Trajectory saved to {traj_file} and log to {log_file}")
    except Exception as e:
        logger.error(f"Production MD failed: {str(e)}")
        return {"status": "error", "message": f"Production MD failed: {str(e)}"}
    
    chk_file = f"{MCP_SCRATCH}/md_final_state_{unique_id}.xml"
    simulation.saveState(chk_file)
    logger.info(f"Final state saved to {chk_file}")
    pdb_file = f"{MCP_SCRATCH}/md_final_structure_{unique_id}.pdb"
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(pdb_file, 'w'))
    logger.info(f"Final structure saved to {pdb_file}")
    return {"status": "success", "trajectory_file": traj_file, "log_file": log_file, "final_state_file": chk_file, "final_pdb_file": pdb_file}



if __name__ == "__main__":
    logger.info("Starting OpenMM MCP Server with all tools...")
    #Creat_system("3HTB_fixer_amber_solv_hmr.parm7", "3HTB_fixer_amber_solv_hmr.rst7")
    #heating_equilibration("3HTB_fixer_amber_solv_hmr.parm7", "system_cj8hGr.xml")
    #run_production_md("3HTB_fixer_amber_solv_hmr.parm7", "equilibrated_iZMTFo.xml", md_time=5.0, report_interval=50.0)
    mcp.run()
    
