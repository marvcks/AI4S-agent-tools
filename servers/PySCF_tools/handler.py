#import pyscf and related modules
import subprocess
import pyscf
from pyscf import gto, scf, dft, mp, tools, tddft
from pyscf.geomopt.geometric_solver import optimize
from pyscf import solvent
from pyscf.hessian import thermo
from pyscf.prop import infrared, nmr

import numpy as np
from typing import Dict, Any

import os
from pathlib import Path

import hashlib
import nanoid

import pathlib
from urllib.parse import urlparse

os.environ["MCP_SCRATCH"] = "/home/zhouoh/scratch" 
SCRATCH_DIR = Path(os.getenv("MCP_SCRATCH"))

def build_mol(job: Dict[str, Any], logger) -> gto.M:
    """
    根据输入的job数据构建pyscf的分子对象
    """
    molecule_data = job.get("molecule")
    basis_set = job.get("basis_set", "def2-svp")
    logger.info(f"Building molecule with basis set: {basis_set}")

    if not molecule_data:
        logger.error("Molecule data is required")
        raise ValueError("Molecule data is required")
    try:
        xyz_path = molecule_data    
        if molecule_data.startswith("local://"):
            xyz_path = urlparse(molecule_data).path
        elif molecule_data.startswith("file://"):
            xyz_path = urlparse(molecule_data).path
        logger.info(f"Parsed molecule path: {xyz_path}")
        #convert 
        mol = gto.M(atom=xyz_path)
    except Exception as e:
        logger.error(f"Failed to read molecule file: {e}")
        raise ValueError("Failed to read molecule file")
    try:
        mol.basis = basis_set
    except Exception as e:
        logger.error(f"Unsupported basis set: {basis_set}")
        raise ValueError(f"Unsupported basis_set: {basis_set}")
    mol.charge = job.get("charge", 0)
    mol.spin = job.get("spin", 1) - 1
    logger.info(f"Built molecule with {len(mol.atom)} atoms, charge={mol.charge}, spin={mol.spin}, basis={mol.basis}")
    return mol

def Check_Restart(mol_xyz: str, logger) -> str:
    """
    检查是否有相同的计算已经完成，如果有则返回结果文件路径
    """
    hash_object = hashlib.md5(mol_xyz.encode())
    hash_hex = hash_object.hexdigest()
    result_file = SCRATCH_DIR / f"{hash_hex}.chk"
    if result_file.exists():
        logger.info(f"Found existing result file: {result_file}")
        return str(result_file)
    return result_file

def SinglePointHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    处理单点能量计算任务
    """
    try:
        # 解析输入数据
        mol = build_mol(job, logger)
    except Exception as e:
        return {"error": str(e)}
    method = job.get("method", "b3lyp-d3bj")
    max_iterations = job.get("max_iterations", 128)
    solvent = job.get("solvent", "water")
    solvent_model = job.get("solvent_model", None)
    solvent_dielectric = job.get("solvent_dielectric", None)

    if solvent_model not in [None, "PCM", "SMD"]:
        logger.error(f"Unsupported solvent model: {solvent_model}")
        return {"error": f"Unsupported solvent model: {solvent_model}"}

    if solvent_model == "PCM" and not solvent_dielectric:
        solvent_dielectric = 78.4  # Default to water dielectric constant

    chkfile = Check_Restart(mol.atom, logger)
    restart_flag = False
    if chkfile and Path(chkfile).exists():
        restart_flag = True
        logger.info(f"Restarting from checkpoint file: {chkfile}")
    energy = 0.0
    if method in ["HF", "RHF", "UHF", "ROHF"]:
        mf = scf.HF(mol).density_fit()
        mf.chkfile = chkfile
        mf.max_cycle = max_iterations
        if restart_flag:
            mf.init_guess = "chkfile"

    elif method in ["MP2", "RMP2", "UMP2"]:
        mf = scf.HF(mol).density_fit()
        mf.chkfile = chkfile
        mf.max_cycle = max_iterations
        if restart_flag:
            mf.init_guess = "chkfile"
        if solvent_model == "PCM":
            mf = mf.PCM()
            mf.with_solvent_method = 'IEF-PCM'
            mf.with_solvent.eps = solvent_dielectric
        elif solvent_model == "SMD":
            mf = mf.SMD()
            mf.with_solvent.solvent = solvent
        try:
            mf.kernel()
        except Exception as e:
            logger.error(f"SCF did not converge: {e}")
            return {"error": "SCF did not converge"}
        try:
            mf = mp.MP2(mf)
            energy = mf.run().e_tot
            result = {"energy": energy}
            return result
        except Exception as e:
            logger.error(f"MP2 calculation failed: {e}")
            return {"error": "MP2 calculation failed"}
    else:
        try:
            mf = dft.KS(mol).density_fit()
            mf.xc = method
            mf.chkfile = chkfile
            mf.max_cycle = max_iterations
        except Exception as e:
            logger.error(f"Unsupported method: {method}")
            return {"error": f"Unsupported method: {method}"}
        if restart_flag:
            mf.init_guess = "chkfile"

    if solvent_model == "PCM":
        mf = mf.PCM()
        mf.with_solvent_method = 'IEF-PCM'
        mf.with_solvent.eps = solvent_dielectric
    elif solvent_model == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = solvent

    try:
        energy = mf.kernel()
    except Exception as e:
        logger.error(f"SCF did not converge: {e}")
        return {"error": "SCF did not converge"}
    
    energy = float(energy)
    
    result = {"energy": energy}
    return result


def _SinglePointHandler(job: Dict[str, Any], logger) -> Any:
    try:
        # 解析输入数据
        mol = build_mol(job, logger)
    except Exception as e:
        return {"error": str(e)}
    method = job.get("method", "b3lyp-d3bj")
    max_iterations = job.get("max_iterations", 128)
    solvent = job.get("solvent", "water")
    solvent_model = job.get("solvent_model", None)
    solvent_dielectric = job.get("solvent_dielectric", None)

    if solvent_model not in [None, "PCM", "SMD"]:
        logger.error(f"Unsupported solvent model: {solvent_model}")
        return {"error": f"Unsupported solvent model: {solvent_model}"}

    if solvent_model == "PCM" and not solvent_dielectric:
        solvent_dielectric = 78.4  # Default to water dielectric constant

    chkfile = Check_Restart(mol.atom, logger)
    restart_flag = False
    if chkfile and Path(chkfile).exists():
        restart_flag = True
        logger.info(f"Restarting from checkpoint file: {chkfile}")
    energy = 0.0
    if method in ["HF", "RHF", "UHF", "ROHF"]:
        mf = scf.HF(mol).density_fit()
        mf.chkfile = chkfile
        mf.max_cycle = max_iterations
        if restart_flag:
            mf.init_guess = "chkfile"

    elif method in ["MP2", "RMP2", "UMP2"]:
        mf = scf.HF(mol).density_fit()
        mf.chkfile = chkfile
        mf.max_cycle = max_iterations
        if restart_flag:
            mf.init_guess = "chkfile"
        if solvent_model == "PCM":
            mf = mf.PCM()
            mf.with_solvent_method = 'IEF-PCM'
            mf.with_solvent.eps = solvent_dielectric
        elif solvent_model == "SMD":
            mf = mf.SMD()
            mf.with_solvent.solvent = solvent
        try:
            mf.kernel()
        except Exception as e:
            logger.error(f"SCF did not converge: {e}")
            return {"error": "SCF did not converge"}
        try:
            mf = mp.MP2(mf)
            energy = mf.run().e_tot
            result = {"energy": energy}
            return result
        except Exception as e:
            logger.error(f"MP2 calculation failed: {e}")
            return {"error": "MP2 calculation failed"}
    else:
        try:
            mf = dft.KS(mol).density_fit()
            mf.xc = method
            mf.chkfile = chkfile
            mf.max_cycle = max_iterations
        except Exception as e:
            logger.error(f"Unsupported method: {method}")
            return {"error": f"Unsupported method: {method}"}
        if restart_flag:
            mf.init_guess = "chkfile"

    if solvent_model == "PCM":
        mf = mf.PCM()
        mf.with_solvent_method = 'IEF-PCM'
        mf.with_solvent.eps = solvent_dielectric
    elif solvent_model == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = solvent

    try:
        energy = mf.kernel()
    except Exception as e:
        logger.error(f"SCF did not converge: {e}")
        return {"error": "SCF did not converge"}
    
    return mf


def GeometryOptimizationHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    处理几何优化任务
    """
    result = {}

    
    max_steps = job.get("max_optimization_steps", 100)
    threshold = job.get("optimization_threshold", "normal")

    conv_params = { # These are the default settings
        'convergence_energy': 1e-6,  # Eh
        'convergence_grms': 3e-4,    # Eh/Bohr
        'convergence_gmax': 4.5e-4,  # Eh/Bohr
        'convergence_drms': 1.2e-3,  # Angstrom
        'convergence_dmax': 1.8e-3,  # Angstrom
    }

    if threshold == "loose":
        conv_params = {
            'convergence_energy': 1e-5,
            'convergence_grms': 1.2e-3,
            'convergence_gmax': 1.8e-3,
            'convergence_drms': 4.0e-3,
            'convergence_dmax': 6.0e-3,
        }

    elif threshold == "tight":
        conv_params = {
            'convergence_energy': 1e-7,
            'convergence_grms': 1e-4,
            'convergence_gmax': 1.5e-4,
            'convergence_drms': 4.0e-4,
            'convergence_dmax': 6.0e-4,
        }

    if threshold not in ["loose", "normal", "tight"]:
        logger.error(f"Unsupported optimization threshold: {threshold}")
        return {"error": f"Unsupported optimization threshold: {threshold}"}

    mf = _SinglePointHandler(job, logger)

    if isinstance(mf, dict) and "error" in mf:
        return mf

    mol_eq = optimize(mf, maxsteps=max_steps, **conv_params)

    #write optimized geometry to xyz file
    hashed = hashlib.md5(mol_eq.tostring("xyz").encode()).hexdigest()
    opt_xyz_file = SCRATCH_DIR / f"opt_{hashed}.xyz"
    with open(opt_xyz_file, "w") as f:
        f.write(mol_eq.tostring("xyz"))

    mf.mol = mol_eq
    energy = mf.kernel()
    energy = float(energy)
    result["energy"] = energy
    logger.info(f"Optimization completed. Final energy: {energy} Ha")
    logger.info(f"Final optimized geometry written to: {str(opt_xyz_file)}")
    result["optimized_geometry"] = str(opt_xyz_file)
    return result

def Get_Calculation_type(job: Dict[str, Any]) -> str:
    Multiplicity = job.get("spin", 1)
    a = ""
    if Multiplicity == 1:
        a = "r"
    elif Multiplicity == 2:
        a = "u"
    
    if job.get("method", "b3lyp-d3bj") in ["HF", "RHF", "UHF", "ROHF"]:
        method = f"{a}hf"
    elif job.get("method", "b3lyp-d3bj") in ["MP2", "RMP2", "UMP2"]:
        method = f"{a}hf"
    else:
        method = f"{a}ks"
    return method


def FrequencyAnalysisHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    处理频率分析任务
    """
    result = {}
    T = job.get("Temperature", 298.15)
    P = job.get("Pressure", 101325)
    if T <= 0 or P <= 0:
        logger.error("Temperature and Pressure must be positive values")
        return {"error": "Temperature and Pressure must be positive values"}

    mf = _SinglePointHandler(job, logger)

    if isinstance(mf, dict) and "error" in mf:
        return mf
    
    calc_type = Get_Calculation_type(job)
    try:
        if calc_type == "rhf":
            mf_ir = infrared.rhf.Infrared(mf).run()
        elif calc_type == "uhf":
            mf_ir = infrared.uhf.Infrared(mf).run()
        elif calc_type == "rks":
            mf_ir = infrared.rks.Infrared(mf).run()
        elif calc_type == "uks":
            mf_ir = infrared.uks.Infrared(mf).run()
    except Exception as e:
        logger.error(f"Frequency calculation failed: {e}")
        return {"error": f"Frequency calculation failed: {e}"}


    intersensities = mf_ir.ir_inten
    result["frequencies"] = mf_ir.vib_dict["freq_wavenumber"].tolist()
    result["intensities"] = intersensities.tolist()
    logger.info(f"Frequencies: {result['frequencies']}")
    logger.info(f"Intensities: {result['intensities']}")
    #dump frequencies and intensities to a npz file
    id = nanoid.generate()
    freq_file = SCRATCH_DIR / f"freq_{id}.npz"
    logger.info(f"Saving frequencies and intensities to {freq_file}")
    np.savez(freq_file, x=result["frequencies"], y=result["intensities"])
    result["frequency_file"] = str(freq_file)   
    logger.info(f"Calculating thermodynamic properties at T={T} K and P={P} Pa")


    thermo_info = thermo.thermo(mf, mf_ir.vib_dict["freq_au"], T, P)
    
    result['ZPE'] = float(thermo_info['ZPE'][0])
    E = float(thermo_info['E0'][0])
    dH = float(thermo_info['H_tot'][0]) - thermo_info['E0'][0]
    dG = float(thermo_info['G_tot'][0]) - thermo_info['E0'][0]
    result['E'] = E
    result['H_corr'] = float(dH)
    result['G_corr'] = float(dG)
    return result

def Hirshfeld_Analysis(molden_file: Path, logger) -> list:

    result = []
    if not molden_file.exists():
        logger.error(f"Molden file does not exist: {molden_file}")
        return []
    prefix = molden_file.stem
    cmd = f"echo -e '\n7\n1\n1\ny\n0\nq' | Multiwfn {molden_file}"
    print(cmd)
    mwfn_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(mwfn_result.stdout)
    with open(f"{prefix}.chg", "r") as f:
        lines = f.readlines()
        for line in lines:
            result.append(float(line.split()[4]))
    return result

def PropAnalysisHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:  

    prop_type = job.get("parameters", {}).get("population_properties", [])
    print(prop_type)
    if not prop_type:
        logger.error("No population properties specified")
        return {"error": "No population properties specified"}
    mf = _SinglePointHandler(job, logger) 

    if isinstance(mf, dict) and "error" in mf:
        return mf
    
    result = {}
    for prop in prop_type:
        if prop == "charges":
            charge_types = job.get("parameters", {}).get("population_analysis_method", "Mulliken")
            if charge_types == "Mulliken":
                charges = mf.mulliken_pop()
                result["charges"] = [float(c) for c in charges[1]]
            elif charge_types == "Hirshfeld":
                #write the wavefunction to a molden file
                molden_file = SCRATCH_DIR / "temp.molden"
                tools.molden.from_scf(mf, str(molden_file))
                charges = Hirshfeld_Analysis(molden_file, logger)
                if charges:
                    result["charges"] = charges
                else:
                    result["charges"] = "Hirshfeld analysis failed"

        elif prop == "dipole":
            dipole = mf.dip_moment()
            dipole_magnitude = np.linalg.norm(dipole)
            result["dipole_moment"] = float(dipole_magnitude)
        elif prop == "orbitals":
            orbitals = mf.mo_energy
            result["orbitals"] = [float(e) for e in orbitals]
        else:
            logger.warning(f"Unsupported property: {prop}")
            result[f"{prop}"] = f"Unsupported property: {prop}"
    return result

def NMRHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    处理NMR化学位移计算任务
    """
    result = {}
    mf = _SinglePointHandler(job, logger)

    if isinstance(mf, dict) and "error" in mf:
        return mf
    
    calc_type = Get_Calculation_type(job)
    try:
        if calc_type == "rhf":
            nmr_calc = nmr.rhf.NMR(mf).run()
        elif calc_type == "uhf":
            nmr_calc = nmr.uhf.NMR(mf).run()
        elif calc_type == "rks":
            nmr_calc = nmr.rks.NMR(mf).run()
        elif calc_type == "uks":
            nmr_calc = nmr.uks.NMR(mf).run()
    except Exception as e:
        logger.error(f"NMR calculation failed: {e}")
        return {"error": f"NMR calculation failed: {e}"}
    
    sigma_list = []
    shielding = nmr_calc.shielding()
    for i in range(len(shielding)):
        B = shielding[i]
        sigma = np.trace(B) / 3.0
        sigma_list.append(sigma)

    result["nmr_shifts"] = [float(s) for s in sigma_list]
    #save nmr shifts to a npz file
    id = nanoid.generate()
    num_shifts = len(result["nmr_shifts"])
    y = np.ones(num_shifts) 
    nmr_file = SCRATCH_DIR / f"nmr_{id}.npz"
    np.savez(nmr_file, x=result["nmr_shifts"], y= y)
    return result

def TDDFTHandler(job: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    处理TDDFT计算任务
    """
    result = {}
    n_states = job.get("n_states", 5)
    if n_states <= 0:
        logger.error("Number of excited states must be a positive integer")
        return {"error": "Number of excited states must be a positive integer"}

    mf = _SinglePointHandler(job, logger)

    if isinstance(mf, dict) and "error" in mf:
        return mf
    
    mf_td = tddft.TDA(mf)
    mf_td.nstates = n_states
    try:
        mf_td.kernel()
    except Exception as e:
        logger.error(f"TDDFT calculation failed: {e}")
        return {"error": f"TDDFT calculation failed: {e}"}
    e_list = mf_td.e
    e_list_ev = [float(e * 27.2114) for e in e_list]  # Convert from Hartree to eV
    result["excitation_energies"] = e_list_ev
    wavelengths = [float(1239.84193 / e) if e != 0 else 0 for e in e_list_ev]  # Convert eV to nm
    result["excitation_wavelengths"] = wavelengths
    osc_strengths = mf_td.oscillator_strength()
    result["oscillator_strengths"] = osc_strengths.tolist()
    #save excitation energies and oscillator strengths to a npz file
    id = nanoid.generate()
    tddft_file = SCRATCH_DIR / f"tddft_{id}.npz"
    np.savez(tddft_file, x=result["excitation_energies"], y=result["oscillator_strengths"])
    result["tddft_file"] = str(tddft_file)
    return result
    
if __name__ == "__main__":
    import loguru
    logger = loguru.logger
    job ="{ \"job_type\": \"population_analysis\", \"molecule\": \"/home/zhouoh/scratch/opt_89cd19c8814ffbbfeec61ed3611bcded.xyz\", \"parameters\": { \"method\": \"B3LYP\", \"basis\": \"def2-SVP\", \"charge\": 0, \"multiplicity\": 1, \"population_analysis_method\": \"Hirshfeld\", \"population_properties\": [\"charges\"] } }"
    import json
    job = json.loads(job)
    result = PropAnalysisHandler(job, logger)
    print(result)
