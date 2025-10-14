from pyscf import gto
from skala.pyscf import SkalaKS

mol = gto.M(
    atom="""H 0 0 0; H 0 0 1.4""",
    basis="def2-tzvp",
)
ks = SkalaKS(mol, xc="skala")
ks.kernel()
print(ks.dump_scf_summary())
