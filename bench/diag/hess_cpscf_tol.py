"""benzene+/def2-SVP UKS B3LYP Hessian on fixed orbitals at conv_tol_cpscf 1e-8 and 1e-12: `save` in one tree
(which runs the SCF), `cmp` in another (or the same, for the run-to-run spread) on the saved orbitals.
Usage: hess_cpscf_tol.py save|cmp OUT   (run under bench/diag/run_in.py)
"""
import sys
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE
from pyscf import gto, dft
mode, out = sys.argv[1], sys.argv[2]
mol = gto.M(atom=BENZENE, basis='def2-svp', charge=1, spin=1, verbose=0)
mf = dft.UKS(mol, xc='B3LYP')
if mode == 'save':
    mf.conv_tol = 1e-11
    mf.kernel()
    res = {'mo': mf.mo_coeff, 'occ': mf.mo_occ, 'e': mf.mo_energy, 'etot': mf.e_tot}
else:
    ref = np.load(out)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot = ref['mo'], ref['occ'], ref['e'], float(ref['etot'])
    mf.converged = True
    res = {}
for tol in (1e-8, 1e-12):
    mf.conv_tol_cpscf = tol
    h = mf.Hessian().kernel()
    res[f'h{tol:.0e}'] = h
    if mode == 'cmp':
        r = ref[f'h{tol:.0e}']
        print(f'conv_tol_cpscf {tol:.0e}: max|H - H_ref| {abs(h - r).max():.1e} (max|H| {abs(r).max():.2f})', flush=True)
if mode == 'save':
    np.savez(out, **res)
