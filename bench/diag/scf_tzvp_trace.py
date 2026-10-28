"""Convergence trace of the tight n-C20H42/def2-TZVP PBE SCF (validation case scf_c20h42_tzvp_pbe): which criterion
stalls?  Usage: scf_tzvp_trace.py [MAX_CYCLE]"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft

mol = gto.M(atom=n_alkane(20), basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='PBE')
mf.chkfile = None
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-8
mf.max_cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 40
nov = mol.nelectron // 2 * (mol.nao - mol.nelectron // 2)
print(f'nao {mol.nao}, nocc*nvir {nov}', flush=True)


def callback(envs):
    g = envs['norm_gorb']
    print(f"cycle {envs['cycle']:3d}  E {envs['e_tot']:.12f}  dE {envs['e_tot'] - envs['last_hf_e']: .2e}  "
          f"|g| {g:.2e}  |g|/sqrt(n) {g / np.sqrt(nov):.2e}"
          f"  |ddm| {envs['norm_ddm']:.2e}", flush=True)


mf.callback = callback
mf.kernel()
print('converged', mf.converged, 'cycles', mf.cycles, 'e_tot', mf.e_tot, flush=True)
