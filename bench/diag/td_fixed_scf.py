"""TDA excitation energies from one fixed SCF solution, with the per-dm or batched direct J/K kernels,
repeated, to separate SCF run-to-run variation from the J/K path.
Usage: td_fixed_scf.py save SCF.npz | run SCF.npz MIN [REPS]   (MIN = _vhf.DIRECT_BATCH_MIN;
run under bench/diag/run_in.py)"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from cases import converged_ks, make_ks
from molecules import CAFFEINE
from pyscf import gto, lib
from pyscf.scf import _vhf

mode, path = sys.argv[1], sys.argv[2]
if mode == 'save':
    mf = converged_ks(CAFFEINE, 'def2-svp', 'B3LYP', tight=True)
    np.savez(path, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy, e_tot=mf.e_tot)
    print('saved', path, 'e_tot %.15f' % mf.e_tot)
else:
    s = np.load(path)
    mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
    mf = make_ks(mol, 'B3LYP', tight=True)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = s['mo_coeff'], s['mo_occ'], s['mo_energy']
    mf.e_tot, mf.converged = float(s['e_tot']), True
    minb = int(sys.argv[3])
    for rep in range(int(sys.argv[4]) if len(sys.argv) > 4 else 1):
        with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=minb):
            td = mf.TDA()
            td.nstates = 5
            td.conv_tol = 1e-6
            td.max_cycle = 200
            td.kernel()
        print(f'MIN={minb} rep={rep} e =', ' '.join('%.15f' % e for e in td.e), flush=True)
