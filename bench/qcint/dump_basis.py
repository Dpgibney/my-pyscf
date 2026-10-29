"""Write a molecule's libcint arrays (atm, bas, env) and PySCF's Schwarz bounds for the direct SCF
(q_cond of init_direct_scf) to a binary file for qbench.c.
Layout: int32 natm, nbas, nenv; int32 atm[natm*6]; int32 bas[nbas*8]; float64 env[nenv];
float64 q_cond[nbas*nbas].
Usage: dump_basis.py SYSTEM OUT.bin   (SYSTEM: water, benzene, caffeine, caffeine-tz, c20, c20-tz,
benzene-ccpvdz with generally contracted shells)"""
import sys

sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, WATER, n_alkane
from pyscf import gto, scf

which, out = sys.argv[1], sys.argv[2]
atom, basis = {'water': (WATER, 'def2-svp'), 'benzene': (BENZENE, 'def2-svp'),
               'caffeine': (CAFFEINE, 'def2-svp'), 'caffeine-tz': (CAFFEINE, 'def2-tzvp'),
               'c20': (n_alkane(20), 'def2-svp'), 'c20-tz': (n_alkane(20), 'def2-tzvp'),
               'benzene-ccpvdz': (BENZENE, 'cc-pvdz')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
q_cond = scf.RHF(mol).init_direct_scf().q_cond
with open(out, 'wb') as f:
    np.array([mol.natm, mol.nbas, mol._env.size], dtype=np.int32).tofile(f)
    np.asarray(mol._atm, dtype=np.int32).tofile(f)
    np.asarray(mol._bas, dtype=np.int32).tofile(f)
    np.asarray(mol._env, dtype=np.float64).tofile(f)
    np.asarray(q_cond, dtype=np.float64).tofile(f)
print(f'{which}: natm {mol.natm} nbas {mol.nbas} nao {mol.nao} -> {out}')
