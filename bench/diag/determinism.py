"""Repeat NumInt.nr_rks on the alkane with one density and compare the results bitwise.
Usage: determinism.py [--unfused] [nrep]"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft
from pyscf.dft import numint

if '--unfused' in sys.argv:
    numint._ao_comps_contiguous = lambda ao: False   # commit A's code paths
nrep = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 5
mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids.build(with_non0tab=True)
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
ni = mf._numint
ref = None
for k in range(nrep):
    n, e, v = ni.nr_rks(mol, mf.grids, 'B3LYP', dm)
    if ref is None:
        ref = (e, v)
        continue
    print(f'rep {k}: |dExc|={abs(e - ref[0]):.1e} max|dV|={abs(v - ref[1]).max():.1e}', flush=True)
