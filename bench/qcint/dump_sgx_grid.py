"""Write the SGX grid that pyscf.sgx builds for a molecule (sgx_fit with pjs, its final grid level) for gbench.c.
Layout: int32 ngrids; float64 coords[ngrids*3].
Usage: dump_sgx_grid.py SYSTEM OUT.bin   (SYSTEM as in dump_basis.py: caffeine, caffeine-tz, c20, ...)"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np  # noqa: E402
from molecules import BENZENE, CAFFEINE, WATER, n_alkane  # noqa: E402
from pyscf import gto, dft, sgx  # noqa: E402

which, out = sys.argv[1], sys.argv[2]
atom, basis = {'water': (WATER, 'def2-svp'), 'benzene': (BENZENE, 'def2-svp'),
               'caffeine': (CAFFEINE, 'def2-svp'), 'caffeine-tz': (CAFFEINE, 'def2-tzvp'),
               'c20': (n_alkane(20), 'def2-svp'), 'c20-tz': (n_alkane(20), 'def2-tzvp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = sgx.sgx_fit(dft.RKS(mol, xc='B3LYP'), pjs=True)
mf.with_df.build(level=mf.with_df.grids_level_f)
coords = np.ascontiguousarray(mf.with_df.grids.coords)
with open(out, 'wb') as f:
    np.array([coords.shape[0]], dtype=np.int32).tofile(f)
    coords.astype(np.float64).tofile(f)
print(f'{which}: {coords.shape[0]} SGX grid points -> {out}')
