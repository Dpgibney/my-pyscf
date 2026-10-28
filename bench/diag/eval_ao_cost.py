"""How much of eval_ao is zero-filling of screened blocks vs computing AO values?
Times eval_ao (deriv=1) on one block_loop-sized block with the real screen index, with every shell
screened (pure zero fill) and with no screening (everything computed)."""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE

mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids.build(with_non0tab=True)
ni = mf._numint
nblk = 1200
b0 = (mf.grids.weights.size // BLKSIZE) // 2 - nblk // 2
ip0, ip1 = b0 * BLKSIZE, (b0 + nblk) * BLKSIZE
coords = mf.grids.coords[ip0:ip1]
mask = np.ascontiguousarray(mf.grids.non0tab[b0:b0 + nblk])
buf = numint._empty_aligned(4 * (ip1 - ip0) * mol.nao)
for deriv in (0, 1):
    for label, m in (('real mask', mask), ('all screened', np.zeros_like(mask)),
                     ('none screened', np.full_like(mask, 100))):
        ni.eval_ao(mol, coords, deriv=deriv, non0tab=m, cutoff=mf.grids.cutoff, out=buf)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            ni.eval_ao(mol, coords, deriv=deriv, non0tab=m, cutoff=mf.grids.cutoff, out=buf)
            ts.append(time.perf_counter() - t0)
        nz = np.count_nonzero(m) / m.size
        gb = (1 + 3 * deriv) * (ip1 - ip0) * mol.nao * 8 / 1e9
        print(f'deriv={deriv} {label:14s} (nonzero {nz:.2f}): {min(ts)*1e3:7.1f} ms '
              f'({gb / min(ts):.0f} GB/s of AO array written)')
