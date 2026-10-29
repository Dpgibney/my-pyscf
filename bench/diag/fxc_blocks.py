'''nr_rks_fxc (dm path) time per set on c20h42/SVP LDA vs nset and max_memory, which set the
number of grid blocks of the fused kernels (_rks_fxc_blksize before It13, _fused_fxc_blksize after)'''
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0, max_memory=20000)
mf = dft.RKS(mol, xc='LDA,VWN')
mf.conv_tol = 1e-6
mf.kernel()
mo, occ = mf.mo_coeff, mf.mo_occ
mocc = mo[:, occ > 0]
nmo, nocc = mo.shape[1], mocc.shape[1]
ni, grids = mf._numint, mf.grids
rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo, occ, 0)
ngrids = grids.weights.size
print(f'threads={lib.num_threads()} nao={mol.nao} nocc={nocc} ngrids={ngrids}', flush=True)
for nset in (186, 62, 18):
    x = np.random.default_rng(1).standard_normal((nset, nmo, nocc)) * 1e-2
    dm1 = lib.einsum('pa,kai,qi->kpq', mo, x, mocc)
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    for mem in (16000, 200000):
        blksize = getattr(numint, '_fused_fxc_blksize', None) or numint._rks_fxc_blksize
        blk = blksize('LDA', nset, mol.nao, ngrids, mem)
        f = lambda: ni.nr_rks_fxc(mol, grids, mf.xc, None, dm1, 0, 1, rho0, vxc, fxc, max_memory=mem)
        f()
        t = []
        for _ in range(2):
            t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
        t = min(t)
        print(f'nset={nset:3d} max_memory={mem:6d}: blksize={blk:6d} nblocks={-(-ngrids//blk):3d} '
              f'time {t:6.2f} s, per set {t/nset*1e3:6.1f} ms', flush=True)
