'''nr_rks_fxc time for 3*natm CPHF-type dms on c20h42/SVP in the tree under test'''
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft, lib
mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0, max_memory=20000)
for xc in sys.argv[1:]:
    mf = dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-6
    mf.kernel()
    mo, occ = mf.mo_coeff, mf.mo_occ
    mocc = mo[:, occ > 0]
    nmo, nocc = mo.shape[1], mocc.shape[1]
    ni, grids = mf._numint, mf.grids
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo, occ, 0)
    nset = 3 * mol.natm
    x = np.random.default_rng(1).standard_normal((nset, nmo, nocc)) * 1e-2
    dm1 = lib.einsum('pa,kai,qi->kpq', mo, x, mocc)
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    f = lambda: ni.nr_rks_fxc(mol, grids, mf.xc, None, dm1, 0, 1, rho0, vxc, fxc, max_memory=16000)
    v = f()
    t0 = time.perf_counter(); f(); t = time.perf_counter() - t0
    print(f'{xc:8s} nset={nset} threads={lib.num_threads()}: nr_rks_fxc {t:6.2f} s, '
          f'|v|={np.linalg.norm(v):.12e}', flush=True)
