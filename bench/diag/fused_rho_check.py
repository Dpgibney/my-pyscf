"""Check VXCeval_rho_sparse_gemm against eval_rho1 (upstream path built from the sparse kernels)."""
import ctypes
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE

libdft = numint.libdft


def _eval_rho_sparse(ao, dm, screen_index, ao_loc, xctype):
    if xctype in ('LDA', 'HF'):
        nderiv, ao0 = 0, ao
    elif xctype in ('GGA', 'NLC'):
        nderiv, ao0 = 1, ao[0]
    else:
        nderiv, ao0 = 2, ao[0]
    ngrids, nao = ao0.shape
    rho = np.empty(((1, 4, 5)[nderiv], ngrids))
    dm = np.asarray(dm, order='C')
    libdft.VXCeval_rho_sparse_gemm(
        rho.ctypes.data_as(ctypes.c_void_p), ao.ctypes.data_as(ctypes.c_void_p),
        dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nderiv), ctypes.c_int(nao),
        ctypes.c_int(ngrids), ctypes.c_int(screen_index.shape[1]),
        screen_index.ctypes.data_as(ctypes.c_void_p), ao_loc.ctypes.data_as(ctypes.c_void_p))
    return rho[0] if nderiv == 0 else rho


mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids.build(with_non0tab=True)
ni = mf._numint
ao_loc = mol.ao_loc_nr()
nblk = 1200
b0 = (mf.grids.weights.size // BLKSIZE) // 2 - nblk // 2
ip0, ip1 = b0 * BLKSIZE, (b0 + nblk) * BLKSIZE
mask = np.ascontiguousarray(mf.grids.non0tab[b0:b0 + nblk])
ao = ni.eval_ao(mol, mf.grids.coords[ip0:ip1], deriv=1, non0tab=mask, cutoff=mf.grids.cutoff)
dm = mf.get_init_guess(key='minao')
for xctype in ('LDA', 'GGA', 'MGGA'):
    a = ao[0] if xctype == 'LDA' else ao
    ref = numint.eval_rho1(mol, a, dm, mask, xctype, hermi=1, with_lapl=False)
    new = _eval_rho_sparse(a, dm, mask, ao_loc, xctype)
    dense = numint.eval_rho(mol, a, dm, None, xctype, hermi=1, with_lapl=False)
    err = abs(new - ref).max() / abs(ref).max()
    errd = abs(new - dense).max() / abs(dense).max()
    t0 = time.perf_counter()
    for _ in range(3):
        numint.eval_rho1(mol, a, dm, mask, xctype, hermi=1, with_lapl=False)
    t1 = time.perf_counter()
    for _ in range(3):
        _eval_rho_sparse(a, dm, mask, ao_loc, xctype)
    t2 = time.perf_counter()
    print(f'{xctype:5s} rel err vs eval_rho1 {err:.1e}, vs dense eval_rho {errd:.1e}; '
          f'eval_rho1 {(t1-t0)/3*1e3:.1f} ms, fused {(t2-t1)/3*1e3:.1f} ms (timings disturbed if busy)')
