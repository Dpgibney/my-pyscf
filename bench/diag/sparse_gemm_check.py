"""Check the gather/BLAS sparse kernels against dense products on real AO data, and time them
against upstream's sparse kernels (still in libdft under their old names).
Usage: sparse_gemm_check.py [c20|c20tz|caffeine]
"""
import ctypes
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE, NBINS

libdft = numint.libdft
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
which = sys.argv[1] if len(sys.argv) > 1 else 'c20'
atom, basis = {'c20': (n_alkane(20), 'def2-svp'), 'c20tz': (n_alkane(20), 'def2-tzvp'),
               'caffeine': (CAFFEINE, 'def2-svp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids.build(with_non0tab=True)
ni = mf._numint
nao, nbas = mol.nao, mol.nbas
ao_loc = mol.ao_loc_nr()
# one block of the size block_loop uses, from the middle of the grid
nblk = 1200
b0 = (mf.grids.weights.size // BLKSIZE) // 2 - nblk // 2
ip0, ip1 = b0 * BLKSIZE, (b0 + nblk) * BLKSIZE
mask = np.ascontiguousarray(mf.grids.non0tab[b0:b0 + nblk])
ao = ni.eval_ao(mol, mf.grids.coords[ip0:ip1], deriv=1, non0tab=mask, cutoff=mf.grids.cutoff)
ngrids = ip1 - ip0
rng = np.random.default_rng(1)
wv = rng.standard_normal((4, ngrids)) * mf.grids.weights[ip0:ip1]
dm = rng.standard_normal((nao, nao)); dm = dm + dm.T
pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
nbins = NBINS * 2 - int(NBINS * np.log(mf.grids.cutoff * 1e2) / np.log(mf.grids.cutoff))
print(f'{which}: nao={nao} ngrids={ngrids} nonzero screen fraction {np.count_nonzero(mask)/mask.size:.3f}')

# screened (block, shell) entries are exactly zero in eval_ao's output
blk_of_ao = np.zeros((nblk, nao), dtype=bool)
for ish in range(nbas):
    blk_of_ao[:, ao_loc[ish]:ao_loc[ish + 1]] = mask[:, ish:ish + 1] == 0
screened = np.repeat(blk_of_ao, BLKSIZE, axis=0)[:ngrids]  # (ngrids, nao)
print('eval_ao leaves screened blocks exactly zero:', all(not ao[c][screened].any() for c in range(4)))

def rel(a, b):
    return abs(a - b).max() / abs(b).max()

# 1. GGA XC matrix: ao0^T (sum_c wv_c ao_c), with NaN where the shell is screened
aow = numint._scale_ao(ao[:4], wv)
ref = ao[0].T @ aow
aow_nan = np.asfortranarray(aow.copy()); aow_nan[screened] = np.nan
out = numint._dot_ao_ao_sparse(ao[0], aow_nan, None, nbins, mask, pair_mask, ao_loc, hermi=0)
print(f'ao^T aow (gemm), NaN in screened blocks: rel err {rel(out, ref):.1e}')
# 2. LDA-like symmetric term with signed weights (sign-split dsyrk path)
ao0_nan = np.asfortranarray(ao[0].copy())
ref = (ao[0].T * wv[0]) @ ao[0]
out = numint._dot_ao_ao_sparse(ao[0], ao[0], wv[0], nbins, mask, pair_mask, ao_loc, hermi=1)
print(f'ao^T wv ao (syrk, hermi=1): rel err {rel(out, ref):.1e}, symmetric {abs(out - out.T).max():.1e}')
out = numint._dot_ao_ao_sparse(ao[0], ao[1], wv[0], nbins, mask, pair_mask, ao_loc, hermi=0)
print(f'ao0^T wv ao1 (gemm with wv): rel err {rel(out, (ao[0].T * wv[0]) @ ao[1]):.1e}')
out = numint._dot_ao_ao_sparse(ao[1], ao[1], None, nbins, mask, pair_mask, ao_loc, hermi=1)
print(f'ao1^T ao1 (syrk, no wv): rel err {rel(out, ao[1].T @ ao[1]):.1e}')
# accumulation into out
out0 = rng.standard_normal((nao, nao)); out0 = out0 + out0.T
out = numint._dot_ao_ao_sparse(ao[0], ao[0], wv[0], nbins, mask, pair_mask, ao_loc, hermi=1, out=out0.copy())
print(f'accumulates into out: rel err {rel(out, out0 + (ao[0].T * wv[0]) @ ao[0]):.1e}')
# 3. ao.dm, NaN in screened blocks of ao, compared where the shell is unscreened
ao_nan = np.asfortranarray(ao[0].copy()); ao_nan[screened] = np.nan
c0 = numint._dot_ao_dm_sparse(ao_nan, dm, nbins, mask, pair_mask, ao_loc)
ref = ao[0] @ dm
print(f'ao.dm (gemm), NaN in screened blocks: rel err on unscreened entries '
      f'{abs(c0 - ref)[~screened].max() / abs(ref).max():.1e}')
rho = numint._contract_rho_sparse(ao[0], c0, mask, ao_loc)
print(f'rho from ao.dm: rel err {rel(rho, np.einsum("gi,gi->g", ao[0], ref)):.1e}')

# --- timings: upstream sparse kernels vs gather/BLAS kernels, same inputs
def old_ao_ao(bra, ket):
    out = np.zeros((nao, nao))
    libdft.VXCdot_ao_ao_sparse(P(out), P(bra), P(ket), ctypes.c_int(nao), ctypes.c_int(ngrids),
                               ctypes.c_int(nbas), ctypes.c_int(0), ctypes.c_int(nbins), P(mask),
                               P(pair_mask), P(ao_loc))
    return out

def old_ao_dm(a, d):
    out = np.empty((nao, ngrids))
    libdft.VXCdot_ao_dm_sparse(P(out), P(a), P(d), ctypes.c_int(nao), ctypes.c_int(ngrids),
                               ctypes.c_int(nbas), ctypes.c_int(nbins), P(mask), P(pair_mask),
                               P(ao_loc))
    return out

def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)

aow = np.asfortranarray(aow)
t_old = best(lambda: old_ao_ao(ao[0], aow))
t_new = best(lambda: numint._dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc))
t_dense = best(lambda: lib.dot(ao[0].T, aow))
print(f'ao^T aow: upstream sparse {t_old*1e3:.1f} ms, gather+dgemm {t_new*1e3:.1f} ms, '
      f'dense dgemm {t_dense*1e3:.1f} ms')
t_old = best(lambda: old_ao_dm(ao[0], dm))
t_new = best(lambda: numint._dot_ao_dm_sparse(ao[0], dm, nbins, mask, pair_mask, ao_loc))
t_dense = best(lambda: lib.dot(dm.T, ao[0].T).T)
print(f'ao.dm:    upstream sparse {t_old*1e3:.1f} ms, gather+dgemm {t_new*1e3:.1f} ms, '
      f'dense dgemm {t_dense*1e3:.1f} ms')
t_new = best(lambda: numint._dot_ao_ao_sparse(ao[0], ao[0], wv[0], nbins, mask, pair_mask, ao_loc, hermi=1))
print(f'ao^T wv ao (hermi): gather+dsyrk {t_new*1e3:.1f} ms')
