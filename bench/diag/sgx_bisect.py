"""Which sparse helper changes the SGX gradient's translational invariance? Swap upstream kernels back."""
import ctypes
import sys

import numpy
from pyscf import gto, scf, lib
from pyscf.dft import numint
from pyscf.sgx.sgx import sgx_fit
import pyscf.sgx.test.test_dm_screening as t

P = lambda a: a.ctypes.data_as(ctypes.c_void_p)


def old_dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc):
    ngrids, nao = ao.shape
    if screen_index is None or pair_mask is None or ngrids % numint.ALIGNMENT_UNIT != 0:
        return lib.dot(dm.T, ao.T).T
    dm = numpy.asarray(dm, order='C')
    out = numint._empty_aligned((nao, ngrids)).T
    numint.libdft.VXCdot_ao_dm_sparse(
        P(out), P(ao), P(dm), ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(screen_index.shape[1]), ctypes.c_int(nbins), P(screen_index), P(pair_mask),
        P(ao_loc))
    return out


def old_dot_ao_ao_sparse(ao1, ao2, wv, nbins, screen_index, pair_mask, ao_loc, hermi=0, out=None):
    ngrids, nao = ao1.shape
    if screen_index is None or pair_mask is None or ngrids % numint.ALIGNMENT_UNIT != 0:
        return numint._dot_ao_ao_dense(ao1, ao2, wv, out)
    if out is None:
        out = numpy.zeros((nao, nao))
    nbas = screen_index.shape[1]
    if wv is None:
        numint.libdft.VXCdot_ao_ao_sparse(
            P(out), P(ao1), P(ao2), ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
            ctypes.c_int(hermi), ctypes.c_int(nbins), P(screen_index), P(pair_mask), P(ao_loc))
    else:
        numint.libdft.VXCdot_aow_ao_sparse(
            P(out), P(ao1), P(ao2), P(wv), ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(hermi), ctypes.c_int(nbins), P(screen_index),
            P(pair_mask), P(ao_loc))
    return out


what = sys.argv[1]
if what in ('dm', 'both'):
    numint._dot_ao_dm_sparse = old_dot_ao_dm_sparse
if what in ('ao', 'both'):
    numint._dot_ao_ao_sparse = old_dot_ao_ao_sparse
mol = gto.M(atom=t.ATOM, basis='6-31g', verbose=0)
for fit_ovlp, optk in ((False, False), (True, True)):
    mf = sgx_fit(scf.RHF(mol))
    t._set_df_args(mf, fit_ovlp, optk)
    mf.kernel()
    g = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True).kernel()
    print(f'old kernels for {what:5s} fit_ovlp={fit_ovlp!s:5s}: e={mf.e_tot:.12f} '
          f'|sum g|={numpy.abs(g.sum(axis=0)).sum():.2e} g[2,2]={g[2,2]:.12f}')
