"""Prototype: NumInt.nr_rks (GGA, meta-GGA) with the AO values computed inside the sparse XC
kernels (fused_xc.c) instead of written to and read back from block_loop's AO blocks.
nr_rks_fused follows nr_rks step by step (same blocks, same LibXC calls, same sums), so with a
plain density matrix (eval_rho1's sparse path) its results must equal nr_rks bit for bit.
  check  bitwise comparison with nr_rks, without and with the radial cache
  time   s per nr_rks call, interleaved: nr_rks with a plain and an MO-tagged dm (as in SCF),
         fused, fused with the radial cache (built once, its size and build time reported)
Usage: fused_xc.py [check|time] [case ...]   cases: caffeine-svp caffeine-tzvp c20h42-svp c20h42-tzvp
Build fused_xc.so first (see fused_xc.c).
"""
import ctypes
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.gto.mole import NCTR_OF

_lib = np.ctypeslib.load_library('fused_xc', os.path.dirname(os.path.abspath(__file__)))
CASES = {'caffeine-svp': (CAFFEINE, 'def2-svp'), 'caffeine-tzvp': (CAFFEINE, 'def2-tzvp'),
         'c20h42-svp': (n_alkane(20), 'def2-svp'), 'c20h42-tzvp': (n_alkane(20), 'def2-tzvp')}
MAX_MEMORY = 20000  # MB, as scripts/pyscf-env.sh: block_loop blocks of 1200*56 points


def ptr(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


class RadialCache:
    '''Contracted radial factors of every (56-point block, shell) the screen index keeps.'''
    def __init__(self, mol, grids, deriv=1):
        mask = grids.non0tab
        sz = ((mask != 0) * ((deriv + 1) * mol._bas[:, NCTR_OF] * BLKSIZE)).ravel().astype(np.int64)
        self.offs = (np.cumsum(sz) - sz).reshape(mask.shape)
        self.data = np.empty(int(sz.sum()))
        coords = np.asarray(grids.coords.T, order='C')
        t0 = time.perf_counter()
        _lib.FUSEDradial_fill(ptr(self.data), ptr(self.offs), ctypes.c_int(deriv),
                              ctypes.c_int(grids.weights.size), ctypes.c_int(mol.nbas),
                              ptr(mask), ptr(coords), ptr(mol._atm), ptr(mol._bas), ptr(mol._env))
        self.t_fill = time.perf_counter() - t0


def nr_rks_fused(ni, mol, grids, xc_code, dm, radial=None, max_memory=MAX_MEMORY):
    xctype = ni._xc_type(xc_code)
    assert xctype in ('GGA', 'MGGA')
    nderiv, nrho = (1, 4) if xctype == 'GGA' else (2, 5)
    dm = np.asarray(dm, order='C')
    nao, nbas, ngrids = mol.nao, mol.nbas, grids.weights.size
    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int32)
    coords = np.asarray(grids.coords.T, order='C')
    mask = grids.non0tab
    mol_args = (ptr(mol._atm), ptr(mol._bas), ptr(mol._env))
    blksize = int(max_memory * 1e6 / (5 * nao * 8 * BLKSIZE))    # block_loop, deriv=1
    blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    nelec = np.zeros(1)
    excsum = np.zeros(1)
    vmat = np.zeros((nao, nao))
    v1 = np.zeros((nao, nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ng, b0 = ip1 - ip0, ip0 // BLKSIZE
        args = (ctypes.c_int(nao), ctypes.c_int(ng), ctypes.c_int(nbas), ptr(mask[b0:]),
                ptr(ao_loc), ctypes.c_void_p(coords.ctypes.data + ip0 * 8),
                ctypes.c_int(ngrids), *mol_args,
                None if radial is None else ptr(radial.data),
                None if radial is None else ptr(radial.offs[b0:]))
        rho = np.empty((nrho, ng))
        _lib.FUSEDeval_rho(ptr(rho), ptr(dm), ctypes.c_int(nderiv), *args)
        weight = grids.weights[ip0:ip1]
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
        den = rho[0] * weight
        nelec[0] += den.sum()
        excsum[0] += np.dot(den, exc)
        wv = weight * vxc
        wv[0] *= .5
        w4 = np.asarray(wv[:4], order='C')
        _lib.FUSEDdot_ao_aow(ptr(vmat), ptr(w4), ctypes.c_int(4), *args)
        if xctype == 'MGGA':
            wv[4] *= .5
            _lib.FUSEDdot_ao_wv_ao(ptr(v1), ptr(np.asarray(wv[4], order='C')), *args)
    vmat = lib.hermi_sum(vmat[None], axes=(0, 2, 1))[0]
    if xctype == 'MGGA':
        vmat += v1
    return nelec[0], excsum[0], vmat


def setup(name):
    atom, basis = CASES[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.build(with_non0tab=True)
    mf.conv_tol = 1e-6
    mf.kernel()
    return mol, mf.grids, mf.make_rdm1()


def check(name):
    mol, grids, dm = setup(name)
    ni = numint.NumInt()
    radial = RadialCache(mol, grids)
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}, radial cache '
          f'{radial.data.nbytes / 1e9:.2f} GB', flush=True)
    for xc in ('PBE', 'r2SCAN'):
        ref = ni.nr_rks(mol, grids, xc, np.asarray(dm), max_memory=MAX_MEMORY)
        for label, rc in (('fused', None), ('fused+radial', radial)):
            out = nr_rks_fused(ni, mol, grids, xc, dm, rc)
            same = all(np.array_equal(a, b) for a, b in zip(ref, out))
            diff = max(abs(np.asarray(a) - b).max() for a, b in zip(ref, out))
            print(f'  {xc:7s} {label:13s} == nr_rks: {same} (max|diff| {diff:.1e})', flush=True)


def timing(name, reps=5):
    mol, grids, dm = setup(name)
    ni = numint.NumInt()
    radial = RadialCache(mol, grids)
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}; radial cache '
          f'{radial.data.nbytes / 1e9:.2f} GB, built in {radial.t_fill:.3f} s', flush=True)
    dm_plain = np.asarray(dm)
    for xc in ('PBE', 'r2SCAN'):
        runs = {'nr_rks': lambda: ni.nr_rks(mol, grids, xc, dm_plain, max_memory=MAX_MEMORY),
                'nr_rks MO dm': lambda: ni.nr_rks(mol, grids, xc, dm, max_memory=MAX_MEMORY),
                'fused': lambda: nr_rks_fused(ni, mol, grids, xc, dm_plain),
                'fused+radial': lambda: nr_rks_fused(ni, mol, grids, xc, dm_plain, radial)}
        t = {k: [] for k in runs}
        for f in runs.values():
            f()
        for _ in range(reps):
            for k, f in runs.items():
                t0 = time.perf_counter()
                f()
                t[k].append(time.perf_counter() - t0)
        med = {k: np.median(v) for k, v in t.items()}
        print(f'  {xc:7s} s/call: ' + ', '.join(f'{k} {v:.3f}' for k, v in med.items()) +
              f' | fused {med["nr_rks"] / med["fused"]:.2f}x, fused+radial '
              f'{med["nr_rks"] / med["fused+radial"]:.2f}x vs nr_rks', flush=True)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[2:] or CASES:
        (check if what == 'check' else timing)(name)
