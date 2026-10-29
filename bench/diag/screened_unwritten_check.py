"""eval_ao leaving the AO values of screened shells unwritten (zero_screened=False) on the
blocks that nr_rks, nr_uks, nr_rks_fxc and nr_uks_fxc read with the sparse kernels.
1. eval_ao itself: unscreened values equal zero_screened=True's, screened ones are untouched.
2. Every XC path, three ways: zeros written (as before), left unwritten, and NaN written into
   every screened value (a reader of screened values would turn the result into NaN). The last
   two must equal the first bit for bit (same kernels, same thread count).
3. Timings, zeros vs unwritten, interleaved: nr_rks per call (time) or whole SCF runs (scf).
Usage: screened_unwritten_check.py [check|time|scf] [system ...]   systems: c20h42 caffeine
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE

SYSTEMS = {'c20h42': n_alkane(20), 'caffeine': CAFFEINE}
# These paths through block_loop and eval_ao, not the kernels that compute AO values in
# place (It11)
numint.FUSED_XC = False
stock_eval_ao = numint.NumInt.__dict__['eval_ao']
stock_reader_kw = numint._sparse_reader_kw
nan_written = [0]   # values set to NaN by nan_eval_ao


def nan_eval_ao(mol, coords, deriv=0, shls_slice=None, non0tab=None, cutoff=None, out=None,
                verbose=None, zero_screened=True):
    ao = numint.eval_ao(mol, coords, deriv, shls_slice, non0tab, cutoff, out, verbose,
                        zero_screened)
    if not zero_screened:
        ao_loc = mol.ao_loc_nr()
        shl = np.repeat(np.arange(mol.nbas), np.diff(ao_loc))
        ng = coords.shape[0]
        screened = non0tab[:(ng + BLKSIZE - 1) // BLKSIZE][:, shl] == 0
        screened = np.repeat(screened, BLKSIZE, axis=0)[:ng]
        if ao.ndim == 2:
            ao[screened] = np.nan
        else:
            ao[:, screened] = np.nan
        nan_written[0] += np.count_nonzero(screened) * (1 if ao.ndim == 2 else ao.shape[0])
    return ao


def mode(name):
    numint.NumInt.eval_ao = staticmethod(nan_eval_ao) if name == 'nan' else stock_eval_ao
    numint._sparse_reader_kw = (lambda ni, *dms: {}) if name == 'zeros' else stock_reader_kw


def setup(name):
    mol = gto.M(atom=SYSTEMS[name], basis='def2-svp', verbose=0)
    grids = dft.gen_grid.Grids(mol).build(with_non0tab=True)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids = grids
    mf.conv_tol = 1e-6
    mf.kernel()
    return mol, grids, mf


def check_eval_ao(mol, grids):
    ni = numint.NumInt()
    ip1 = 1200 * BLKSIZE
    coords, mask = grids.coords[:ip1], grids.non0tab
    for deriv in (0, 1, 2):
        ref = ni.eval_ao(mol, coords, deriv=deriv, non0tab=mask)
        out = np.full(ref.size, np.nan)
        ao = ni.eval_ao(mol, coords, deriv=deriv, non0tab=mask, out=out, zero_screened=False)
        written = ~np.isnan(ao)
        # unwritten = exactly the zeros of the screened shells
        assert np.array_equal(ao[written], ref[written])
        assert not np.any(ref[~written]), 'a value left unwritten is nonzero'
        shl = np.repeat(np.arange(mol.nbas), np.diff(mol.ao_loc_nr()))
        expect = np.repeat(mask[:1200][:, shl] == 0, BLKSIZE, axis=0)[:ip1]
        assert np.array_equal(~written if ao.ndim == 2 else ~written[0], expect)
        print(f'  eval_ao deriv={deriv}: {np.mean(~written):.1%} of values left unwritten, '
              f'the rest identical', flush=True)


def xc_paths(mol, grids, mf):
    ni = numint.NumInt()
    nao = mol.nao
    dm = mf.make_rdm1()           # MO-tagged, as in SCF
    dm_plain = np.asarray(dm)
    rng = np.random.default_rng(1)
    dm1 = rng.standard_normal((3, nao, nao)) * .01
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    dma = lib.tag_array(dm_plain * .5, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ * .5)
    paths = {}
    for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN'):
        paths[f'nr_rks {xc}'] = lambda xc=xc: ni.nr_rks(mol, grids, xc, dm)
        paths[f'nr_rks {xc} plain dm'] = lambda xc=xc: ni.nr_rks(mol, grids, xc, dm_plain)
        paths[f'nr_uks {xc}'] = lambda xc=xc: ni.nr_uks(mol, grids, xc, (dma, dma * 1.01))
    for xc in ('LDA,VWN', 'PBE', 'r2SCAN'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc, dm_plain)[2]
        paths[f'nr_rks_fxc {xc}'] = lambda xc=xc, fxc=fxc: ni.nr_rks_fxc(
            mol, grids, xc, dm_plain, dm1, fxc=fxc)
        fxcu = ni.cache_xc_kernel1(mol, grids, xc, (dm_plain * .5, dm_plain * .5), spin=1)[2]
        paths[f'nr_uks_fxc {xc}'] = lambda xc=xc, fxc=fxcu: ni.nr_uks_fxc(
            mol, grids, xc, (dm_plain * .5, dm_plain * .5), (dm1, dm1 * .5), fxc=fxc)
    return paths


def check(name):
    mol, grids, mf = setup(name)
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}, screen index nonzero '
          f'{np.count_nonzero(grids.non0tab) / grids.non0tab.size:.1%}', flush=True)
    check_eval_ao(mol, grids)
    for label, f in xc_paths(mol, grids, mf).items():
        res = {}
        nan_written[0] = 0
        for m in ('zeros', 'unwritten', 'nan'):
            mode(m)
            r = f()
            res[m] = [np.asarray(x) for x in (r if isinstance(r, tuple) else (r,))]
        mode('unwritten')
        same = [all(np.array_equal(a, b) for a, b in zip(res['zeros'], res[m]))
                for m in ('unwritten', 'nan')]
        finite = all(np.isfinite(a).all() for a in res['nan'])
        diff = max(abs(a - b).max() for a, b in zip(res['zeros'], res['nan']))
        print(f'  {label:28s} unwritten == zeros: {same[0]}, NaN-filled == zeros: {same[1]}'
              f' (finite {finite}, max|diff| {diff:.1e}, {nan_written[0]:.2e} NaN written)',
              flush=True)


def timing(name, reps=5):
    mol, grids, mf = setup(name)
    ni = numint.NumInt()
    dm = mf.make_rdm1()
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}', flush=True)
    for xc in ('PBE', 'B3LYP', 'r2SCAN'):
        t = {'zeros': [], 'unwritten': []}
        for m in t:
            mode(m)
            ni.nr_rks(mol, grids, xc, dm)
        for _ in range(reps):
            for m in t:
                mode(m)
                t0 = time.perf_counter()
                ni.nr_rks(mol, grids, xc, dm)
                t[m].append(time.perf_counter() - t0)
        a, b = np.median(t['zeros']), np.median(t['unwritten'])
        print(f'  nr_rks {xc:7s}: zeros {a:.3f} s, unwritten {b:.3f} s ({a / b:.2f}x)', flush=True)
    mode('unwritten')


def scf(name, reps=2):
    mol = gto.M(atom=SYSTEMS[name], basis='def2-svp', verbose=0)
    print(f'{name}: nao={mol.nao}', flush=True)
    for xc in ('B3LYP', 'r2SCAN'):
        t = {'zeros': [], 'unwritten': []}
        e = {}
        for _ in range(reps):
            for m in t:
                mode(m)
                mf = dft.RKS(mol, xc=xc)
                t0 = time.perf_counter()
                e[m] = mf.kernel()
                t[m].append(time.perf_counter() - t0)
        a, b = np.median(t['zeros']), np.median(t['unwritten'])
        print(f'  SCF {xc:7s}: zeros {a:.1f} s, unwritten {b:.1f} s ({a / b:.2f}x), '
              f'|dE| {abs(e["zeros"] - e["unwritten"]):.1e}', flush=True)
    mode('unwritten')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[2:] or SYSTEMS:
        {'check': check, 'time': timing, 'scf': scf}[what](name)
