"""NumInt's fused XC kernels (lib/dft/nr_numint_fused.c, numint._FusedXC) against the
block_loop path (numint.FUSED_XC = False). Every result must be the same bit for bit.
  check  nr_rks and nr_uks (LDA, PBE, r2SCAN; MO-tagged dms by either density path, plain
         dms, three non-symmetric dms), nr_rks_fxc (hermi 0 and 1) and nr_uks_fxc,
         cache_xc_kernel and cache_xc_kernel1 (RKS, UKS). Each in four variants: fused with
         the radial cache, without it, with one set per VXCfused_dot_wv call, and the
         reference. Counts how many calls took the fused path.
  time   s per call, block_loop vs fused (with the radial cache, built before), interleaved
Usage: fused_numint_check.py [check|time] [system ...]
  systems: caffeine c20h42 (def2-SVP), caffeine-tzvp c20h42-tzvp
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

SYSTEMS = {'caffeine': (CAFFEINE, 'def2-svp'), 'c20h42': (n_alkane(20), 'def2-svp'),
           'caffeine-tzvp': (CAFFEINE, 'def2-tzvp'), 'c20h42-tzvp': (n_alkane(20), 'def2-tzvp')}
XCS = ('LDA,VWN', 'PBE', 'R2SCAN')
MAX_MEMORY = 20000

stock_fused_xc = numint._fused_xc
stock_dot = numint._FusedXC.dot
stock_rho1_preferred = numint._rho1_preferred
taken = [0, 0]  # calls of _fused_xc that returned a _FusedXC, all calls


def counting_fused_xc(*args, **kwargs):
    fused = stock_fused_xc(*args, **kwargs)
    taken[0] += fused is not None
    taken[1] += 1
    return fused


def one_set_dot(self, *args, **kwargs):
    max_memory, self.max_memory = self.max_memory, 0
    try:
        return stock_dot(self, *args, **kwargs)
    finally:
        self.max_memory = max_memory


def setup(name):
    atom, basis = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=MAX_MEMORY)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.build(with_non0tab=True)
    mf.conv_tol = 1e-6
    mf.kernel()
    nocc = mol.nelectron // 2
    occa = np.zeros(mol.nao)
    occb = np.zeros(mol.nao)
    occa[:nocc] = occb[:nocc-1] = 1  # a cation-like UKS density from the RKS orbitals
    mo = np.asarray((mf.mo_coeff, mf.mo_coeff))
    dmu = np.einsum('xpi,xi,xqi->xpq', mo, (occa, occb), mo)
    dmu = lib.tag_array(dmu, mo_coeff=mo, mo_occ=np.asarray((occa, occb)))
    return mol, mf, dmu, mo, (occa, occb)


def xc_results(mol, grids, mf, dmu, mo, occ, rho_paths=False):
    ni = numint.NumInt()
    nao = mol.nao
    dm = mf.make_rdm1()
    rng = np.random.default_rng(3)
    dm1 = rng.standard_normal((3, nao, nao)) * 1e-2
    dm1s = dm1 + dm1.transpose(0, 2, 1)
    out = {}
    for xc in XCS:
        out[f'nr_rks {xc} MO dm'] = ni.nr_rks(mol, grids, xc, dm, max_memory=MAX_MEMORY)
        out[f'nr_uks {xc} MO dm'] = ni.nr_uks(mol, grids, xc, dmu, max_memory=MAX_MEMORY)
        if rho_paths:
            continue
        out[f'nr_rks {xc} plain dm'] = ni.nr_rks(mol, grids, xc, np.asarray(dm))
        out[f'nr_rks {xc} 3 dms hermi=0'] = ni.nr_rks(mol, grids, xc, dm1, hermi=0)
        out[f'nr_uks {xc} plain dm'] = ni.nr_uks(mol, grids, xc, np.asarray(dmu))
        fxc = ni.cache_xc_kernel1(mol, grids, xc, np.asarray(dm))[2]
        out[f'nr_rks_fxc {xc} hermi=0'] = ni.nr_rks_fxc(mol, grids, xc, None, dm1, hermi=0, fxc=fxc)
        out[f'nr_rks_fxc {xc} hermi=1'] = ni.nr_rks_fxc(mol, grids, xc, None, dm1s, hermi=1, fxc=fxc)
        fxcu = ni.cache_xc_kernel1(mol, grids, xc, np.asarray(dmu), spin=1)[2]
        out[f'nr_uks_fxc {xc}'] = ni.nr_uks_fxc(mol, grids, xc, None, (dm1, dm1 * .5), fxc=fxcu)
        out[f'cache_xc_kernel {xc} RKS'] = ni.cache_xc_kernel(mol, grids, xc, mf.mo_coeff, mf.mo_occ)
        out[f'cache_xc_kernel {xc} UKS'] = ni.cache_xc_kernel(mol, grids, xc, mo, occ, spin=1)
        out[f'cache_xc_kernel1 {xc} RKS'] = ni.cache_xc_kernel1(mol, grids, xc, np.asarray(dm))
        out[f'cache_xc_kernel1 {xc} UKS'] = ni.cache_xc_kernel1(mol, grids, xc, np.asarray(dmu), spin=1)
    return out


def flatten(x):
    if isinstance(x, (tuple, list)):
        return [a for y in x for a in flatten(y)]
    return [np.asarray(x)]


def compare(ref, out):
    bad = []
    for key, r in ref.items():
        a, b = flatten(r), flatten(out[key])
        if not all(np.array_equal(x, y) for x, y in zip(a, b)):
            bad.append(f'{key} (max|diff| {max(abs(x - y).max() for x, y in zip(a, b)):.1e})')
    return bad


def check(name):
    mol, mf, dmu, mo, occ = setup(name)
    grids = mf.grids
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}', flush=True)
    numint._fused_xc = counting_fused_xc
    try:
        numint.FUSED_XC = False
        ref = xc_results(mol, grids, mf, dmu, mo, occ)
        numint.FUSED_XC = True
        variants = {'fused + radial cache': {},
                    'fused, no cache': {'RADIAL_CACHE_FRACTION': 0},
                    'fused, one set per call': {}}
        for label, env in variants.items():
            grids._fused_ao_data = None
            taken[:] = [0, 0]
            numint._FusedXC.dot = one_set_dot if 'one set' in label else stock_dot
            with lib.temporary_env(numint, **env):
                out = xc_results(mol, grids, mf, dmu, mo, occ)
            data = grids._fused_ao_data
            cache = 'none' if data is None or data.radial is None else \
                f'deriv {data.deriv}, {data.radial.nbytes / 1e9:.2f} GB'
            bad = compare(ref, out)
            print(f'  {label:25s}: {len(ref) - len(bad)}/{len(ref)} identical, fused in '
                  f'{taken[0]}/{taken[1]} calls, cache {cache}' + ''.join(
                      f'\n    DIFFERENT: {b}' for b in bad), flush=True)
        numint._FusedXC.dot = stock_dot
        for forced in (True, False):  # each density path of make_rho for MO-tagged dms
            numint._rho1_preferred = lambda ratio, xctype, f=forced: f
            numint.FUSED_XC = False
            ref = xc_results(mol, grids, mf, dmu, mo, occ, rho_paths=True)
            numint.FUSED_XC = True
            out = xc_results(mol, grids, mf, dmu, mo, occ, rho_paths=True)
            bad = compare(ref, out)
            path = 'dm (eval_rho1)' if forced else 'MO (eval_rho2)'
            print(f'  MO-tagged dms by the {path} path: {len(ref) - len(bad)}/{len(ref)} '
                  'identical' + ''.join(f'\n    DIFFERENT: {b}' for b in bad), flush=True)
    finally:
        numint._fused_xc = stock_fused_xc
        numint._rho1_preferred = stock_rho1_preferred
        numint.FUSED_XC = True


def timing(name, reps=5):
    mol, mf, dmu, mo, occ = setup(name)
    grids = mf.grids
    ni = numint.NumInt()
    dm = mf.make_rdm1()
    nao = mol.nao
    rng = np.random.default_rng(3)
    dm1 = rng.standard_normal((5, nao, nao)) * 1e-2
    print(f'{name}: nao={mol.nao} ngrids={grids.weights.size}', flush=True)
    for xc in ('PBE', 'R2SCAN'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc, np.asarray(dm), max_memory=MAX_MEMORY)[2]
        calls = {'nr_rks': lambda: ni.nr_rks(mol, grids, xc, dm, max_memory=MAX_MEMORY),
                 'nr_uks': lambda: ni.nr_uks(mol, grids, xc, dmu, max_memory=MAX_MEMORY),
                 'nr_rks_fxc 5 dms': lambda: ni.nr_rks_fxc(mol, grids, xc, None, dm1, fxc=fxc,
                                                           max_memory=MAX_MEMORY)}
        for label, f in calls.items():
            t = {False: [], True: []}
            for fused in t:
                numint.FUSED_XC = fused
                f()
            for _ in range(reps):
                for fused in t:
                    numint.FUSED_XC = fused
                    t0 = time.perf_counter()
                    f()
                    t[fused].append(time.perf_counter() - t0)
            a, b = np.median(t[False]), np.median(t[True])
            print(f'  {xc:7s} {label:17s}: block_loop {a:.3f} s, fused {b:.3f} s ({a / b:.2f}x)',
                  flush=True)
    numint.FUSED_XC = True
    data = grids._fused_ao_data
    print(f'  radial cache: deriv {data.deriv}, {data.radial.nbytes / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[2:] or ('caffeine', 'c20h42'):
        (check if what == 'check' else timing)(name)
