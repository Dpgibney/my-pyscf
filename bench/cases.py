"""Benchmark/validation cases.

A case is a function ``setup(tight) -> run``; only ``run()`` is timed. ``run()`` returns a dict of
JSON-serializable values used by validate.py. ``tight=True`` selects convergence settings for
correctness comparisons; timing runs use PySCF defaults. pyscf is imported inside the functions so the
worker picks up whichever tree PYTHONPATH selects.
"""
import numpy as np
from molecules import CAFFEINE, BENZENE, WATER, n_alkane

CASES = {}  # name -> (setup, suites)


def case(name, suites):
    def deco(fn):
        CASES[name] = (fn, suites)
        return fn
    return deco


def make_ks(mol, xc, tight, density_fit=False):
    from pyscf import dft
    mf = dft.RKS(mol, xc=xc) if mol.spin == 0 else dft.UKS(mol, xc=xc)
    mf.chkfile = None
    if tight:
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 100
    if density_fit:
        mf = mf.density_fit()
    return mf


def scf_case(atom, basis, xc, charge=0, spin=0, density_fit=False):
    def setup(tight):
        from pyscf import gto
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)

        def run():
            mf = make_ks(mol, xc, tight, density_fit)
            mf.kernel()
            return {'e_tot': mf.e_tot, 'cycles': mf.cycles, 'converged': bool(mf.converged)}
        return run
    return setup


def converged_ks(atom, basis, xc, tight, charge=0, spin=0, density_fit=False):
    from pyscf import gto
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = make_ks(mol, xc, tight, density_fit)
    mf.kernel()
    assert mf.converged
    return mf


def grad_case(atom, basis, xc):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight)

        def run():
            g = mf.nuc_grad_method().kernel()
            return {'e_tot': mf.e_tot, 'grad': g.tolist()}
        return run
    return setup


def td_case(atom, basis, xc, method, nstates=5):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight)

        def run():
            td = getattr(mf, method)()
            td.nstates = nstates
            if tight:
                td.conv_tol = 1e-8
                td.max_cycle = 200
            td.kernel()
            return {'e_tot': mf.e_tot, 'excitations': np.asarray(td.e).tolist(),
                    'converged': np.asarray(td.converged).tolist()}
        return run
    return setup


def hess_case(atom, basis, xc, charge=0, spin=0, density_fit=False):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight, charge, spin, density_fit)

        def run():
            h = mf.Hessian().kernel()
            return {'e_tot': mf.e_tot, 'hessian': h.tolist()}
        return run
    return setup


def grid_case(atom, basis):
    '''Script-style direct use: eval_ao -> eval_rho -> libxc.eval_xc over a whole grid in 50k chunks.'''
    def setup(tight):
        from pyscf import gto, dft
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        grids = dft.gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        dm = dft.RKS(mol).get_init_guess(key='minao')
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        def run():
            from pyscf import lib
            from pyscf.dft import numint, libxc
            out = {}
            for deriv, xctype, xc in [(0, 'LDA', 'LDA,VWN'), (1, 'GGA', 'PBE'),
                                      (1, 'MGGA', 'SCAN'), (2, 'GGA', 'PBE')]:
                exc_sum = 0.
                for i0, i1 in lib.prange(0, grids.weights.size, 50000):
                    ao = numint.eval_ao(mol, grids.coords[i0:i1], deriv=deriv)
                    if deriv == 2:
                        ao = ao[:4]
                    rho = numint.eval_rho(mol, ao, dm, xctype=xctype, with_lapl=False)
                    exc = libxc.eval_xc(xc, rho, deriv=1)[0]
                    rho0 = rho if xctype == 'LDA' else rho[0]
                    exc_sum += np.dot(exc * rho0, grids.weights[i0:i1])
                out[f'd{deriv}_{xc}'] = exc_sum
            return out
        return run
    return setup


def micro_gemm():
    '''lib.dot throughput on the shapes numint produces (k = grid block) plus square matrices.'''
    def setup(tight):
        def run():
            import time
            from pyscf import lib
            rng = np.random.default_rng(0)
            out = {}
            for m, k, n in [(246, 67200, 246), (67200, 246, 246), (494, 67200, 494),
                            (67200, 494, 494), (2000, 2000, 2000), (4000, 4000, 4000)]:
                a = rng.random((m, k))
                b = rng.random((k, n))
                lib.dot(a, b)
                t0 = time.perf_counter()
                for _ in range(3):
                    lib.dot(a, b)
                out[f'gflops_{m}x{k}x{n}'] = 6 * m * k * n / (time.perf_counter() - t0) / 1e9
            return out
        return run
    return setup


def micro_libxc():
    '''Raw libxc throughput (eval_xc_eff, 1M points) by functional and derivative order.'''
    def setup(tight):
        rng = np.random.default_rng(1)
        npts = 1_000_000
        rho = np.empty((5, npts))
        rho[0] = rng.random(npts) * 2 + 1e-4
        rho[1:4] = rng.standard_normal((3, npts)) * .3
        rho[4] = rng.random(npts) * 2 + 1e-3

        def run():
            import time
            from pyscf.dft import libxc
            out = {}
            for xc, nvar in [('LDA,VWN', 1), ('PBE', 4), ('B3LYP', 4), ('SCAN', 5), ('r2SCAN', 5)]:
                r = rho[0] if nvar == 1 else rho[:nvar]
                for d in (1, 2, 3):
                    try:
                        libxc.eval_xc_eff(xc, r[..., :1000].copy(), deriv=d)
                    except AssertionError:  # deriv beyond the functional's max_deriv_order
                        out[f'{xc}_d{d}_s'] = float('nan')
                        continue
                    t0 = time.perf_counter()
                    libxc.eval_xc_eff(xc, r, deriv=d)
                    out[f'{xc}_d{d}_s'] = time.perf_counter() - t0
            return out
        return run
    return setup


QUICK, FULL, MICRO = 'quick', 'full', 'micro'
C20H42 = n_alkane(20)

case('scf_caffeine_svp_pbe', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'PBE'))
case('scf_caffeine_svp_b3lyp', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'B3LYP'))
case('scf_caffeine_svp_r2scan', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'r2SCAN'))
case('grad_benzene_svp_b3lyp', (QUICK, FULL))(grad_case(BENZENE, 'def2-svp', 'B3LYP'))
case('tda_benzene_svp_b3lyp', (QUICK, FULL))(td_case(BENZENE, 'def2-svp', 'B3LYP', 'TDA'))
case('grid_caffeine_svp', (QUICK, FULL))(grid_case(CAFFEINE, 'def2-svp'))

case('scf_caffeine_tzvp_dfpbe', (FULL,))(scf_case(CAFFEINE, 'def2-tzvp', 'PBE', density_fit=True))
case('scf_caffeine+_svp_b3lyp_uks', (FULL,))(scf_case(CAFFEINE, 'def2-svp', 'B3LYP', charge=1, spin=1))
case('tddft_benzene_svp_b3lyp', (FULL,))(td_case(BENZENE, 'def2-svp', 'B3LYP', 'TDDFT'))
case('hess_benzene_svp_b3lyp', (FULL,))(hess_case(BENZENE, 'def2-svp', 'B3LYP'))
case('hess_benzene_svp_pbe', (FULL,))(hess_case(BENZENE, 'def2-svp', 'PBE'))
# Hessian code paths: LDA/MGGA, range-separated hybrids, UKS, density fitting ('hessian' suite)
HESS = 'hessian'
for _xc in ('LDA,VWN', 'r2SCAN', 'CAM-B3LYP'):
    case(f'hess_water_svp_{_xc}', (HESS,))(hess_case(WATER, 'def2-svp', _xc))
for _xc in ('LDA,VWN', 'PBE', 'r2SCAN', 'wB97X'):
    case(f'hess_water+_svp_{_xc}_uks', (HESS,))(hess_case(WATER, 'def2-svp', _xc, charge=1, spin=1))
case('hess_water_svp_b3lyp_df', (HESS,))(hess_case(WATER, 'def2-svp', 'B3LYP', density_fit=True))
case('hess_water+_svp_pbe_uks_df', (HESS,))(hess_case(WATER, 'def2-svp', 'PBE', charge=1, spin=1,
                                                      density_fit=True))
case('scf_benzene+_svp_r2scan_uks', (FULL,))(scf_case(BENZENE, 'def2-svp', 'r2SCAN', charge=1, spin=1))
case('grad_caffeine_svp_b3lyp', (FULL,))(grad_case(CAFFEINE, 'def2-svp', 'B3LYP'))
case('tda_caffeine_svp_b3lyp', (FULL,))(td_case(CAFFEINE, 'def2-svp', 'B3LYP', 'TDA'))
case('scf_c20h42_svp_b3lyp', (FULL,))(scf_case(C20H42, 'def2-svp', 'B3LYP'))
case('scf_c20h42_tzvp_pbe', (FULL,))(scf_case(C20H42, 'def2-tzvp', 'PBE'))

case('micro_gemm', (MICRO,))(micro_gemm())
case('micro_libxc', (MICRO,))(micro_libxc())
