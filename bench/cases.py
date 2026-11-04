"""Benchmark/validation cases.

A case is a function ``setup(tight) -> run``; only ``run()`` is timed. ``run()`` returns a dict of
JSON-serializable values used by validate.py. ``tight=True`` selects convergence settings for
correctness comparisons; timing runs use PySCF defaults. pyscf is imported inside the functions so the
worker picks up whichever tree PYTHONPATH selects.
"""
import os

import numpy as np
from molecules import CAFFEINE, BENZENE, WATER, n_alkane

CASES = {}  # name -> (setup, suites)


def case(name, suites):
    def deco(fn):
        CASES[name] = (fn, suites)
        return fn
    return deco


# Tight SCF settings use conv_tol=1e-12, except for molecules with more than LARGE_NATM atoms: on the
# 62-atom alkane |dE| < 1e-12 (a few ULPs of ~800 Eh) is met erratically in every build, unmodified
# upstream included (14-100 cycles, converged energies spread over 1e-10), as direct J/K are summed in
# dynamic thread order and updated incrementally (see _full_jk_rebuild).
LARGE_NATM = 40


def _full_jk_rebuild(mf):
    '''Build J/K from the full density in every SCF cycle. Direct SCF (dft/rks.py get_veff) adds the J/K
    of each density change to the previous cycle's, and the screening error of these updates accumulates:
    on n-C20H42/def2-TZVP the energy drifts by ~1e-10 Eh per cycle once the density is stationary, so
    conv_tol=1e-11 is never met (100 cycles and 1.4e-8 Eh of drift, in every build).'''
    get_veff = mf.get_veff
    mf.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_veff(mol, dm, hermi=hermi)


def make_ks(mol, xc, tight, density_fit=False, full_jk=False):
    from pyscf import dft, scf
    mf = dft.RKS(mol, xc=xc) if mol.spin == 0 else dft.UKS(mol, xc=xc)
    mf.chkfile = None
    # SCF guess and convergence knobs (It21), for A/B runs within one tree
    if os.environ.get('BENCH_INIT_GUESS'):
        mf.init_guess = os.environ['BENCH_INIT_GUESS']
    if os.environ.get('BENCH_DIIS'):
        mf.DIIS = getattr(scf.diis, os.environ['BENCH_DIIS'])
    if os.environ.get('BENCH_DIIS_SPACE'):
        mf.diis_space = int(os.environ['BENCH_DIIS_SPACE'])
    if os.environ.get('BENCH_PRE_SCF_TOL'):
        mf.pre_scf_conv_tol = float(os.environ['BENCH_PRE_SCF_TOL'])
    if os.environ.get('BENCH_PRE_SCF_LEVEL'):
        mf.pre_scf_grids_level = int(os.environ['BENCH_PRE_SCF_LEVEL'])
    # the VV10 (NLC) grid: a grid level, or sg1 for (50, 194) with SG-1 pruning (Q-Chem's default NLC grid)
    if os.environ.get('BENCH_NLC_GRID'):
        from pyscf.dft import gen_grid
        if os.environ['BENCH_NLC_GRID'] == 'sg1':
            mf.nlcgrids.atom_grid = (50, 194)
            mf.nlcgrids.prune = gen_grid.sg1_prune
        else:
            mf.nlcgrids.level = int(os.environ['BENCH_NLC_GRID'])
    if tight:
        mf.conv_tol = 1e-12 if mol.natm <= LARGE_NATM else 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 100
        if full_jk:
            _full_jk_rebuild(mf)
    if density_fit:
        mf = mf.density_fit()
    elif os.environ.get('BENCH_SGX'):
        # seminumerical exchange (pyscf.sgx, COSX-like): sgx = SGX for J and K, no screening (sgx_fit's default);
        # pjs = density-fitted J and SGX K with P-junction screening
        from pyscf import sgx
        mf = sgx.sgx_fit(mf, pjs=os.environ['BENCH_SGX'] == 'pjs')
    return mf


def scf_case(atom, basis, xc, charge=0, spin=0, density_fit=False, full_jk=False):
    def setup(tight):
        from pyscf import gto
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)

        def run():
            mf = make_ks(mol, xc, tight, density_fit, full_jk)
            mf.kernel()
            out = {'e_tot': mf.e_tot, 'cycles': mf.cycles, 'converged': bool(mf.converged)}
            if mf.nlcgrids.weights is not None:  # VV10 functionals: the NLC grid after small-rho pruning
                out['nlc_points'] = int(mf.nlcgrids.weights.size)
            sgx_grids = getattr(getattr(mf, 'with_df', None), 'grids', None)
            if getattr(sgx_grids, 'weights', None) is not None:  # BENCH_SGX: the SGX grid of the last cycle
                out['sgx_points'] = int(sgx_grids.weights.size)
            return out
        return run
    return setup


def converged_ks(atom, basis, xc, tight, charge=0, spin=0, density_fit=False, full_jk=False):
    '''A converged KS object for the post-SCF cases. In validation the reference writes its SCF solution to
    BENCH_SCF_OUT and the candidate starts from it (BENCH_SCF_IN) instead of its own SCF: the endpoint of a
    tight direct SCF scatters between runs of any build, because it stops at a cycle that depends on the
    summation order of J/K (caffeine/B3LYP: 24-27 cycles, densities up to 6e-9 apart), and that moves TD
    excitation energies by ~2e-9 Eh. Post-SCF cases thus compare post-SCF code on the same orbitals; the
    scf_ cases cover the SCF.'''
    from pyscf import gto
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = make_ks(mol, xc, tight, density_fit, full_jk)
    if os.environ.get('BENCH_SCF_IN'):
        s = np.load(os.environ['BENCH_SCF_IN'])
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = s['mo_coeff'], s['mo_occ'], s['mo_energy']
        mf.e_tot = float(s['e_tot'])
        mf.converged = True
        return mf
    mf.kernel()
    assert mf.converged
    if os.environ.get('BENCH_SCF_OUT'):
        np.savez(os.environ['BENCH_SCF_OUT'], mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
                 mo_energy=mf.mo_energy, e_tot=mf.e_tot)
    return mf


def grad_case(atom, basis, xc, charge=0, spin=0, full_jk=False):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight, charge, spin, full_jk=full_jk)

        def run():
            g = mf.nuc_grad_method().kernel()
            return {'e_tot': mf.e_tot, 'grad': g.tolist()}
        return run
    setup.post_scf = True  # starts from converged_ks, see there
    return setup


def td_case(atom, basis, xc, method, nstates=5, charge=0, spin=0, full_jk=False):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight, charge, spin, full_jk=full_jk)

        def run():
            td = getattr(mf, method)()
            td.nstates = nstates
            if tight:
                # conv_tol is the residual-norm tolerance of the eigensolver (tdscf/_lr_eig.py), which drops
                # preconditioned corrections shorter than sqrt(lindep) = 1e-6. On caffeine the residuals then
                # stall near 1.5e-7 (the excitation energies have converged to 1e-13 Eh by then), so 1e-8 ran
                # to max_cycle. At 1e-6 the energies are converged to ~1e-11 Eh.
                td.conv_tol = 1e-6
                td.max_cycle = 200
            td.kernel()
            return {'e_tot': mf.e_tot, 'excitations': np.asarray(td.e).tolist(),
                    'converged': np.asarray(td.converged).tolist()}
        return run
    setup.post_scf = True  # starts from converged_ks, see there
    return setup


def hess_case(atom, basis, xc, charge=0, spin=0, density_fit=False, full_jk=False):
    def setup(tight):
        mf = converged_ks(atom, basis, xc, tight, charge, spin, density_fit, full_jk)

        def run():
            h = mf.Hessian().kernel()
            return {'e_tot': mf.e_tot, 'hessian': h.tolist()}
        return run
    setup.post_scf = True  # starts from converged_ks, see there
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
SPARSE = 'sparse'  # large molecules, which take the screened (sparse) grid path
C20H42 = n_alkane(20)

case('scf_caffeine_svp_pbe', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'PBE'))
case('scf_caffeine_svp_b3lyp', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'B3LYP'))
case('scf_caffeine_svp_r2scan', (QUICK, FULL))(scf_case(CAFFEINE, 'def2-svp', 'r2SCAN'))
case('grad_benzene_svp_b3lyp', (QUICK, FULL))(grad_case(BENZENE, 'def2-svp', 'B3LYP'))
case('tda_benzene_svp_b3lyp', (QUICK, FULL))(td_case(BENZENE, 'def2-svp', 'B3LYP', 'TDA'))
case('grid_caffeine_svp', (QUICK, FULL))(grid_case(CAFFEINE, 'def2-svp'))

case('scf_caffeine_tzvp_dfpbe', (FULL,))(scf_case(CAFFEINE, 'def2-tzvp', 'PBE', density_fit=True))
# Conventional TZVP hybrid (ERIs 60 GB, of which 41 GB pass the Schwarz test), and a range-separated
# hybrid, whose long-range K is direct even when the regular ERIs are in memory. The tight TZVP SCF
# with incremental J/K drifts and did not converge in 100 cycles in unmodified PySCF; see _full_jk_rebuild.
case('scf_caffeine_tzvp_b3lyp', (FULL,))(scf_case(CAFFEINE, 'def2-tzvp', 'B3LYP', full_jk=True))
# The tight CAM-B3LYP SCF with incremental J/K stalled short of conv_tol=1e-12 in 1 of 3 runs in both
# the It13 and It15 trees (results/camb3lyp-flaky.log), so it rebuilds J/K too.
case('scf_caffeine_svp_camb3lyp', (FULL,))(scf_case(CAFFEINE, 'def2-svp', 'CAM-B3LYP', full_jk=True))
# Tight direct UKS drifts by up to ~5e-12 Eh per cycle with incremental J/K, so conv_tol=1e-12 was met
# or missed by chance (47-51 cycles, or not within 100, in every build); see _full_jk_rebuild.
case('scf_caffeine+_svp_b3lyp_uks', (FULL,))(scf_case(CAFFEINE, 'def2-svp', 'B3LYP', charge=1, spin=1,
                                                     full_jk=True))
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
# Hessians for timings only (by --cases; no suite): a medium molecule, and meta-GGA
case('hess_caffeine_svp_b3lyp', ())(hess_case(CAFFEINE, 'def2-svp', 'B3LYP'))
case('hess_benzene_svp_r2scan', ())(hess_case(BENZENE, 'def2-svp', 'r2SCAN'))
case('hess_caffeine_svp_r2scan', ())(hess_case(CAFFEINE, 'def2-svp', 'r2SCAN'))
case('hess_water+_svp_pbe_uks_df', (HESS,))(hess_case(WATER, 'def2-svp', 'PBE', charge=1, spin=1,
                                                      density_fit=True))
# UKS TD code paths, also in the 'hessian' suite (the response functions it shares with the UKS
# Hessians): TDA for each functional type, and TDDFT, which is Casida's for pure functionals and the
# full RPA for hybrids
for _xc in ('LDA,VWN', 'PBE', 'r2SCAN', 'B3LYP', 'wB97X'):
    case(f'tda_water+_svp_{_xc}_uks', (HESS,))(td_case(WATER, 'def2-svp', _xc, 'TDA', charge=1, spin=1))
for _xc in ('PBE', 'r2SCAN', 'B3LYP'):
    case(f'tddft_water+_svp_{_xc}_uks', (HESS,))(td_case(WATER, 'def2-svp', _xc, 'TDDFT', charge=1,
                                                          spin=1))
# UKS response on medium molecules (caffeine+: full J/K builds in the tight SCF, as for its scf_ case)
case('tda_caffeine+_svp_b3lyp_uks', (FULL,))(td_case(CAFFEINE, 'def2-svp', 'B3LYP', 'TDA', charge=1,
                                                     spin=1, full_jk=True))
case('hess_benzene+_svp_b3lyp_uks', (FULL,))(hess_case(BENZENE, 'def2-svp', 'B3LYP', charge=1, spin=1))
case('hess_caffeine+_svp_b3lyp_uks', ())(hess_case(CAFFEINE, 'def2-svp', 'B3LYP', charge=1, spin=1,
                                                   full_jk=True))
case('scf_benzene+_svp_r2scan_uks', (FULL,))(scf_case(BENZENE, 'def2-svp', 'r2SCAN', charge=1, spin=1))
case('grad_caffeine_svp_b3lyp', (FULL,))(grad_case(CAFFEINE, 'def2-svp', 'B3LYP'))
# UKS gradient (the UHF branch of the J/K energy gradient), and the gradient code paths of range-separated
# hybrids and UKS on water in the 'hessian' suite
case('grad_caffeine+_svp_b3lyp_uks', (FULL,))(grad_case(CAFFEINE, 'def2-svp', 'B3LYP', charge=1, spin=1,
                                                      full_jk=True))
case('grad_water_svp_CAM-B3LYP', (HESS,))(grad_case(WATER, 'def2-svp', 'CAM-B3LYP'))
for _xc in ('PBE', 'B3LYP', 'wB97X'):
    case(f'grad_water+_svp_{_xc}_uks', (HESS,))(grad_case(WATER, 'def2-svp', _xc, charge=1, spin=1))
case('tda_caffeine_svp_b3lyp', (FULL,))(td_case(CAFFEINE, 'def2-svp', 'B3LYP', 'TDA'))
# wB97M-V: VV10 on the NLC grid, which BENCH_NLC_GRID sets (node2 comparison against GPU4PySCF's part 33)
case('scf_caffeine_svp_wb97mv', ())(scf_case(CAFFEINE, 'def2-svp', 'wB97M-V'))
case('grad_caffeine_svp_wb97mv', ())(grad_case(CAFFEINE, 'def2-svp', 'wB97M-V'))
case('scf_c20h42_svp_b3lyp', (FULL, SPARSE))(scf_case(C20H42, 'def2-svp', 'B3LYP'))
case('scf_c20h42_tzvp_pbe', (FULL, SPARSE))(scf_case(C20H42, 'def2-tzvp', 'PBE', full_jk=True))
case('scf_c20h42_tzvp_b3lyp', ())(scf_case(C20H42, 'def2-tzvp', 'B3LYP', full_jk=True))
case('scf_c20h42_svp_r2scan', (SPARSE,))(scf_case(C20H42, 'def2-svp', 'r2SCAN'))
case('grad_c20h42_svp_b3lyp', (SPARSE,))(grad_case(C20H42, 'def2-svp', 'B3LYP'))
case('tda_c20h42_svp_pbe', (SPARSE,))(td_case(C20H42, 'def2-svp', 'PBE', 'TDA'))

case('micro_gemm', (MICRO,))(micro_gemm())
case('micro_libxc', (MICRO,))(micro_libxc())
