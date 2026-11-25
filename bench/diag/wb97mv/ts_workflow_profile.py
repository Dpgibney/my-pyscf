'''The steps of examples/wb97mv_ts.py, timed by their parts, for a closed- or open-shell molecule: wB97M-V, grids
(99,590) and (50,194), SMD water, the two-level SCF, response grids 2/1; the exact-J/K SCF, gradient and analytical
Hessian (grid_response, sgx_response), and the RIJCOSX (aug_etb) SCF and gradient of the search. The functions of
the Hessian named in PARTS are wrapped: each call's wall time is accumulated and, above 5 s, printed as it ends,
with the resident memory. Nested parts overlap (the table at the end is not a sum).
Usage: ts_workflow_profile.py XYZ [--charge 0] [--spin 0] [--basis def2-tzvp] [--steps scf,grad,hess,rx]
           [--solvent water|none] [--no-grid-response] [--no-sgx-response] [--response-auxbasis aug_etb]
           [--cphf-atoms N] [--save PREFIX] [--load PREFIX] [--hess-out FILE] [--cphf-sets 18,57,171]
  --cphf-atoms N  the CPHF of the first N atoms only (their perturbations in one batch): a sample of its cost
  --save/--load   the SCF's orbitals (npz); --load starts the SCF from them
  --steps cphf    one call of the CPHF's response function for each number of perturbations of --cphf-sets, with
                  the orbitals of --load (no SCF) and first-order trial vectors of the core Hamiltonian's
                  derivatives: the cost of a CPHF iteration by its parts, without the hours of the Hessian
                  before it. --cphf-rij: then also the RI-J (aug_etb) of the same total densities, alone, against
                  the direct J of the response
  --rx-profile    in step rx, a second displaced geometry under cProfile (the functions by cumulative time)'''
import argparse, time, functools, sys
import numpy
import pyscf
from pyscf import gto, dft, df, lib, grad, hessian  # noqa: F401
from pyscf.hessian import rhf as rhf_hess, uhf as uhf_hess, rks as rks_hess, uks as uks_hess
from pyscf.hessian import thermo
from pyscf.scf import cphf, ucphf, _response_functions  # noqa: F401
from pyscf.dft import numint

p = argparse.ArgumentParser()
p.add_argument('mol'); p.add_argument('--charge', type=int, default=0); p.add_argument('--spin', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp'); p.add_argument('--steps', default='scf,grad,hess,rx')
p.add_argument('--solvent', default='water'); p.add_argument('--no-grid-response', action='store_true')
p.add_argument('--no-sgx-response', action='store_true'); p.add_argument('--response-auxbasis')
p.add_argument('--cphf-atoms', type=int); p.add_argument('--save'); p.add_argument('--load')
p.add_argument('--hess-out'); p.add_argument('--xc', default='wb97m-v')
p.add_argument('--verbose', type=int, default=4); p.add_argument('--cphf-sets', default='18,57,171'); p.add_argument('--cphf-rij', action='store_true'); p.add_argument('--rx-profile', action='store_true')
a = p.parse_args()
steps = a.steps.split(',')
T0 = time.perf_counter()
TIMES = {}


def say(msg):
    print(f'>>> [{time.perf_counter() - T0:7.0f} s, {lib.current_memory()[0] / 1e3:5.1f} GB] {msg}', flush=True)


def timed(fn, label, quiet=5.):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t
            n, tot = TIMES.get(label, (0, 0.))
            TIMES[label] = (n + 1, tot + dt)
            if dt >= quiet:
                say(f'{label}: {dt:.1f} s (call {n + 1}, total {tot + dt:.1f} s)')
    return wrapper


def wrap(obj, name, label=None, quiet=5.):
    if hasattr(obj, name):
        setattr(obj, name, timed(getattr(obj, name), label or name, quiet))


PARTS = [
    # the energy's second derivative at fixed orbitals
    (rhf_hess, 'hess_jk_energy'), (uhf_hess, '_partial_hess_ejk'), (rhf_hess, '_partial_hess_ejk'),
    (uks_hess, '_get_vxc_diag', 'uks._get_vxc_diag'), (uks_hess, '_get_vxc_deriv2_dm', 'uks._get_vxc_deriv2_dm'),
    (rks_hess, '_get_vxc_diag', 'rks._get_vxc_diag'), (rks_hess, '_get_vxc_deriv2_dm', 'rks._get_vxc_deriv2_dm'),
    (rks_hess, '_get_exc_deriv2_grid_response'), (rks_hess, '_get_enlc_deriv2'),
    (rks_hess, '_vv10_grid_response_terms'), (rks_hess, '_becke_weight_derivs'),
    (rks_hess._VV10HessData, 'weight1_pair', 'vv10.weight1_pair'),
    (rks_hess._VV10HessData, 'grid_response_pair', 'vv10.grid_response_pair'),
    (rks_hess._VV10HessData, '__init__', 'vv10 data'),
    # the Fock matrix derivatives
    (uks_hess, '_get_vxc_deriv1', 'uks._get_vxc_deriv1'), (rks_hess, '_get_vxc_deriv1', 'rks._get_vxc_deriv1'),
    (rks_hess, '_get_vnlc_deriv1'),
    (uks_hess, '_get_jk', 'make_h1 _get_jk', 1e9), (uhf_hess, '_get_jk', 'make_h1 _get_jk', 1e9),
    (rhf_hess, '_get_jk', 'make_h1 _get_jk', 1e9),
    # the CPHF equations
    (ucphf, 'solve_residual', 'ucphf.solve_residual'), (cphf, 'solve_residual', 'cphf.solve_residual'),
    (rks_hess, 'get_vnlc_resp'), (rks_hess, 'get_vnlc_resp_mo'),
    (numint.NumInt, 'nr_uks_fxc'), (numint.NumInt, 'nr_rks_fxc'), (numint.NumInt, 'cache_xc_kernel'),
    (rhf_hess, '_hess_elec_mo1'),
    # the Hessian's steps
    (uks_hess.Hessian, 'partial_hess_elec', 'partial_hess_elec'), (rks_hess.Hessian, 'partial_hess_elec', 'partial_hess_elec'),
    # (make_h1 itself is not wrapped: hess_elec looks its split form up by the function, hessian.rhf._MAKE_H1_SPLIT)
    (uks_hess.Hessian, 'solve_mo1', 'solve_mo1'), (rks_hess.Hessian, 'solve_mo1', 'solve_mo1'),
]
for part in PARTS:
    wrap(*part)
# the fused kernels of the XC response
for name in dir(numint):
    cls = getattr(numint, name)
    if isinstance(cls, type):
        for meth in ('uks_fxc_mo', 'rks_fxc_mo'):
            if meth in vars(cls):
                wrap(cls, meth, meth)
# the solvent's terms
try:
    from pyscf.solvent.hessian import pcm as pcm_hess, smd as smd_hess
    from pyscf.solvent import pcm as pcm_mod
    for name in ('analytical_grad_vmat', 'analytical_hess_nuc', 'analytical_hess_qv', 'analytical_hess_solver'):
        wrap(pcm_hess, name, 'solvent ' + name)
    wrap(smd_hess, 'get_cds', 'solvent get_cds')
    wrap(pcm_mod.PCM, '_B_dot_x', 'solvent _B_dot_x', 1e9)
except ImportError:
    pass
# SGX's K of many densities (module functions: a wrapper on the SGX object would be copied to its range-separated
# copies, bound to the wrong object)
from pyscf.sgx import sgx_jk
wrap(sgx_jk, 'get_k_mo', 'response sgx_jk.get_k_mo (factors)')
wrap(sgx_jk, '_get_k_only_many', 'response sgx_jk._get_k_only_many')
# the J and K of the CPHF response under sgx_response
_sgx_scf = rhf_hess._sgx_response_scf


def _sgx_response_scf(*args, **kwargs):
    mf = _sgx_scf(*args, **kwargs)
    for name in ('get_jk', 'get_j', 'get_k'):
        setattr(mf, name, timed(getattr(mf, name), 'response ' + name))
    with_df = mf.with_df
    with_df.verbose = 6     # debug1: the parts of each seminumerical K (AO values, F, G, K)
    return mf


rhf_hess._sgx_response_scf = _sgx_response_scf

if a.cphf_atoms:
    # the perturbations of the first atoms only, in one batch
    for mod in (uhf_hess, rhf_hess):
        solve = mod.solve_mo1

        def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao, fx=None, atmlst=None, *args, _solve=solve, **kwargs):
            out = _solve(mf, mo_energy, mo_coeff, mo_occ, h1ao, fx, range(a.cphf_atoms), *args, **kwargs)

            def fill(xs):
                # zeros for the atoms left out, so that the steps after the CPHF run (the Hessian is not one)
                if isinstance(xs, tuple):
                    return tuple(fill(x) for x in xs)
                return [numpy.zeros_like(xs[0]) if x is None else x for x in xs]
            return fill(out[0]), fill(out[1])
        mod.solve_mo1 = solve_mo1


def table(title):
    say(title)
    for label, (n, tot) in sorted(TIMES.items(), key=lambda kv: -kv[1][1]):
        print(f'      {tot:9.1f} s  {n:5d} call(s)  {label}', flush=True)
    TIMES.clear()


mol = gto.M(atom=a.mol, basis=a.basis, charge=a.charge, spin=a.spin, verbose=a.verbose)
UKS = a.spin != 0


def make_mf(mol, rijcosx=False):
    mf = (dft.UKS if UKS else dft.RKS)(mol, xc=a.xc)
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    if rijcosx:
        mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
    if a.solvent != 'none':
        mf = mf.SMD()
        mf.with_solvent.solvent = a.solvent
    mf.two_level = True
    mf.set_second_grids(2)
    mf.set_second_nlcgrids(1)
    return mf


say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads, max_memory {mol.max_memory} MB; {mol.natm} atoms, '
    f'{mol.nao} AOs, {mol.nelec} electrons, {"UKS" if UKS else "RKS"} {a.xc}/{a.basis}, solvent {a.solvent}')
mf = make_mf(mol)
if 'cphf' in steps:
    z = numpy.load(a.load + '.npz')
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = z['mo_coeff'], z['mo_occ'], z['mo_energy']
    mf.e_tot, mf.converged = float(z['e_tot']), True
    dm = mf.make_rdm1()
    if a.solvent != 'none':
        mf.with_solvent.build()
        mf.with_solvent._get_vind(dm[0] + dm[1] if UKS else dm)
    hobj = mf.Hessian()
    hobj.sgx_response = not a.no_sgx_response
    if a.response_auxbasis:
        hobj.sgx_response_auxbasis = a.response_auxbasis
    hobj.verbose = 6
    base = mf
    if hobj.sgx_response:
        base = rhf_hess._sgx_response_scf(mf, hobj.sgx_response_level, hobj.sgx_response_auxbasis)
    nsets = [int(x) for x in a.cphf_sets.split(',')]
    natm = min(mol.natm, (max(nsets) + 2) // 3)
    hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
    mo_coeff = mf.mo_coeff if UKS else (mf.mo_coeff,)
    mo_occ = mf.mo_occ if UKS else (mf.mo_occ,)
    mo_energy = mf.mo_energy if UKS else (mf.mo_energy,)
    xs = []
    for c, occ, e in zip(mo_coeff, mo_occ, mo_energy):
        # x[a,i] = -h1[a,i] / (e_a - e_i) in the virtual rows, zero in the occupied ones
        x = numpy.array([c.T @ h @ c[:, occ > 0] for ia in range(natm) for h in hcore_deriv(ia)])
        x /= -(e[:, None] - e[occ > 0])
        x[:, occ > 0] = 0
        xs.append(x.reshape(len(x), -1))
    x = numpy.hstack(xs)
    say(f'CPHF response: {len(x)} trial vectors of {x.shape[1]} elements, sgx_response {hobj.sgx_response}, '
        f'response auxbasis {hobj.sgx_response_auxbasis}')
    solv = lib.temporary_env(mf.with_solvent, equilibrium_solvation=True) if a.solvent != 'none' else lib.temporary_env(mf)
    with solv:
        fx = (uhf_hess if UKS else rhf_hess).gen_vind(base, mf.mo_coeff, mf.mo_occ)
        for n in nsets:
            t = time.perf_counter()
            v = fx(x[:n])
            say(f'response of {n} perturbations: {time.perf_counter() - t:.1f} s (|v| {numpy.linalg.norm(v):.6f})')
            table(f'parts of the response of {n} perturbations (wall time, nested parts overlap):')
        if a.cphf_rij:
            # the Coulomb response alone, of the total densities of the last number of perturbations: RI-J in
            # aug_etb (hobj.sgx_response_auxbasis = 'aug_etb') and the J of the response above
            n = nsets[-1]
            mocc = [c[:, occ > 0] for c, occ in zip(mo_coeff, mo_occ)]
            for label, mfj in (('RI-J (aug_etb)', rhf_hess._sgx_response_scf(mf, hobj.sgx_response_level, 'aug_etb')),
                               ('the J of the response above', base)):
                t = time.perf_counter()
                norm = 0.
                for p0 in range(0, n, 57):
                    d = 0
                    for c, q, xk in zip(mo_coeff, mocc, xs):
                        ak = numpy.einsum('pa,kai->kpi', c, xk[p0:p0+57].reshape(-1, c.shape[1], q.shape[1]))
                        d = d + numpy.matmul(ak, q.T)
                    d = d + d.transpose(0, 2, 1)
                    if not UKS:
                        d *= 2
                    norm += numpy.linalg.norm(mfj.get_j(mol, d, hermi=1))**2
                say(f'{label} of the total densities of {n} perturbations, 57 at a time: {time.perf_counter() - t:.1f} s '
                    f'(|J| {norm**.5:.6f})')
            TIMES.clear()
if 'scf' in steps or 'grad' in steps or 'hess' in steps:
    dm0 = None
    if a.load:
        z = numpy.load(a.load + '.npz')
        dm0 = mf.make_rdm1(z['mo_coeff'], z['mo_occ'])
    t = time.perf_counter()
    mf.kernel(dm0=dm0)
    say(f'exact SCF: {time.perf_counter() - t:.1f} s, E = {mf.e_tot:.10f}, converged {mf.converged}; '
        f'XC grid {mf.grids.weights.size} points, VV10 grid {mf.nlcgrids.weights.size}'
        + (f', {mf.with_solvent.surface["grid_coords"].shape[0]} surface points' if a.solvent != 'none' else ''))
    if a.save:
        numpy.savez(a.save + '.npz', mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy, e_tot=mf.e_tot)
if 'grad' in steps:
    t = time.perf_counter()
    g = mf.Gradients().kernel()
    say(f'exact gradient: {time.perf_counter() - t:.1f} s, max |g| {abs(g).max():.2e}')
if 'hess' in steps:
    hobj = mf.Hessian()
    hobj.grid_response = not a.no_grid_response
    hobj.sgx_response = not a.no_sgx_response
    if a.response_auxbasis:
        hobj.sgx_response_auxbasis = a.response_auxbasis
    hobj.verbose = 6    # debug1: the timers of the Hessian
    t = time.perf_counter()
    h = hobj.kernel()
    say(f'Hessian: {time.perf_counter() - t:.1f} s (grid_response {hobj.grid_response}, sgx_response '
        f'{hobj.sgx_response}, response auxbasis {hobj.sgx_response_auxbasis})')
    table('parts of the Hessian (wall time, nested parts overlap):')
    if a.hess_out:
        numpy.save(a.hess_out, h)
    if not a.cphf_atoms:
        w = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
        say('lowest frequencies (cm-1): ' + ' '.join(f'{x:.1f}' for x in w[:8]))
if 'rx' in steps:
    mfx = make_mf(mol, True)
    dm0 = None
    if a.load:
        z = numpy.load(a.load + '.npz')
        dm0 = mfx.make_rdm1(z['mo_coeff'], z['mo_occ'])
    t = time.perf_counter()
    mfx.kernel(dm0=dm0)
    say(f'RIJCOSX SCF{" from the density of --load" if a.load else ""}: {time.perf_counter() - t:.1f} s, '
        f'E = {mfx.e_tot:.10f}, converged {mfx.converged}, {mfx.cycles if hasattr(mfx, "cycles") else "?"} cycles')
    t = time.perf_counter()
    gs = mfx.Gradients().as_scanner()
    e, g = gs(mol)
    say(f'RIJCOSX scanner, first call (SCF from its own density + gradient): {time.perf_counter() - t:.1f} s')
    # a second geometry, as a step of the search: the SCF starts from the last density
    x = mol.atom_coords() + numpy.random.default_rng(0).normal(scale=.02, size=(mol.natm, 3))
    t = time.perf_counter()
    e, g = gs(mol.set_geom_(x, unit='Bohr', inplace=False))
    say(f'RIJCOSX scanner, displaced geometry (0.02 Bohr rms per coordinate): {time.perf_counter() - t:.1f} s, '
        f'E = {e:.10f}, max |g| {abs(g).max():.2e}')
    if a.rx_profile:
        # the same step again from another displacement, under cProfile: where a step of the search spends its time
        import cProfile, pstats, io
        x = mol.atom_coords() + numpy.random.default_rng(1).normal(scale=.02, size=(mol.natm, 3))
        pr = cProfile.Profile()
        t = time.perf_counter()
        pr.enable()
        e, g = gs(mol.set_geom_(x, unit='Bohr', inplace=False))
        pr.disable()
        say(f'RIJCOSX scanner, a second displaced geometry, profiled: {time.perf_counter() - t:.1f} s')
        out = io.StringIO()
        pstats.Stats(pr, stream=out).sort_stats('cumulative').print_stats(90)
        print(out.getvalue(), flush=True)
say(f'total {time.perf_counter() - T0:.0f} s')
