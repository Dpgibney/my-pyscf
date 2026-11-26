'''Two prunings measured on the Hessian of examples/wb97mv_ts.py's model (wB97M-V/def2-TZVP, XC (99,590), VV10
(50,194), SMD water, two_level, response grids 2/1, grid_response, sgx_response with RI-J in aug_etb):

- rho:X     mf.small_rho_cutoff = X: the points of the SCF's XC and VV10 grids (and the cheap model's) where
            rho w < X / npoints are dropped when the grids are built (PySCF's prune_by_density_, off by default
            since #3192). The SCF starts from the reference's orbitals, so the points go by its density.
- rhoall:X  the same, and the response grids (second_grids, second_nlcgrids) pruned alike before the Hessian.
- vv10:R    the reference SCF; in the Hessian the pair kernels of VV10 with many densities (eval_f_t: the density
            responses of the energy term and of every CPHF iteration; eval_EUW_with_weight1: the weight
            derivatives) from libvv10cut.so (bench/diag/prune/vv10_cut.c), which leaves out the pairs of blocks of
            128 points whose bounding spheres are more than R Bohr apart. The points go to the kernel in the order
            of a k-d split into blocks of 128 (compact blocks); vv10:0 keeps every pair (the order alone).
- ref       neither (ref2: again, at the end, for the drift of the timings).

Each variant: E, the gradient, the Hessian's time by its parts, the harmonic frequencies and the free energy (1 mol/L,
298.15 K, quasi-RRHO 100 cm^-1, as the example), compared at the end with ref.

Usage: prune_check.py XYZ --spin 1 [--variants ref,rho:1e-7,vv10:12,...,ref2] [--load NPZ] [--save NPZ]
                         [--json OUT.jsonl] [--cutlib libvv10cut.so]
       prune_check.py XYZ --spin 1 --opt OUT.xyz        (geomeTRIC minimum of the model first)
       prune_check.py XYZ --spin 1 --ts OUT.xyz         (or a saddle point, from the exact Hessian at XYZ)
       prune_check.py XYZ --spin 1 --big [--rcuts 16,12,9,6] [--rho-cutoffs 1e-7,1e-6,1e-5] [--ntrial 16]
                         (no SCF: the grids of the initial guess's density; the points each cutoff keeps, an XC
                          potential on the full and the pruned grid, and eval_f_t with and without the VV10 cutoff)
'''
import argparse, ctypes, functools, json, os, sys, time
import numpy
import pyscf
from pyscf import gto, dft, df, lib, symm, grad, hessian  # noqa: F401
from pyscf.data import nist
from pyscf.hessian import rhf as rhf_hess, uhf as uhf_hess, rks as rks_hess, uks as uks_hess
from pyscf.hessian import thermo
from pyscf.scf import cphf, ucphf, _vhf
from pyscf.dft import numint

p = argparse.ArgumentParser()
p.add_argument('mol'); p.add_argument('--charge', type=int, default=0); p.add_argument('--spin', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp'); p.add_argument('--solvent', default='water')
p.add_argument('--variants', default='ref'); p.add_argument('--load'); p.add_argument('--save')
p.add_argument('--json'); p.add_argument('--opt'); p.add_argument('--ts'); p.add_argument('--big', action='store_true')
p.add_argument('--rcuts', default='16,12,9,6'); p.add_argument('--rho-cutoffs', default='1e-7,1e-6,1e-5')
p.add_argument('--ntrial', type=int, default=16)
p.add_argument('--cutlib', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                                                 'scratch', 'p44', 'libvv10cut.so'))
p.add_argument('--verbose', type=int, default=4)
a = p.parse_args()
T0 = time.perf_counter()
TIMES = {}


def say(msg):
    print(f'>>> [{time.perf_counter() - T0:7.0f} s, {lib.current_memory()[0] / 1e3:5.1f} GB] {msg}', flush=True)


def timed(fn, label):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            n, tot = TIMES.get(label, (0, 0.))
            TIMES[label] = (n + 1, tot + time.perf_counter() - t)
    return wrapper


PARTS = [
    (rhf_hess, 'hess_jk_energy'), (uhf_hess, '_partial_hess_ejk'),
    (uks_hess, '_get_vxc_diag', 'xc uks._get_vxc_diag'), (uks_hess, '_get_vxc_deriv2_dm', 'xc uks._get_vxc_deriv2_dm'),
    (rks_hess, '_get_exc_deriv2_grid_response', 'xc _get_exc_deriv2_grid_response'),
    (uks_hess, '_get_vxc_deriv1', 'xc uks._get_vxc_deriv1'),
    (rks_hess, '_get_enlc_deriv2', 'vv10 _get_enlc_deriv2'), (rks_hess, '_get_vnlc_deriv1', 'vv10 _get_vnlc_deriv1'),
    (rks_hess, 'get_vnlc_resp_mo', 'vv10 get_vnlc_resp_mo'),
    (rks_hess._VV10HessData, '__init__', 'vv10 data (ground-state pair sums)'),
    (rks_hess._VV10HessData, 'grid_response_pair', 'vv10 grid_response_pair (not cut)'),
    (ucphf, 'solve_residual', 'cphf ucphf.solve_residual'),
    (uks_hess.Hessian, 'partial_hess_elec', 'partial_hess_elec'), (uks_hess.Hessian, 'solve_mo1', 'solve_mo1'),
]
for part in PARTS:
    obj, name = part[:2]
    if hasattr(obj, name):
        setattr(obj, name, timed(getattr(obj, name), part[2] if len(part) > 2 else name))
for name in dir(numint):
    cls = getattr(numint, name)
    if isinstance(cls, type) and 'uks_fxc_mo' in vars(cls):
        setattr(cls, 'uks_fxc_mo', timed(cls.uks_fxc_mo, 'xc uks_fxc_mo (response grid)'))


def table(title):
    say(title)
    for label, (n, tot) in sorted(TIMES.items(), key=lambda kv: -kv[1][1]):
        print(f'      {tot:9.1f} s  {n:5d} call(s)  {label}', flush=True)


# ---- the VV10 pair kernels with a cutoff (libvv10cut.so), in place of _VV10HessData.f_t and .weight1_pair
CUT = None
MODE = {'rcut': None}        # None: the tree's kernels
KERNEL = {}                  # label -> [calls, seconds, block pairs kept, all]
F_T_KEYS = ('coords', 'weights', 'rho', 'omega', 'kappa', 'U', 'W', 'A', 'B', 'C', 'domega_drho', 'domega_dgamma',
            'dkappa_drho', 'd2omega_drho2', 'd2omega_dgamma2', 'd2omega_drho_dgamma', 'd2kappa_drho2')
F_T_ORIG = rks_hess._VV10HessData.f_t
W1_ORIG = rks_hess._VV10HessData.weight1_pair


def cutlib():
    global CUT
    if CUT is None:
        CUT = ctypes.CDLL(os.path.abspath(a.cutlib))
        CUT.VV10CUT_set_rcut.argtypes = [ctypes.c_double]
    return CUT


def kd_order(coords, bs=128):
    '''The points in blocks of bs (the last one partial): split at the median of the longest extent, the left part a
    multiple of bs, down to single blocks'''
    out = []

    def split(idx):
        n = idx.size
        if n <= bs:
            out.append(idx)
            return
        nl = (-(-n // bs) // 2) * bs
        c = coords[idx]
        ax = int(numpy.argmax(c.max(axis=0) - c.min(axis=0)))
        part = numpy.argpartition(c[:, ax], nl - 1)
        split(idx[part[:nl]])
        split(idx[part[nl:]])
    split(numpy.arange(len(coords)))
    return numpy.concatenate(out)


def _sorted(self):
    s = self.__dict__.get('_cut_sorted')
    if s is None:
        perm = kd_order(self.coords)
        s = {k: numpy.ascontiguousarray(getattr(self, k)[perm]) for k in F_T_KEYS}
        s['perm'] = perm
        self._cut_sorted = s
    return s


def _ptr(x):
    return x.ctypes.data_as(ctypes.c_void_p)


def _progress(self, label):
    import contextlib
    log = getattr(self, '_log', None)
    if log is None:
        return contextlib.nullcontext()
    return _vhf.oe_progress(log, label, unit='grid', progress=cutlib().VV10CUT_progress)


def _record(label, dt, cut):
    rec = KERNEL.setdefault(label, [0, 0., 0, 0])
    rec[0] += 1
    rec[1] += dt
    if cut:
        st = (ctypes.c_long * 2)()
        CUT.VV10CUT_stats(st)
        rec[2] += st[0]
        rec[3] += st[1]


def f_t(self, rho_t, gamma_t):
    t = time.perf_counter()
    rcut = MODE['rcut']
    if rcut is None:
        out = F_T_ORIG(self, rho_t, gamma_t)
    else:
        lib_ = cutlib()
        lib_.VV10CUT_set_rcut(rcut)
        rho_t = numpy.asarray(rho_t, dtype=numpy.double)
        gamma_t = numpy.asarray(gamma_t, dtype=numpy.double)
        nt = rho_t.shape[0]
        s = _sorted(self)
        perm = s['perm']
        rt = numpy.ascontiguousarray(rho_t[:, perm])
        gt = numpy.ascontiguousarray(gamma_t[:, perm])
        fr = numpy.empty((nt, self.ngrids))
        fg = numpy.empty((nt, self.ngrids))
        with _progress(self, 'VV10 pair kernel (cutoff %g Bohr), %d densities' % (rcut, nt)):
            lib_.VV10CUT_eval_f_t(_ptr(fr), _ptr(fg), *[_ptr(s[k]) for k in F_T_KEYS], _ptr(rt), _ptr(gt),
                                  ctypes.c_int(self.ngrids), ctypes.c_int(nt))
        rt = gt = None
        out = numpy.empty_like(fr), numpy.empty_like(fg)
        out[0][:, perm] = fr
        out[1][:, perm] = fg
    _record('f_t, %d points' % self.ngrids, time.perf_counter() - t, rcut is not None)
    return out


def weight1_pair(self, dw, keep=False):
    rcut = MODE['rcut']
    if rcut is None:
        t = time.perf_counter()
        out = W1_ORIG(self, dw, keep)
        _record('weight1, %d points' % self.ngrids, time.perf_counter() - t, False)
        return out
    # as _VV10HessData.weight1_pair
    euw = self.__dict__.pop('_euw_weight1', None)
    if euw is not None and euw[0].shape == dw.shape:
        return euw
    t = time.perf_counter()
    lib_ = cutlib()
    lib_.VV10CUT_set_rcut(rcut)
    s = _sorted(self)
    perm = s['perm']
    dws = numpy.ascontiguousarray(numpy.asarray(dw)[:, perm])
    nd = dws.shape[0]
    res = [numpy.empty((nd, self.ngrids)) for i in range(3)]
    with _progress(self, 'VV10 pair kernel (cutoff %g Bohr), %d weight derivatives' % (rcut, nd)):
        lib_.VV10CUT_eval_EUW_with_weight1(*[_ptr(x) for x in res], _ptr(s['coords']), _ptr(dws), _ptr(s['rho']),
                                           _ptr(s['omega']), _ptr(s['kappa']), ctypes.c_int(self.ngrids),
                                           ctypes.c_int(nd))
    euw = []
    for x in res:
        y = numpy.empty_like(x)
        y[:, perm] = x
        euw.append(y)
    if keep:
        self._euw_weight1 = [x.copy() for x in euw]
    _record('weight1, %d points' % self.ngrids, time.perf_counter() - t, True)
    return euw


rks_hess._VV10HessData.f_t = f_t
rks_hess._VV10HessData.weight1_pair = weight1_pair


def kernel_table():
    for label, (n, dt, kept, allp) in sorted(KERNEL.items()):
        frac = f', {kept / allp:.3f} of the block pairs' if allp else ''
        print(f'      {dt:9.1f} s  {n:5d} call(s)  VV10 {label}{frac}', flush=True)


# ---- the model
mol = gto.M(atom=a.mol, basis=a.basis, charge=a.charge, spin=a.spin, verbose=a.verbose)
UKS = a.spin != 0


def make_mf(mol, small_rho_cutoff=0):
    '''as examples/wb97mv_ts.py make_mf, with SMD water'''
    mf = (dft.UKS if UKS else dft.RKS)(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.small_rho_cutoff = small_rho_cutoff
    if a.solvent != 'none':
        mf = mf.SMD()
        mf.with_solvent.solvent = a.solvent
    mf.two_level = True
    mf.set_second_grids(2)
    mf.set_second_nlcgrids(1)
    return mf


def total_dm(dm):
    dm = numpy.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm


def free_energy(mf, freq_au):
    pressure = 101325.
    if getattr(mf, 'with_solvent', None) is not None:
        pressure = 1e3 * nist.AVOGADRO * nist.BOLTZMANN * 298.15
    with lib.temporary_env(symm.geom, TOLERANCE=1e-2):
        res = thermo.thermo(mf, freq_au, 298.15, pressure, qrrho=True, qrrho_freq0=100.)
    return res['G_tot'][0], res['G_tot'][0] - res['E0'][0] if 'E0' in res else None


say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads, max_memory {mol.max_memory} MB; {mol.natm} atoms, '
    f'{mol.nao} AOs, {mol.nelec} electrons, {"UKS" if UKS else "RKS"} wB97M-V/{a.basis}, solvent {a.solvent}')

if a.opt:
    from pyscf.geomopt.geometric_solver import optimize
    mf = make_mf(mol)
    t = time.perf_counter()
    mol_eq = optimize(mf, maxsteps=100)
    say(f'minimum: {time.perf_counter() - t:.0f} s')
    with open(a.opt, 'w') as f:
        f.write('%d\nminimum of the model of examples/wb97mv_ts.py (SMD water), from %s\n'
                % (mol_eq.natm, os.path.basename(a.mol)))
        for s, r in zip([mol_eq.atom_symbol(i) for i in range(mol_eq.natm)], mol_eq.atom_coords(unit='Angstrom')):
            f.write('%-2s %15.9f %15.9f %15.9f\n' % (s, *r))
    sys.exit()

if a.ts:
    # as examples/wb97mv_ts.py: the exact Hessian at the guess, then geomeTRIC's saddle-point search on the gradients
    # (tighter thresholds than the example's, for clean soft modes)
    from pyscf.geomopt import geometric_solver
    mf = make_mf(mol)
    t = time.perf_counter()
    mf.kernel()
    hobj = mf.Hessian()
    hobj.grid_response = True
    hobj.sgx_response = True
    hobj.sgx_response_auxbasis = 'aug_etb'
    h0 = hobj.kernel()
    w = numpy.asarray(thermo.harmonic_analysis(mol, h0, imaginary_freq=False)['freq_wavenumber']).real
    say(f'start Hessian: {time.perf_counter() - t:.0f} s; lowest frequencies (cm-1) ' +
        ' '.join(f'{x:.1f}' for x in w[:6]))
    g_scanner = mf.Gradients().as_scanner()
    g_scanner.verbose = 2
    t = time.perf_counter()
    conv, mol_ts = geometric_solver.kernel(g_scanner, transition=True, hessian=h0, maxsteps=100,
                                           convergence_energy=1e-7, convergence_grms=3e-5, convergence_gmax=1e-4,
                                           convergence_drms=3e-4, convergence_dmax=6e-4)
    say(f'saddle point: converged {conv}, {time.perf_counter() - t:.0f} s')
    with open(a.ts, 'w') as f:
        f.write('%d\nsaddle point of the model of examples/wb97mv_ts.py (SMD water), from %s\n'
                % (mol_ts.natm, os.path.basename(a.mol)))
        for s, r in zip([mol_ts.atom_symbol(i) for i in range(mol_ts.natm)], mol_ts.atom_coords(unit='Angstrom')):
            f.write('%-2s %15.9f %15.9f %15.9f\n' % (s, *r))
    sys.exit()

if a.big:
    mf = make_mf(mol)
    t = time.perf_counter()
    dm = mf.get_init_guess()
    dmt = total_dm(dm)
    say(f'initial guess {time.perf_counter() - t:.1f} s; nelec {numpy.einsum("ij,ji", dmt, mf.get_ovlp()):.6f}')
    ni = mf._numint
    cutoffs = [float(x) for x in a.rho_cutoffs.split(',')]
    from pyscf.dft import gen_grid
    cheap_xc = getattr(mf, 'two_level_grids_level', None)
    cheap_nlc = getattr(mf, 'two_level_nlcgrids_level', None)
    specs = [('XC (99,590)', mf.grids), ('VV10 (50,194)', mf.nlcgrids),
             ('response XC, level 2', mf.second_grids), ('response VV10, level 1', mf.second_nlcgrids)]
    if cheap_xc is not None:
        g = gen_grid.Grids(mol); g.level = cheap_xc; specs.append((f'cheap model XC, level {cheap_xc}', g))
    if cheap_nlc is not None:
        g = gen_grid.Grids(mol); g.level = cheap_nlc; specs.append((f'cheap model VV10, level {cheap_nlc}', g))
    for label, g in specs:
        t = time.perf_counter()
        g.build(with_non0tab=True)
        rho = ni.get_rho(mol, dmt, g, mf.max_memory)
        n = g.weights.size
        line = f'{label}: {n} points ({time.perf_counter() - t:.0f} s)'
        line += f'; rho >= 1e-8 (the VV10 kernels\' mask): {numpy.count_nonzero(rho >= 1e-8)}'
        rw = abs(rho * g.weights)
        for x in cutoffs:
            keep = rw > x / n
            line += f'; cutoff {x:g}: {numpy.count_nonzero(keep)} ({numpy.count_nonzero(keep) / n:.3f})'
            if 'VV10' in label:
                line += f', of them rho >= 1e-8: {numpy.count_nonzero(keep & (rho >= 1e-8))}'
        say(line)
    # an XC potential (the SCF's Fock build) on the full grid and on the grid pruned at each cutoff, alternated
    g_full = mf.grids
    rho = ni.get_rho(mol, dmt, g_full, mf.max_memory)
    grids = {'full': g_full}
    for x in cutoffs:
        g = g_full.copy()
        grids[x] = g.prune_by_density_(rho.copy(), x)
    dms = numpy.asarray(dm) if UKS else dm
    order = ['full'] + cutoffs + ['full'] + cutoffs[::-1]
    times = {}
    for key in order:
        g = grids[key]
        t = time.perf_counter()
        n, exc, vxc = (ni.nr_uks if UKS else ni.nr_rks)(mol, g, mf.xc, dms, max_memory=mf.max_memory)
        dt = time.perf_counter() - t
        times.setdefault(key, []).append(dt)
        say(f'XC potential on the grid {key!s:>6} ({g.weights.size} points): {dt:.1f} s, E_xc {exc:.10f}, '
            f'nelec {numpy.sum(n):.8f}')
    for key in grids:
        say(f'XC potential, {key}: mean {numpy.mean(times[key]):.1f} s of {times[key]}')
    # eval_f_t on the VV10 grids of the energy term and of the CPHF responses, with and without the cutoff
    rcuts = [float(x) for x in a.rcuts.split(',')]
    nlc_pars = tuple(ni.nlc_coeff(mf.xc)[0][0])
    for label, g in (('VV10 (50,194)', mf.nlcgrids), ('response VV10, level 1', mf.second_nlcgrids)):
        t = time.perf_counter()
        vv = rks_hess._VV10HessData(mf, mol, dmt, nlc_pars, mf.max_memory, g)
        say(f'{label}: VV10 data of {vv.ngrids} points (ground-state pair sums): {time.perf_counter() - t:.1f} s')
        rng = numpy.random.default_rng(7)
        rho_t = vv.rho * rng.standard_normal((a.ntrial, vv.ngrids))
        gamma_t = vv.rho * rng.standard_normal((a.ntrial, vv.ngrids))
        ref = None
        for rc in [None, 0.] + rcuts + [None]:
            MODE['rcut'] = rc
            KERNEL.clear()
            t = time.perf_counter()
            fr, fg = vv.f_t(rho_t, gamma_t)
            dt = time.perf_counter() - t
            line = f'{label}: eval_f_t of {a.ntrial} densities, ' + ('tree' if rc is None else f'cutoff {rc:g}')
            line += f': {dt:.1f} s'
            rec = list(KERNEL.values())[0]
            if rec[3]:
                line += f', {rec[2] / rec[3]:.3f} of the block pairs'
            if ref is None:
                ref = fr, fg
            else:
                line += (f'; max |d f_rho_t| / max |f_rho_t| {abs(fr - ref[0]).max() / abs(ref[0]).max():.2e}, '
                         f'f_gamma_t {abs(fg - ref[1]).max() / abs(ref[1]).max():.2e}')
            say(line)
        MODE['rcut'] = None
        vv = rho_t = gamma_t = ref = fr = fg = None
    sys.exit()

# ---- the variants at the geometry of XYZ
variants = a.variants.split(',')
out = open(a.json, 'a') if a.json else None
mf_ref = make_mf(mol)
dm0 = None
if a.load:
    z = numpy.load(a.load)
    dm0 = mf_ref.make_rdm1(z['mo_coeff'], z['mo_occ'])
t = time.perf_counter()
mf_ref.kernel(dm0=dm0)
say(f'reference SCF: {time.perf_counter() - t:.1f} s, E = {mf_ref.e_tot:.10f}, converged {mf_ref.converged}; XC grid '
    f'{mf_ref.grids.weights.size} points, VV10 grid {mf_ref.nlcgrids.weights.size}')
if a.save:
    numpy.savez(a.save, mo_coeff=mf_ref.mo_coeff, mo_occ=mf_ref.mo_occ)
t = time.perf_counter()
gobj = mf_ref.Gradients()
gobj.verbose = 2     # not the table of the gradient
g_ref = gobj.kernel()
say(f'reference gradient: {time.perf_counter() - t:.1f} s, max |g| {abs(g_ref).max():.2e}')
dm_ref = mf_ref.make_rdm1()
results = {}


def selftest(vv):
    '''the library's kernels with every pair, in the tree's order and in the k-d order, against the tree's'''
    rng = numpy.random.default_rng(3)
    nt = 6
    rho_t = vv.rho * rng.standard_normal((nt, vv.ngrids))
    gamma_t = vv.rho * rng.standard_normal((nt, vv.ngrids))
    MODE['rcut'] = None
    r0 = vv.f_t(rho_t, gamma_t)
    lib_ = cutlib()
    lib_.VV10CUT_set_rcut(0.)
    fr = numpy.empty((nt, vv.ngrids)); fg = numpy.empty((nt, vv.ngrids))
    args = [numpy.ascontiguousarray(getattr(vv, k)) for k in F_T_KEYS]
    lib_.VV10CUT_eval_f_t(_ptr(fr), _ptr(fg), *[_ptr(x) for x in args], _ptr(rho_t), _ptr(gamma_t),
                          ctypes.c_int(vv.ngrids), ctypes.c_int(nt))
    same = numpy.array_equal(fr, r0[0]) and numpy.array_equal(fg, r0[1])
    MODE['rcut'] = 0.
    r1 = vv.f_t(rho_t, gamma_t)
    d = max(abs(r1[0] - r0[0]).max() / abs(r0[0]).max(), abs(r1[1] - r0[1]).max() / abs(r0[1]).max())
    dw = vv.rho * rng.standard_normal((5, vv.ngrids))
    MODE['rcut'] = None
    w0 = vv.weight1_pair(dw)
    MODE['rcut'] = 0.
    w1 = vv.weight1_pair(dw)
    dwd = max(abs(x - y).max() / abs(x).max() for x, y in zip(w0, w1))
    MODE['rcut'] = None
    KERNEL.clear()
    say(f'self-test on {vv.ngrids} VV10 points: eval_f_t of the library in the tree\'s order bitwise equal: {same}; '
        f'in the k-d order, relative to the largest element: f_t {d:.1e}, weight1 {dwd:.1e}')


for var in variants:
    kind, _, val = var.partition(':')
    TIMES.clear()
    KERNEL.clear()
    MODE['rcut'] = None
    mf = mf_ref
    t = time.perf_counter()
    if kind in ('rho', 'rhoall'):
        mf = make_mf(mol, float(val))
        mf.kernel(dm0=dm_ref)
        t_scf = time.perf_counter() - t
        t = time.perf_counter()
        gobj = mf.Gradients()
        gobj.verbose = 2
        g = gobj.kernel()
        t_grad = time.perf_counter() - t
        say(f'{var}: SCF {t_scf:.1f} s, E - E_ref {mf.e_tot - mf_ref.e_tot:+.3e}, gradient {t_grad:.1f} s, max |dg| '
            f'{abs(g - g_ref).max():.2e}; XC grid {mf.grids.weights.size} points (ref {mf_ref.grids.weights.size}), '
            f'VV10 grid {mf.nlcgrids.weights.size} (ref {mf_ref.nlcgrids.weights.size})')
        if kind == 'rhoall':
            dmt = total_dm(mf.make_rdm1())
            for key in ('second_grids', 'second_nlcgrids'):
                gr = getattr(mf, key)
                if gr.coords is None:
                    gr.build(with_non0tab=True)
                n0 = gr.weights.size
                gr.prune_by_density_(mf._numint.get_rho(mol, dmt, gr, mf.max_memory), float(val))
                say(f'{var}: {key} {gr.weights.size} points of {n0}')
    elif kind == 'vv10':
        if not results.get('_selftest'):
            selftest(rks_hess._vv10_hess_data(mf, mol, total_dm(dm_ref), mf.max_memory))
            results['_selftest'] = True
        MODE['rcut'] = float(val)
    elif kind not in ('ref', 'ref2'):
        raise ValueError(var)
    hobj = mf.Hessian()
    hobj.grid_response = True
    hobj.sgx_response = True
    hobj.sgx_response_auxbasis = 'aug_etb'
    t = time.perf_counter()
    h = hobj.kernel()
    t_hess = time.perf_counter() - t
    MODE['rcut'] = None
    ha = thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    freq = numpy.asarray(ha['freq_wavenumber']).real
    G, _ = free_energy(mf, ha['freq_au'])
    say(f'{var}: Hessian {t_hess:.1f} s; lowest frequencies (cm-1) ' + ' '.join(f'{x:.2f}' for x in freq[:6]) +
        f'; G {G:.10f} Eh')
    table(f'{var}: parts of the Hessian (wall time, nested parts overlap):')
    kernel_table()
    rec = {'variant': var, 'e_tot': mf.e_tot, 'hessian_s': t_hess, 'freq': freq.tolist(), 'G': G,
           'xc_points': int(mf.grids.weights.size), 'nlc_points': int(mf.nlcgrids.weights.size),
           'parts': {k: v[1] for k, v in TIMES.items()}, 'kernels': {k: v for k, v in KERNEL.items()},
           'hess': h.tolist()}
    results[var] = rec
    if out:
        out.write(json.dumps(rec) + '\n')
        out.flush()
    if mf is not mf_ref:
        mf = None

# ---- against ref
if 'ref' in results:
    r0 = results['ref']
    f0 = numpy.array(r0['freq'])
    h0 = numpy.array(r0['hess'])
    say('against ref: E (Eh), the largest change of a frequency (cm-1) and of the lowest three, of a Hessian element '
        '(Eh/Bohr^2), of G (Eh and kcal/mol), and the Hessian\'s time')
    for var, r in results.items():
        if var.startswith('_'):
            continue
        f = numpy.array(r['freq'])
        dh = abs(numpy.array(r['hess']) - h0).max()
        dG = r['G'] - r0['G']
        print(f'   {var:14s} dE {r["e_tot"] - r0["e_tot"]:+.2e}  max|df| {abs(f - f0).max():.4f}  lowest '
              + ' '.join(f'{x:+.4f}' for x in (f - f0)[:3]) + f'  max|dH| {dh:.1e}  dG {dG:+.2e} '
              f'({dG * nist.HARTREE2J * nist.AVOGADRO / 4184:+.4f} kcal/mol)  Hessian {r["hessian_s"]:.1f} s',
              flush=True)
