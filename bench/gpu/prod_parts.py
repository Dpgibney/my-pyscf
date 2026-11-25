"""Where a production-like GPU4PySCF job spends its time (node1): wB97M-V with the XC grid (99,590) and the VV10 grid
(50,194), in SMD water, the SCF and then the nuclear gradient and/or the analytical Hessian. Every part named below is
timed between device synchronizations; the table gives its calls, its inclusive time and its exclusive time (inclusive
minus the timed parts it called). A warm-up on water with the same settings runs first, untimed, so that loading the
CUDA modules is not charged to the job. Run it with PYTHONPATH set to the tree (source scripts/node1-gpu-env.sh perf).

--set NAME=VALUE sets an attribute of the SCF object before it runs (dotted names reach into it:
--set nlcgrids.level=1), --gset NAME=VALUE one of the gradient object, --hset NAME=VALUE one of the Hessian object.
--save PREFIX writes the energy, the gradient and the Hessian to PREFIX.npz, --compare PREFIX prints their largest
differences from another run's (for the Hessian also that of the harmonic frequencies). --out FILE appends a JSON line.
--spin N (2S, not zero) runs the unrestricted model: UKS, its gradient and its Hessian.
Usage: prod_parts.py [--mol caffeine|benzene|water|FILE.xyz] [--charge 0] [--spin 0] [--basis def2-svp] [--xc wb97m-v] [--grid 99,590]
                     [--nlcgrid 50,194] [--solvent smd|cpcm|iefpcm|none] [--tasks scf,grad,hess] [--conv-tol 1e-10]
                     [--set NAME=VALUE ...] [--gset NAME=VALUE ...] [--hset NAME=VALUE ...] [--label L] [--save PREFIX]
                     [--compare PREFIX] [--out FILE] [--verbose N]"""
import argparse
import ast
import collections
import json
import os
import sys
import time
import traceback
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks, uks, numint  # noqa: E402
from gpu4pyscf.grad import rhf as rhf_grad, rks as rks_grad, uks as uks_grad  # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess, rks as rks_hess, uhf as uhf_hess, uks as uks_hess  # noqa: E402
try:
    from gpu4pyscf.sgx import sgx as sgx_mod  # noqa: E402
except ImportError:   # a tree before part 37 (v1.9.0)
    sgx_mod = None
from gpu4pyscf.solvent.grad import pcm as pcm_grad, smd as smd_grad  # noqa: E402
from gpu4pyscf.solvent.hessian import pcm as pcm_hess, smd as smd_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', help='caffeine, benzene, water, or an xyz file')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0, help='2S; not zero: UKS')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlcgrid', default='50,194')
ap.add_argument('--solvent', default='smd', choices=['smd', 'cpcm', 'iefpcm', 'none'])
ap.add_argument('--tasks', default='scf,grad')
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--gset', action='append', default=[])
ap.add_argument('--hset', action='append', default=[])
ap.add_argument('--label', default='')
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--out', default=None)
ap.add_argument('--verbose', type=int, default=0)
ap.add_argument('--no-warmup', action='store_true')
args = ap.parse_args()
tasks = args.tasks.split(',')
MOLS = {'caffeine': CAFFEINE, 'benzene': BENZENE, 'water': WATER}


def sync():
    cupy.cuda.Device().synchronize()


class Parts:
    """Inclusive and exclusive wall times of wrapped callables, each bracketed by device synchronizations."""

    def __init__(self):
        self.rows = collections.OrderedDict()   # label -> [calls, inclusive, exclusive]
        self.stack = []
        self.pool_peak = 0
        self.on = False

    def wrap(self, owner, name, label=None, keyfn=None):
        if not hasattr(owner, name):
            return
        orig = getattr(owner, name)
        base = label or name
        parts = self

        def timed(*a, **kw):
            if not parts.on:
                return orig(*a, **kw)
            label = base + keyfn(*a, **kw) if keyfn else base
            sync()
            parts.stack.append([label, 0.0])
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                sync()
                dt = time.perf_counter() - t0
                _, inner = parts.stack.pop()
                row = parts.rows.setdefault(label, [0, 0.0, 0.0])
                row[0] += 1
                row[1] += dt
                row[2] += dt - inner
                if parts.stack:
                    parts.stack[-1][1] += dt
                parts.pool_peak = max(parts.pool_peak, cupy.get_default_memory_pool().total_bytes())
        timed.__name__ = getattr(orig, '__name__', name)
        setattr(owner, name, timed)

    def take(self):
        rows, self.rows = self.rows, collections.OrderedDict()
        return rows


parts = Parts()


def set_attr(obj, spec):
    name, value = spec.split('=', 1)
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    *path, last = name.split('.')
    for p in path:
        obj = getattr(obj, p)
    setattr(obj, last, value)


def make_mf(atom, charge=0, spin=0):
    mol = gto.M(atom=atom, basis=args.basis, charge=charge, spin=spin, verbose=args.verbose)
    mf = (uks.UKS if spin else rks.RKS)(mol, xc=args.xc)
    mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
    if args.nlcgrid != 'none':
        mf.nlcgrids.atom_grid = tuple(int(x) for x in args.nlcgrid.split(','))
    if args.solvent == 'smd':
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    elif args.solvent != 'none':
        mf = mf.PCM()
        mf.with_solvent.method = {'cpcm': 'C-PCM', 'iefpcm': 'IEF-PCM'}[args.solvent]
        mf.with_solvent.eps = 78.3553
    mf.conv_tol = args.conv_tol
    for spec in args.set:
        set_attr(mf, spec)
    return mf


def instrument(mf, g):
    for name in ('get_veff', 'get_j', 'get_k', 'get_jk', 'get_hcore', 'get_ovlp', 'eig', 'get_fock',
                 'get_init_guess'):
        parts.wrap(type(mf), name, 'scf.' + name)
    # by grid size: a two-level SCF evaluates these on the coarse grids and on the model's
    for name in ('nr_rks', 'nr_uks', 'nr_nlc_vxc'):
        parts.wrap(type(mf._numint), name, 'numint.' + name,
                   keyfn=lambda ni, mol, grids, *a, **kw: '[%d]' % grids.size)
    parts.wrap(numint, '_vv10nlc', 'numint._vv10nlc')
    try:   # the seminumerical K of a two-level SCF's cheap model (mf.two_level_sgx_level)
        from gpu4pyscf.sgx import sgx, sgx_jk
        parts.wrap(sgx.SGX, 'build', 'sgx.build')
        parts.wrap(sgx.SGX, 'get_k', 'sgx.get_k')
        parts.wrap(sgx_jk, 'contract', 'sgx.contract')
    except ImportError:
        pass
    if getattr(mf, 'with_solvent', None) is not None:
        for name in ('build', '_get_vind', '_get_v', '_get_vmat', 'left_solve_K', 'left_multiply_R'):
            parts.wrap(type(mf.with_solvent), name, 'solvent.' + name)
    if g is not None:
        for name in ('get_veff', 'jk_energy_per_atom', 'get_hcore', 'hcore_generator', 'get_ovlp', 'grad_nuc'):
            parts.wrap(type(g), name, 'grad.' + name)
        for name in ('get_exc', 'get_exc_full_response', 'get_nlc_exc', 'get_nlc_exc_full_response'):
            parts.wrap(rks_grad, name, 'grad.' + name)
            # the unrestricted gradient's own functions, and the names it imported from grad.rks
            parts.wrap(uks_grad, name, 'grad.' + name)
        for name in ('grad_qv', 'grad_solver', 'grad_nuc'):
            parts.wrap(pcm_grad, name, 'solvent.' + name)
        parts.wrap(smd_grad, 'get_cds', 'solvent.grad_cds')


class _TimedLib:
    """A compiled library whose entries of a given prefix are timed as parts"""

    def __init__(self, lib, prefix, label):
        self._lib, self._prefix, self._label = lib, prefix, label
        self._timed = {}

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        if not name.startswith(self._prefix):
            return fn
        if name not in self._timed:
            holder = type('entry', (), {'fn': staticmethod(fn)})
            parts.wrap(holder, 'fn', self._label + name[len(self._prefix):], _cheap_suffix)
            self._timed[name] = holder.fn
        return self._timed[name]


# whether the response being built is the cheap model's (the two-level CPHF): set where get_veff_resp_mo is entered
_response = {'cheap': False}


def _cheap_suffix(*args, **kwargs):
    return ' (cheap)' if _response['cheap'] else ''


def _veff_resp_suffix(hessobj, *args, **kwargs):
    _response['cheap'] = getattr(hessobj, '_cheap_response', None) is not None
    return _cheap_suffix()


def instrument_hess(mf):
    """The parts of the analytical Hessian: the second derivatives of the energy at fixed orbitals
    (partial_hess_elec), the first-order Fock matrices (make_h1), the CPHF equations (solve_mo1, whose response of a
    set of trial densities is veff_resp_mo), the solvent's terms."""
    for cls in (rks_hess.Hessian, uks_hess.Hessian):
        for name, label in (('hess_elec', 'hess.hess_elec'), ('partial_hess_elec', 'hess.partial_hess_elec'),
                            ('make_h1', 'hess.make_h1')):
            parts.wrap(cls, name, label)
        # the response of a set of trial densities: the model's own, and the cheap model's of the two-level CPHF
        parts.wrap(cls, 'get_veff_resp_mo', 'hess.veff_resp_mo', _veff_resp_suffix)
        parts.wrap(cls, 'get_jk_mo', 'hess.jk_mo', _cheap_suffix)
        if 'solve_mo1' in vars(cls):
            parts.wrap(cls, 'solve_mo1', 'hess.solve_mo1')
    parts.wrap(rhf_hess.HessianBase, 'solve_mo1', 'hess.solve_mo1')
    for mod in (rhf_hess, uhf_hess):
        for name in ('_partial_ejk_ip2', '_hcore_energy', '_get_jk_ip1', 'krylov', '_cheap_start_and_trials'):
            parts.wrap(mod, name, 'hess.' + name.lstrip('_'))
    # (the unrestricted module has functions of its own under these names, and holds those it imported from
    # hessian.rks under their names)
    for mod in (rks_hess, uks_hess):
        for name in ('_get_exc_deriv2', '_get_enlc_deriv2', '_get_vxc_deriv1', '_get_vnlc_deriv1'):
            parts.wrap(mod, name, 'hess.' + name.lstrip('_'))
        for name in ('nr_rks_fxc_mo', 'nr_uks_fxc_mo', 'nr_rks_fnlc_mo', 'nr_uks_fnlc_mo',
                     'get_nlc_response_ground_state'):
            parts.wrap(mod, name, 'hess.' + name.lstrip('_'), _cheap_suffix)
    # the unrestricted XC response from AO density matrices
    parts.wrap(type(mf._numint), 'nr_uks_fxc', 'numint.nr_uks_fxc', _cheap_suffix)
    if sgx_mod is not None:
        parts.wrap(sgx_mod.SGX, 'get_k', 'sgx.get_k', _cheap_suffix)
    parts.wrap(rhf_grad, 'get_grad_hcore', 'hess.grad_hcore')
    # the compiled VV10 kernels of the Hessian (each N_grid^2 pairs; f_t once per set of trial densities)
    if hasattr(rks_hess, 'libgdft') and not isinstance(rks_hess.libgdft, _TimedLib):
        rks_hess.libgdft = _TimedLib(rks_hess.libgdft, 'VXC_vv10nlc_hess_eval_', 'vv10.')
    with_solvent = getattr(mf, 'with_solvent', None)
    if with_solvent is not None:
        parts.wrap(type(with_solvent), 'hess', 'solvent.hess')
        parts.wrap(type(with_solvent), '_B_dot_x', 'solvent._B_dot_x', _cheap_suffix)
        # the same response from the integrals with the orbital products, and making those
        parts.wrap(pcm_hess, '_response_from_mo_integrals', 'solvent.response_mo', _cheap_suffix)
        parts.wrap(pcm_hess, 'surface_mo_integrals', 'solvent.mo_integrals')
        parts.wrap(pcm_hess, 'get_dqsym_dx', 'solvent.get_dqsym_dx')
        for name in ('analytical_hess_nuc', 'analytical_hess_qv', 'analytical_hess_solver', 'analytical_grad_vmat'):
            parts.wrap(pcm_hess, name, 'solvent.' + name)
        parts.wrap(smd_hess, 'get_cds', 'solvent.hess_cds')


def report(task, out):
    """The table of one task, printed as soon as it ends (a later task may run out of memory)."""
    note = '  FAILED: ' + out['failed'][task] if task in out.get('failed', {}) else ''
    print(f"--- {task}: {out['times'][task]:.2f} s, SM clock {out['sm_mhz'][task]}{note}")
    print(f"    {'part':34s} {'calls':>5s} {'incl s':>9s} {'excl s':>9s}")
    for label, (n, incl, excl) in out['parts'][task].items():
        print(f"    {label:34s} {n:5d} {incl:9.2f} {excl:9.2f}")
    top = sum(excl for _, _, excl in out['parts'][task].values())
    print(f"    {'(not in any timed part)':34s} {'':5s} {'':9s} {out['times'][task] - top:9.2f}", flush=True)


def run(atom, timed, charge=0, spin=0):
    mf = make_mf(atom, charge, spin)
    g = None
    if 'grad' in tasks:
        g = mf.Gradients()
        for spec in args.gset:
            set_attr(g, spec)
    h = None
    if 'hess' in tasks:
        h = mf.Hessian()
        for spec in args.hset:
            set_attr(h, spec)
    if timed:
        instrument(mf, g)
        if h is not None:
            instrument_hess(mf)
    out = {'times': {}, 'parts': {}, 'sm_mhz': {}, 'failed': {}}
    parts.on = timed
    sync()
    with GpuMonitor() as mon:
        t0 = time.perf_counter()
        e = mf.kernel()
        sync()
        out['times']['scf'] = time.perf_counter() - t0
    out['sm_mhz']['scf'] = mon.summary.get('sm_mhz')
    out['parts']['scf'] = parts.take()
    out['e'] = float(e)
    out['converged'] = bool(mf.converged)
    out['cycles'] = int(getattr(mf, 'cycles', -1))
    if timed:
        ngrids = int(mf.grids.coords.shape[0])
        nlc = int(mf.nlcgrids.coords.shape[0]) if mf.nlcgrids.coords is not None else 0
        with_solvent = getattr(mf, 'with_solvent', None)
        nsurf = int(with_solvent.surface['grid_coords'].shape[0]) if with_solvent is not None else 0
        out.update(ngrids=ngrids, nlcgrids=nlc, nsurf=nsurf)
        print(f"{args.label or 'run'}: {args.mol} {args.basis} {args.xc} solvent={args.solvent} nao={mf.mol.nao} "
              f"grid={ngrids} nlcgrid={nlc} surface={nsurf} tree={os.path.dirname(gpu4pyscf.__file__)}")
        print(f"E = {out['e']:.12f}  converged={out['converged']}  cycles={out['cycles']}  "
              f"pool peak {parts.pool_peak / 2**20:.0f} MiB")
        report('scf', out)
    if g is not None:
        with GpuMonitor() as mon:
            t0 = time.perf_counter()
            try:
                de = g.kernel()
                out['grad'] = cupy.asnumpy(de) if isinstance(de, cupy.ndarray) else numpy.asarray(de)
            except Exception as err:   # e.g. out of memory: keep what was timed until then
                if not timed:
                    raise
                parts.stack.clear()
                out['failed']['grad'] = f'{type(err).__name__}: {str(err)[:80]}'
                traceback.print_exc(file=sys.stdout)
            sync()
            out['times']['grad'] = time.perf_counter() - t0
        out['sm_mhz']['grad'] = mon.summary.get('sm_mhz')
        out['parts']['grad'] = parts.take()
        if timed:
            print(f"pool peak {parts.pool_peak / 2**20:.0f} MiB")
            report('grad', out)
    if h is not None:
        with GpuMonitor() as mon:
            t0 = time.perf_counter()
            try:
                hess = h.kernel()
                out['hess'] = cupy.asnumpy(hess) if isinstance(hess, cupy.ndarray) else numpy.asarray(hess)
            except Exception as err:   # e.g. out of memory: keep what was timed until then
                if not timed:
                    raise
                parts.stack.clear()
                out['failed']['hess'] = f'{type(err).__name__}: {str(err)[:80]}'
                traceback.print_exc(file=sys.stdout)
            sync()
            out['times']['hess'] = time.perf_counter() - t0
        out['sm_mhz']['hess'] = mon.summary.get('sm_mhz')
        out['parts']['hess'] = parts.take()
        if timed:
            print(f"pool peak {parts.pool_peak / 2**20:.0f} MiB")
            report('hess', out)
    parts.on = False
    return mf, out


if not args.no_warmup:
    # (the water cation for an open shell)
    run(WATER, False, args.spin % 2, args.spin % 2)
mf, out = run(MOLS.get(args.mol, args.mol), True, args.charge, args.spin)
ngrids, nlc, nsurf = out['ngrids'], out['nlcgrids'], out['nsurf']

record = {'label': args.label, 'tree': os.path.dirname(gpu4pyscf.__file__), 'mol': args.mol, 'spin': args.spin,
          'basis': args.basis,
          'xc': args.xc, 'solvent': args.solvent, 'grid': args.grid, 'nlcgrid': args.nlcgrid, 'nao': int(mf.mol.nao),
          'ngrids': ngrids, 'nlcgrids': nlc, 'nsurf': nsurf, 'set': args.set, 'gset': args.gset, 'hset': args.hset,
          'E': out['e'],
          'converged': out['converged'], 'cycles': out['cycles'], 'failed': out['failed'],
          'times': {k: round(v, 3) for k, v in out['times'].items()},
          'parts': {t: {k: [v[0], round(v[1], 3), round(v[2], 3)] for k, v in p.items()}
                    for t, p in out['parts'].items()},
          'sm_mhz': out['sm_mhz'], 'pool_peak_mib': round(parts.pool_peak / 2**20)}


def frequencies(hess):
    """Harmonic wavenumbers (cm^-1, imaginary ones negative) of a Hessian [natm,natm,3,3], rotations and translations
    not projected out: for comparing two Hessians of the same structure."""
    from pyscf.data import nist
    mass = mf.mol.atom_mass_list(isotope_avg=True)
    natm = len(mass)
    h = hess.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
    w = numpy.repeat(mass, 3) ** -.5
    e = numpy.linalg.eigvalsh(h * w[:, None] * w)
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI ** 2)) ** .5 / (2 * numpy.pi)
    return numpy.sign(e) * abs(e) ** .5 * au2hz / nist.LIGHT_SPEED_SI * 1e-2


if args.save:
    numpy.savez(args.save + '.npz', e=out['e'], grad=out.get('grad', numpy.zeros(0)),
                hess=out.get('hess', numpy.zeros(0)))
if args.compare:
    ref = numpy.load(args.compare + '.npz')
    record['dE'] = out['e'] - float(ref['e'])
    msg = f"vs {args.compare}: dE = {record['dE']:.3e}"
    if 'grad' in out and ref['grad'].size:
        record['max_dgrad'] = float(abs(out['grad'] - ref['grad']).max())
        msg += f", max |dgrad| = {record['max_dgrad']:.3e} (max |grad| {abs(ref['grad']).max():.3e})"
    if 'hess' in out and 'hess' in ref and ref['hess'].size:
        record['max_dhess'] = float(abs(out['hess'] - ref['hess']).max())
        f0, f1 = frequencies(ref['hess']), frequencies(out['hess'])
        record['max_dfreq'] = float(abs(f1 - f0).max())
        msg += (f", max |dhess| = {record['max_dhess']:.3e} (max |hess| {abs(ref['hess']).max():.3e}), "
                f"max |dfreq| = {record['max_dfreq']:.3f} cm^-1")
    print(msg)
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
