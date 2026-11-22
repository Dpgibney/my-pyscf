"""Where a production-like GPU4PySCF job spends its time (node1): wB97M-V with the XC grid (99,590) and the VV10 grid
(50,194), in SMD water, the SCF and then the nuclear gradient. Every part named below is timed between device
synchronizations; the table gives its calls, its inclusive time and its exclusive time (inclusive minus the timed parts
it called). A warm-up on water with the same settings runs first, untimed, so that loading the CUDA modules is not
charged to the job. Run it with PYTHONPATH set to the tree (source scripts/node1-gpu-env.sh perf).

--set NAME=VALUE sets an attribute of the SCF object before it runs (dotted names reach into it:
--set nlcgrids.level=1), --gset NAME=VALUE one of the gradient object. --save PREFIX writes the energy and the gradient
to PREFIX.npz, --compare PREFIX prints their largest differences from another run's. --out FILE appends a JSON line.
Usage: prod_parts.py [--mol caffeine|benzene|water|FILE.xyz] [--charge 0] [--basis def2-svp] [--xc wb97m-v] [--grid 99,590]
                     [--nlcgrid 50,194] [--solvent smd|cpcm|iefpcm|none] [--tasks scf,grad] [--conv-tol 1e-10]
                     [--set NAME=VALUE ...] [--gset NAME=VALUE ...] [--label L] [--save PREFIX] [--compare PREFIX]
                     [--out FILE] [--verbose N]"""
import argparse
import ast
import collections
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks, numint  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402
from gpu4pyscf.solvent.grad import pcm as pcm_grad, smd as smd_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', help='caffeine, benzene, water, or an xyz file')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlcgrid', default='50,194')
ap.add_argument('--solvent', default='smd', choices=['smd', 'cpcm', 'iefpcm', 'none'])
ap.add_argument('--tasks', default='scf,grad')
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--gset', action='append', default=[])
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


def make_mf(atom, charge=0):
    mol = gto.M(atom=atom, basis=args.basis, charge=charge, verbose=args.verbose)
    mf = rks.RKS(mol, xc=args.xc)
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
    for name in ('nr_rks', 'nr_nlc_vxc'):
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
        for name in ('grad_qv', 'grad_solver', 'grad_nuc'):
            parts.wrap(pcm_grad, name, 'solvent.' + name)
        parts.wrap(smd_grad, 'get_cds', 'solvent.grad_cds')


def report(task, out):
    """The table of one task, printed as soon as it ends (a later task may run out of memory)."""
    note = '  FAILED: ' + out['failed'][task] if task in out.get('failed', {}) else ''
    print(f"--- {task}: {out['times'][task]:.2f} s, SM clock {out['sm_mhz'][task]}{note}")
    print(f"    {'part':34s} {'calls':>5s} {'incl s':>9s} {'excl s':>9s}")
    for label, (n, incl, excl) in out['parts'][task].items():
        print(f"    {label:34s} {n:5d} {incl:9.2f} {excl:9.2f}")
    top = sum(excl for _, _, excl in out['parts'][task].values())
    print(f"    {'(not in any timed part)':34s} {'':5s} {'':9s} {out['times'][task] - top:9.2f}", flush=True)


def run(atom, timed, charge=0):
    mf = make_mf(atom, charge)
    g = None
    if 'grad' in tasks:
        g = mf.Gradients()
        for spec in args.gset:
            set_attr(g, spec)
    if timed:
        instrument(mf, g)
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
            sync()
            out['times']['grad'] = time.perf_counter() - t0
        out['sm_mhz']['grad'] = mon.summary.get('sm_mhz')
        out['parts']['grad'] = parts.take()
        if timed:
            print(f"pool peak {parts.pool_peak / 2**20:.0f} MiB")
            report('grad', out)
    parts.on = False
    return mf, out


if not args.no_warmup:
    run(WATER, False)
mf, out = run(MOLS.get(args.mol, args.mol), True, args.charge)
ngrids, nlc, nsurf = out['ngrids'], out['nlcgrids'], out['nsurf']

record = {'label': args.label, 'tree': os.path.dirname(gpu4pyscf.__file__), 'mol': args.mol, 'basis': args.basis,
          'xc': args.xc, 'solvent': args.solvent, 'grid': args.grid, 'nlcgrid': args.nlcgrid, 'nao': int(mf.mol.nao),
          'ngrids': ngrids, 'nlcgrids': nlc, 'nsurf': nsurf, 'set': args.set, 'gset': args.gset, 'E': out['e'],
          'converged': out['converged'], 'cycles': out['cycles'], 'failed': out['failed'],
          'times': {k: round(v, 3) for k, v in out['times'].items()},
          'parts': {t: {k: [v[0], round(v[1], 3), round(v[2], 3)] for k, v in p.items()}
                    for t, p in out['parts'].items()},
          'sm_mhz': out['sm_mhz'], 'pool_peak_mib': round(parts.pool_peak / 2**20)}
if args.save:
    numpy.savez(args.save + '.npz', e=out['e'], grad=out.get('grad', numpy.zeros(0)))
if args.compare:
    ref = numpy.load(args.compare + '.npz')
    record['dE'] = out['e'] - float(ref['e'])
    msg = f"vs {args.compare}: dE = {record['dE']:.3e}"
    if 'grad' in out and ref['grad'].size:
        record['max_dgrad'] = float(abs(out['grad'] - ref['grad']).max())
        msg += f", max |dgrad| = {record['max_dgrad']:.3e} (max |grad| {abs(ref['grad']).max():.3e})"
    print(msg)
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
