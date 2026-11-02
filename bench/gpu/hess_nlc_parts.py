"""Where a VV10 functional's analytical RKS Hessian spends its time (node1): runs the SCF (timed only), then the
Hessian once (--task hessian) with synchronizing timers around its steps (partial_hess_elec, make_h1, solve_mo1 and its response
get_veff_resp_mo), the VV10 parts inside them (_get_enlc_deriv2, _get_vnlc_deriv1, nr_rks_fnlc_mo once per CPHF
iteration) and every libgdft
VV10 entry those call, with the entry's trailing int arguments (ngrids, then ntrial or natm where it has them).
Synchronizing after every call removes the overlap of launches, so the totals run a little above an untimed run.
--grid-response turns on the Hessian's grid response. Writes the Hessian to --save; --compare REF.npy prints the
largest difference. --memcheck (for compute-sanitizer) turns CuPy's pool off and makes the grids small: (30, 110) for
the XC part, (20, 50) for VV10. --task tddft runs TDDFT for --nstates singlets instead (with VV10's response,
exclude_nlc = False), and --task polar the static polarizability, with the same timers on the VV10 response
(get_nlc_response_ground_state and nr_rks_fnlc_mo, part 30); they save the excitation energies or the polarizability.
Source scripts/node1-gpu-env.sh first.
Usage: hess_nlc_parts.py [--task hessian|tddft|polar] [--nstates 5] [--xc b97m_v] [--basis def2-svp]
                         [--mol benzene|caffeine|water] [--grid-response] [--save H.npy] [--compare REF.npy] [--memcheck]"""
import argparse
import ctypes
import os
import sys
import time
from collections import defaultdict
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--task', default='hessian', choices=['hessian', 'tddft', 'polar'])
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--xc', default='b97m_v')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='benzene', choices=['caffeine', 'benzene', 'water'])
ap.add_argument('--grid-response', action='store_true')
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--memcheck', action='store_true')
args = ap.parse_args()


def sync():
    cupy.cuda.Device().synchronize()


times = {}  # by call path, in the order the calls start
counts = defaultdict(int)
int_args = defaultdict(set)
stack = []


def timed(name, fn):
    def wrapper(*a, **k):
        key = '/'.join(stack + [name])
        times.setdefault(key, 0.)
        stack.append(name)
        sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sync()
            times[key] += time.perf_counter() - t0
            counts[key] += 1
            stack.pop()
            if name.startswith('VXC_'):
                int_args[key].add(tuple(x.value for x in a if isinstance(x, ctypes.c_int)))
    return wrapper


class TimedLib:
    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        return timed(name, fn) if name.startswith('VXC_vv10') else fn


if args.memcheck:
    cupy.cuda.set_allocator(cupy.cuda.memory._malloc)
mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE, 'water': WATER}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
if args.memcheck:
    mf.grids.atom_grid = (30, 110)
    mf.nlcgrids.atom_grid = (20, 50)
mf.conv_tol = 1e-10
sync()
t0 = time.perf_counter()
e = mf.kernel()
sync()
t_scf = time.perf_counter() - t0
assert mf.converged

for fn in ('_get_enlc_deriv2', '_get_vnlc_deriv1', 'nr_rks_fnlc_mo', 'get_nlc_response_ground_state'):
    if hasattr(rks_hess, fn):
        setattr(rks_hess, fn, timed(fn, getattr(rks_hess, fn)))
rks_hess.libgdft = TimedLib(rks_hess.libgdft)
if args.task == 'hessian':
    hess = mf.Hessian()
    hess.grid_response = args.grid_response
    for step in ('partial_hess_elec', 'make_h1', 'solve_mo1', 'get_veff_resp_mo'):
        setattr(hess, step, timed(step, getattr(hess, step)))
    run = hess.kernel
elif args.task == 'tddft':
    td = mf.TDDFT()
    td.nstates = args.nstates
    td.exclude_nlc = False  # with VV10's response, which gpu4pyscf's TDDFT leaves out by default
    run = lambda: td.kernel()[0]  # noqa: E731
else:
    from gpu4pyscf.properties import polarizability
    run = lambda: polarizability.eval_polarizability(mf)  # noqa: E731
sync()
t0 = time.perf_counter()
h = run()
sync()
t_task = time.perf_counter() - t0
h = cupy.asnumpy(h) if isinstance(h, cupy.ndarray) else numpy.asarray(h)

task = {'hessian': f'Hessian (grid response {args.grid_response})', 'tddft': f'TDDFT ({args.nstates} states)',
        'polar': 'polarizability'}[args.task]
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.mol} {args.basis} {args.xc}, '
      f'{mf.nlcgrids.weights.size} NLC grid points: E {e:.10f}, SCF {mf.cycles} cycles {t_scf:.1f} s, '
      f'{task} {t_task:.1f} s')
if args.task != 'hessian':
    print(f'  result: {numpy.array2string(h.ravel(), precision=10, max_line_width=110)}')
for key, t in times.items():
    depth = key.count('/')
    shapes = ' '.join(str(s) for s in sorted(int_args[key])) if key in int_args else ''
    print(f'  {"  " * depth}{key.rsplit("/", 1)[-1]:<{56 - 2 * depth}} {t:9.2f} s  x{counts[key]:<4} {shapes}')
if args.save:
    numpy.save(args.save, h)
if args.compare:
    ref = numpy.load(args.compare)
    print(f'  against {os.path.basename(args.compare)}: max |difference| {abs(h - ref).max():.2e} '
          f'(max |value| {abs(ref).max():.2e})')
