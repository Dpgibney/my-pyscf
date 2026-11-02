"""One workload of the overview against unmodified GPU4PySCF (node1): for a molecule, basis and functional, the SCF, the
nuclear gradient, the analytical Hessian and TDDFT (5 singlets), each timed once between device synchronizations from
the same converged SCF, with the GPU clocks (gpu_monitor). A warm-up on water with the same basis, functional and tasks
runs first, so that neither tree is charged for loading its CUDA modules or for first-call setup. The tasks' results
(E, gradient, Hessian, excitation energies) go to PREFIX_*.npy with --save; --compare PREFIX prints and records their
largest differences from another tree's. --df runs it with density fitting (def2-universal-jkfit), --one-pass with
SCF.jk_one_pass = True (opt-in, the current tree only). Appends a JSON line to --out. Run it with PYTHONPATH set to the
tree.
Usage: overview.py --label base|current [--mol caffeine|benzene] [--basis def2-svp] [--xc b3lyp] [--df] [--one-pass]
                   [--tasks scf,grad,hess,tddft] [--save PREFIX] [--compare PREFIX] [--out FILE]"""
import argparse
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
from gpu4pyscf.dft import rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--df', action='store_true')
ap.add_argument('--one-pass', action='store_true')
ap.add_argument('--tasks', default='scf,grad,hess')
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--out', default=None)
args = ap.parse_args()
tasks = args.tasks.split(',')


def as_numpy(x):
    return cupy.asnumpy(x) if isinstance(x, cupy.ndarray) else numpy.asarray(x)


def timed(fn):
    cupy.cuda.Device().synchronize()
    with GpuMonitor() as mon:
        t0 = time.perf_counter()
        out = fn()
        cupy.cuda.Device().synchronize()
        dt = time.perf_counter() - t0
    return dt, out, mon.summary.get('sm_mhz')


def run(atom):
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    mf = rks.RKS(mol, xc=args.xc)
    if args.df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    if args.one_pass:
        mf.jk_one_pass = True
    mf.conv_tol = 1e-10
    times, results, clocks = {}, {}, {}
    times['scf'], e, clocks['scf'] = timed(mf.kernel)
    assert mf.converged
    results['e'] = numpy.array([e])
    if 'grad' in tasks:
        times['grad'], g, clocks['grad'] = timed(lambda: mf.Gradients().kernel())
        results['grad'] = as_numpy(g)
    if 'hess' in tasks:
        times['hess'], h, clocks['hess'] = timed(lambda: mf.Hessian().kernel())
        results['hess'] = as_numpy(h)
    if 'tddft' in tasks:
        td = mf.TDDFT()
        td.nstates = 5
        # GPU4PySCF sizes the TDDFT trial space from max_memory (MB). The default, 4000, asks for about 7 GB of GPU
        # memory for caffeine/def2-SVP (four nocc*nvir x max_space holders and four (2 max_space)^2 matrices), more
        # than the GTX 1650 Super has; 1000 still allows over 1500 trial vectors.
        td.max_memory = 1000
        times['tddft'], out, clocks['tddft'] = timed(td.kernel)
        results['tddft'] = as_numpy(out[0])
    return mf, times, results, clocks


run(WATER)  # warm-up
mf, times, results, clocks = run({'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol])
record = {'label': args.label, 'tree': os.path.dirname(gpu4pyscf.__file__), 'mol': args.mol, 'basis': args.basis,
          'xc': args.xc, 'df': args.df, 'one_pass': args.one_pass, 'nao': mf.mol.nao, 'E': float(results['e'][0]), 'cycles': mf.cycles,
          'times': {k: round(v, 2) for k, v in times.items()}, 'sm_mhz': clocks}
if args.save:
    for k, v in results.items():
        numpy.save(f'{args.save}_{k}.npy', v)
if args.compare:
    record['max_diff'] = {}
    for k, v in results.items():
        ref = numpy.load(f'{args.compare}_{k}.npy')
        record['max_diff'][k] = float(abs(v - ref).max())
        record.setdefault('max_ref', {})[k] = float(abs(ref).max())
print(json.dumps(record), flush=True)
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
