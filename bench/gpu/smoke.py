"""GPU4PySCF smoke test (node1): one RKS calculation done by PySCF on the CPU and by GPU4PySCF on the GPU, with the
same molecule, basis, functional and convergence threshold. Prints the versions and the device, then for each run
the energy, cycles, grid size and wall time. The GPU run is done twice; the second excludes CUDA start-up. Logs
(verbose 4) go to LOGDIR, one per run.
Usage: smoke.py [--mol caffeine|benzene] [--basis def2-svp] [--xc b3lyp] [--tol 1e-10] [--no-cpu]
                [--logdir .] [--out results.jsonl]"""
import argparse
import json
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks as gpu_rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--no-cpu', action='store_true')
ap.add_argument('--logdir', default='.')
ap.add_argument('--out', default=None)
args = ap.parse_args()

rt = cupy.cuda.runtime
p = rt.getDeviceProperties(0)
try:
    from cupy_backends.cuda.libs import cutensor
    cutensor_version = cutensor.get_version()
except Exception as e:
    cutensor_version = f'unavailable ({e.__class__.__name__}: {e})'
env = {
    'pyscf': pyscf.__version__, 'gpu4pyscf': gpu4pyscf.__version__, 'cupy': cupy.__version__,
    'cutensor': cutensor_version, 'cuda_runtime': rt.runtimeGetVersion(), 'cuda_driver': rt.driverGetVersion(),
    'device': p['name'].decode(), 'cc': f"{p['major']}.{p['minor']}", 'sms': p['multiProcessorCount'],
    'mem_gib': round(p['totalGlobalMem'] / 2**30, 2), 'shm_per_block_optin': p['sharedMemPerBlockOptin'],
    'kernel_timeout': bool(p['kernelExecTimeoutEnabled']), 'cpu_threads': lib.num_threads(),
}
for k, v in env.items():
    print(f'{k:20s} {v}')

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
print(f'{args.mol} {args.basis} {args.xc}: nao {mol.nao}, conv_tol {args.tol:g}', flush=True)


def run(label, mf):
    log = os.path.join(args.logdir, f'smoke_{args.mol}_{args.xc}_{label}.log')
    with open(log, 'w') as f:
        mf.verbose = 4
        mf.stdout = f
        mf.conv_tol = args.tol
        t0 = time.perf_counter()
        mf.kernel()
        cupy.cuda.Device().synchronize()
        wall = time.perf_counter() - t0
    cycles = getattr(mf, 'cycles', None)
    if cycles is None:
        cycles = sum(line.startswith('cycle=') for line in open(log))
    r = {'run': label, 'e_tot': float(mf.e_tot), 'converged': bool(mf.converged), 'cycles': int(cycles),
         'ngrids': int(mf.grids.weights.size), 'wall_s': round(wall, 2)}
    print(f"{label:6s} E = {r['e_tot']:.10f}  converged {r['converged']}  cycles {r['cycles']:3d}  "
          f"grids {r['ngrids']}  {r['wall_s']:7.2f} s", flush=True)
    return r


runs = []
if not args.no_cpu:
    runs.append(run('cpu', dft.RKS(mol, xc=args.xc)))
for label in ('gpu1', 'gpu2'):
    runs.append(run(label, gpu_rks.RKS(mol, xc=args.xc)))
if not args.no_cpu:
    for r in runs[1:]:
        print(f"{r['run']} - cpu: dE = {r['e_tot'] - runs[0]['e_tot']:.1e}")
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps({'mol': args.mol, 'basis': args.basis, 'xc': args.xc, 'tol': args.tol,
                            'env': env, 'runs': runs}) + '\n')
