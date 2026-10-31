"""Gate and timing of the opt-in one-pass J+K (mf.jk_one_pass in the gpu4pyscf perf tree) on node1. For each case
the SCF runs with jk_one_pass off and on, interleaved (off, on, off, on, ...); it prints both energies, the cycles
and the median wall times. --tree wheel runs the unmodified wheel instead (off only), to check that the default
path is unchanged. Source scripts/node1-gpu-env.sh with `perf` for the perf tree, without for the wheel.
Usage: one_pass_gate.py [--cases rks_b3lyp,...] [--reps 2] [--tree perf|wheel] [--tol 1e-10] [--out FILE]"""
import argparse
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import scf, dft  # noqa: E402

# case: (charge, spin, constructor)
CASES = {
    'rks_b3lyp': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp')),
    'rks_pbe0': (0, 0, lambda mol: dft.RKS(mol, xc='pbe0')),
    'uks_b3lyp': (1, 1, lambda mol: dft.UKS(mol, xc='b3lyp')),
    'rhf': (0, 0, lambda mol: scf.RHF(mol)),
    'uhf': (1, 1, lambda mol: scf.UHF(mol)),
    'rks_wb97x': (0, 0, lambda mol: dft.RKS(mol, xc='wb97x')),  # range-separated: builders unchanged
    'rks_pbe': (0, 0, lambda mol: dft.RKS(mol, xc='pbe')),      # pure: no K
}

ap = argparse.ArgumentParser()
ap.add_argument('--cases', default=','.join(CASES))
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--tree', default='perf', choices=['perf', 'wheel'])
ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--out', default=None)
args = ap.parse_args()
in_tree = 'site-packages' not in gpu4pyscf.__file__
assert in_tree == (args.tree == 'perf'), f'--tree {args.tree} but gpu4pyscf is {gpu4pyscf.__file__}'
print(f'gpu4pyscf {gpu4pyscf.__version__} from {os.path.dirname(gpu4pyscf.__file__)}', flush=True)


def run(case, one_pass):
    charge, spin, method = CASES[case]
    mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=charge, spin=spin, verbose=0)
    mf = method(mol)
    mf.conv_tol = args.tol
    if args.tree == 'perf':
        mf.jk_one_pass = one_pass
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    mf.kernel()
    cupy.cuda.Device().synchronize()
    r = {'case': case, 'tree': args.tree, 'jk_one_pass': one_pass, 'e_tot': float(mf.e_tot),
         'cycles': int(mf.cycles), 'converged': bool(mf.converged), 'wall_s': round(time.perf_counter() - t0, 2)}
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
    return r


for case in args.cases.split(','):
    flags = (False, True) if args.tree == 'perf' else (False,)
    runs = {flag: [] for flag in flags}
    for _ in range(args.reps):
        for flag in flags:
            runs[flag].append(run(case, flag))
    line = f'{case:10s}'
    for flag in flags:
        rs = runs[flag]
        es = [r['e_tot'] for r in rs]
        line += (f"  {'on ' if flag else 'off'} E {es[0]:.10f} (spread {max(es) - min(es):.0e}) "
                 f"cyc {rs[0]['cycles']:3d}{'' if all(r['converged'] for r in rs) else ' NOT CONVERGED'} "
                 f"{numpy.median([r['wall_s'] for r in rs]):7.2f} s")
    if args.tree == 'perf':
        off, on = runs[False], runs[True]
        line += (f"  dE {on[0]['e_tot'] - off[0]['e_tot']:+.1e}  speed-up "
                 f"{numpy.median([r['wall_s'] for r in off]) / numpy.median([r['wall_s'] for r in on]):.3f}x")
    print(line, flush=True)
