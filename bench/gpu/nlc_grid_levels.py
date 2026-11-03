"""The cost and the effect of the NLC grid for a VV10 functional (node1): the SCF (and the gradient) of --xc with the
NLC grid at each of --levels (GPU4PySCF's default is level 3, the main grid's) and SG-1 (Q-Chem's default NLC grid,
(50, 194) with SG-1 pruning), timed after a warm-up; prints each setting's NLC points, SCF time, energy and gradient
against the default's. A coarser NLC grid is an approximation, not an exact speedup: this only measures it.
Usage: nlc_grid_levels.py [--mol caffeine] [--basis def2-svp] [--xc wb97m_v] [--levels 3,2,1,0,sg1] [--grad]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks, gen_grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m_v')
ap.add_argument('--levels', default='3,2,1,0,sg1')
ap.add_argument('--grad', action='store_true')
args = ap.parse_args()


def run(atom, level):
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    mf = rks.RKS(mol, xc=args.xc)
    mf.conv_tol = 1e-10
    if level == 'sg1':
        mf.nlcgrids.atom_grid = (50, 194)
        mf.nlcgrids.prune = gen_grid.sg1_prune
    else:
        mf.nlcgrids.level = int(level)
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.Device().synchronize()
    t_scf = time.perf_counter() - t0
    assert mf.converged
    g = t_grad = None
    if args.grad:
        t0 = time.perf_counter()
        g = cupy.asnumpy(cupy.asarray(mf.Gradients().kernel()))
        cupy.cuda.Device().synchronize()
        t_grad = time.perf_counter() - t0
    return mf, e, t_scf, g, t_grad


run(WATER, '3')  # warm-up
atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol]
ref = None
print(f'{args.mol} {args.basis} {args.xc}: the NLC grid against the default (level 3)')
for level in args.levels.split(','):
    mf, e, t_scf, g, t_grad = run(atom, level)
    if ref is None:
        ref = (e, g)
    line = (f'  NLC grid {level:>4s}: {mf.nlcgrids.weights.size:7d} points, SCF {mf.cycles:2d} cycles {t_scf:7.1f} s, '
            f'E {e:.10f} ({e - ref[0]:+.2e} Eh)')
    if args.grad:
        line += f', gradient {t_grad:5.1f} s, max |dg| {numpy.abs(g - ref[1]).max():.2e}'
    print(line, flush=True)
