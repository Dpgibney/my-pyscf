"""Pass structures for the two-electron part of a range-separated hybrid's nuclear gradient (node1). The exchange of
such a functional is hyb K_full + (alpha - hyb) K_LR = alpha K_full + (hyb - alpha) K_SR, and the Rys derivative
kernels evaluate the long-range integrals with n roots, the short-range and the mixed ones with 2n. After an SCF
(default grids; the XC part plays no role here) each structure is timed between device synchronizations at the
converged density, after one untimed call of each pass:

  A  Rys(J + alpha K_full)  +  Rys((hyb - alpha) K_SR)            what grad.rks.energy_ee does at v1.9.0
  B  Rys(J + hyb K_full)    +  Rys((alpha - hyb) K_LR)
  C  J engine               +  Rys(K of alpha erf + hyb erfc)      one mixed pass, 2n roots
  D  J engine  +  Rys(hyb K_full)    +  Rys((alpha - hyb) K_LR)
  E  J engine  +  Rys(alpha K_full)  +  Rys((hyb - alpha) K_SR)

and the sums are compared with A's. Run it with PYTHONPATH set to a tree whose jk_energy_per_atom takes lr_factor
and sr_factor (source scripts/node1-gpu-env.sh perf).
Usage: rsh_grad_passes.py [--mol caffeine|nad|FILE.xyz] [--charge 0] [--basis def2-svp] [--xc wb97x] [--repeat 2]
                          [--no-warm] [--out FILE]"""
import argparse
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--charge', type=int, default=None)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97x')
ap.add_argument('--repeat', type=int, default=2)
ap.add_argument('--no-warm', action='store_true', help='no untimed first call of each pass (large molecules)')
ap.add_argument('--out', default=None)
args = ap.parse_args()

if args.mol == 'caffeine':
    atom, charge = CAFFEINE, 0
elif args.mol == 'nad':
    atom, charge = os.path.join(here, '..', 'diag', 'wb97mv', 'nad+_mmff.xyz'), 1
else:
    atom, charge = args.mol, 0
if args.charge is not None:
    charge = args.charge
mol = gto.M(atom=atom, basis=args.basis, charge=charge, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
t0 = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t0
assert mf.converged
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
assert omega != 0 and alpha != 0 and hyb != 0
g = mf.Gradients()
dm = mf.make_rdm1()

PASSES = {
    'J+aK': lambda: g.jk_energy_per_atom(dm, 1, alpha),
    'J+hK': lambda: g.jk_energy_per_atom(dm, 1, hyb),
    'Jeng': lambda: g.jk_energy_per_atom(dm, 1, 0),
    'aK': lambda: g.jk_energy_per_atom(dm, 0, alpha),
    'hK': lambda: g.jk_energy_per_atom(dm, 0, hyb),
    'K_SR': lambda: g.jk_energy_per_atom(dm, 0, hyb - alpha, omega=-omega),
    'K_LR': lambda: g.jk_energy_per_atom(dm, 0, alpha - hyb, omega=omega),
    'K_mix': lambda: g.jk_energy_per_atom(dm, 0, 1, omega=omega, lr_factor=alpha, sr_factor=hyb),
}
STRUCTURES = {'A': ['J+aK', 'K_SR'], 'B': ['J+hK', 'K_LR'], 'C': ['Jeng', 'K_mix'], 'D': ['Jeng', 'hK', 'K_LR'],
              'E': ['Jeng', 'aK', 'K_SR']}


def timed(fn):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = fn()
    cupy.cuda.Device().synchronize()
    return numpy.asarray(cupy.asnumpy(out)), time.perf_counter() - t0


if not args.no_warm:
    for fn in PASSES.values():   # each pass once, untimed: screening data, kernel modules
        fn()
times = {k: [] for k in PASSES}
vals = {}
with GpuMonitor() as mon:
    for _ in range(args.repeat):
        for k, fn in PASSES.items():
            vals[k], t = timed(fn)
            times[k].append(t)
best = {k: min(v) for k, v in times.items()}
ref = sum(vals[k] for k in STRUCTURES['A'])
print(f"{args.mol} {args.basis} {args.xc}: nao={mol.nao} natm={mol.natm} omega={omega} alpha={alpha} hyb={hyb:.6f} "
      f"SCF {t_scf:.1f} s, {mf.cycles} cycles, SM clock {mon.summary.get('sm_mhz')}")
print('passes (best of %d, s): ' % args.repeat + '  '.join(f'{k} {v:.3f}' for k, v in best.items()))
rec = {'mol': args.mol, 'basis': args.basis, 'xc': args.xc, 'nao': int(mol.nao), 'natm': int(mol.natm),
       'tree': os.path.dirname(gpu4pyscf.__file__), 'passes': {k: [round(x, 4) for x in v] for k, v in times.items()},
       'structures': {}, 'sm_mhz': mon.summary.get('sm_mhz')}
for name, ks in STRUCTURES.items():
    tot = sum(best[k] for k in ks)
    diff = float(abs(sum(vals[k] for k in ks) - ref).max())
    rec['structures'][name] = {'s': round(tot, 4), 'max_diff_from_A': diff}
    print(f"  {name}: {' + '.join(ks):22s} {tot:8.3f} s   {tot / sum(best[k] for k in STRUCTURES['A']):5.2f} of A   "
          f"max |d| from A {diff:.2e}")
print(f"  (largest gradient component of the two-electron part: {abs(ref).max():.3e})")
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
