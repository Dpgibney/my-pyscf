'''An RKS/RHF Hessian with the CPHF's response calls counted: the calls of fvind, the vectors passed, their
time, and the Hessian saved to compare trees (vv10_hess_parts.py --compare A.npz B.npz).

Usage: cphf_calls.py OUT.npz [--mol water|propane|caffeine|alkane:N] [--basis def2-svp] [--xc wb97m-v]
       [--nlc-level 1] [--warm-start]'''
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, scf, lib  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--mol', default='water')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--nlc-level', type=int, default=1)
ap.add_argument('--warm-start', action='store_true')
args = ap.parse_args()

if args.mol == 'water':
    atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
elif args.mol == 'caffeine':
    atom = CAFFEINE
elif args.mol.startswith('alkane:'):
    atom = n_alkane(int(args.mol.split(':')[1]))
else:
    atom = n_alkane(3)
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
if args.xc.lower() == 'hf':
    mf = scf.RHF(mol)
else:
    mf = dft.RKS(mol, xc=args.xc)
    mf.nlcgrids.level = args.nlc_level
mf.conv_tol = 1e-11
if os.environ.get("CPSCF_TOL"):
    mf.conv_tol_cpscf = float(os.environ["CPSCF_TOL"])
mf.kernel()

calls = []
solve0 = cphf.solve
def solve(fvind, *a, **k):
    def counted(mo1):
        t = time.perf_counter()
        v = fvind(mo1)
        calls.append((len(mo1.reshape(-1, mo1.shape[-2]*mo1.shape[-1])) if mo1.ndim > 1 else 1,
                      time.perf_counter() - t))
        return v
    return solve0(counted, *a, **k)
cphf.solve = solve

hobj = mf.Hessian()
if args.warm_start:
    hobj.cphf_warm_start = True
t = time.perf_counter()
h = hobj.kernel()
total = time.perf_counter() - t
nvec = [n for n, t_ in calls]
print(f'{args.mol}/{args.basis} {args.xc}: E {mf.e_tot:.10f}; Hessian {total:.1f} s; fvind calls {len(calls)}, '
      f'vectors {nvec} (sum {sum(nvec)}, 3*natm {3*mol.natm}), {sum(t_ for n, t_ in calls):.1f} s in fvind; '
      f'max|H| {abs(h).max():.6f}', flush=True)
numpy.savez(args.out, hessian=h, e_tot=numpy.array([mf.e_tot]))
