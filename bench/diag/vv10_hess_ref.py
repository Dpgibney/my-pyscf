'''Whole results through the VV10 Hessian code, saved to compare trees: the RKS Hessian of a wB97M-V molecule and
the lowest TDA and TDDFT excitation energies (their response takes get_vnlc_resp), with the time of each.
Compare two outputs with vv10_hess_parts.py --compare A.npz B.npz.

Usage: vv10_hess_ref.py OUT.npz [--mol water|propane] [--basis def2-svp] [--nlc-level 1] [--xc wb97m-v]'''
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--mol', default='water')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--nlc-level', type=int, default=1)
ap.add_argument('--xc', default='wb97m-v')
args = ap.parse_args()

atom = {'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        'propane': n_alkane(3)}[args.mol]
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-11
mf.nlcgrids.level = args.nlc_level
mf.kernel()
out = {'e_tot': numpy.array([mf.e_tot])}
t = time.perf_counter()
out['hessian'] = mf.Hessian().kernel()
print(f'{args.mol}/{args.basis} {args.xc} VV10 level {args.nlc_level}: E {mf.e_tot:.10f}, '
      f'Hessian {time.perf_counter() - t:.1f} s, max|H| {abs(out["hessian"]).max():.6f}', flush=True)
for name in ('TDA', 'TDDFT'):
    td = getattr(mf, name)()
    td.nstates = 4
    td.exclude_nlc = False  # PySCF leaves VV10 out of the TD response by default
    td.conv_tol = 1e-10
    t = time.perf_counter()
    td.kernel()
    out[name] = numpy.asarray(td.e)
    print(f'  {name} {time.perf_counter() - t:.1f} s: {" ".join(f"{e:.10f}" for e in td.e)}', flush=True)
numpy.savez(args.out, **out)
