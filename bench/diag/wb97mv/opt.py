'''Geometry optimization for the Hessian tests: BFGS in Cartesian coordinates from scipy, started with the inverse of
a cheap Hessian (RHF/def2-SVP), on the wB97M-V/def2-TZVPD surface (exact J/K, XC grid level 3, VV10 grid level 1).
Usage: opt.py NAME [--charge 0] [--gtol 1e-4] -> NAME_opt.xyz in the scratchpad'''
import argparse
import os
import time
import numpy
import scipy.optimize
import pyscf
from pyscf import gto, scf, dft, lib

GEOM = {
    'h2o2': '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561''',
    'glycine': '''
N   0.000   0.000   0.000
C   1.449   0.000   0.000
C   2.006   1.420   0.000
O   1.330   2.420   0.000
O   3.360   1.470   0.000
H   3.620   2.400   0.000
H  -0.340  -0.480   0.830
H  -0.340  -0.480  -0.830
H   1.830  -0.520   0.880
H   1.830  -0.520  -0.880''',
    'h2po4': '''
P   0.000   0.000   0.000
O   0.000   1.250   0.830
O   0.000  -1.250   0.830
O   1.300   0.000  -0.960
O  -1.300   0.000  -0.960
H   1.900   0.700  -0.700
H  -1.900  -0.700  -0.700''',
}
ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--gtol', type=float, default=1e-4)
ap.add_argument('--maxiter', type=int, default=60)
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)

mol0 = gto.M(atom=GEOM[args.name], basis='def2-svp', charge=args.charge, verbose=0)
h0 = scf.RHF(mol0).run().Hessian().kernel()
n = mol0.natm * 3
h0 = h0.transpose(0, 2, 1, 3).reshape(n, n)
w, v = numpy.linalg.eigh((h0 + h0.T) / 2)
hinv = (v / numpy.maximum(abs(w), 0.02)) @ v.T
hinv = (hinv + hinv.T) / 2

mol = gto.M(atom=GEOM[args.name], basis=args.basis, charge=args.charge, verbose=0)
mf = dft.RKS(mol, xc='wb97m-v')
mf.grids.level = 3
mf.nlcgrids.level = 1
mf.conv_tol = 1e-10
scanner = mf.nuc_grad_method().as_scanner()
hist = []


def fun(x):
    t = time.perf_counter()
    e, g = scanner(mol.set_geom_(x.reshape(-1, 3), unit='Bohr', inplace=False))
    hist.append(e)
    print(f'  eval {len(hist):3d}  E {e:.10f}  max|g| {abs(g).max():.2e}  {time.perf_counter() - t:.1f} s', flush=True)
    return e, g.ravel()


x0 = mol.atom_coords().ravel()
res = scipy.optimize.minimize(fun, x0, jac=True, method='BFGS',
                              options={'gtol': args.gtol, 'norm': numpy.inf, 'hess_inv0': hinv, 'maxiter': args.maxiter})
print(res.message, 'nfev', res.nfev, 'E', res.fun, 'max|g|', abs(res.jac).max())
xyz = res.x.reshape(-1, 3) * lib.param.BOHR
out = os.path.join(here, args.name + '_opt.xyz')
with open(out, 'w') as f:
    f.write(f'{mol.natm}\n{args.name} wB97M-V/{args.basis} E={res.fun:.10f} max|g|={abs(res.jac).max():.2e}\n')
    for i in range(mol.natm):
        f.write(f'{mol.atom_symbol(i):2s} {xyz[i,0]:14.8f} {xyz[i,1]:14.8f} {xyz[i,2]:14.8f}\n')
print('wrote', out)
