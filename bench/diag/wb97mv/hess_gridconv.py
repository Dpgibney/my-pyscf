'''How far the "exact" wB97M-V Hessian is converged with its own grids: SCF + Hessian (exact response, CPHF to 1e-6) at
one geometry for several XC grids and VV10 grids, each calculation consistent (the same grids in the SCF and in every
Hessian term). The CPU Hessian has no grid-weight derivatives for the XC terms. Frequencies against the first setting.
Usage: hess_gridconv.py XYZ [--charge 0] [--settings 'xc;nlc|xc;nlc|...']  (a grid: a level, or nrad,nang)'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.hessian import thermo

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--settings', default='99,590;50,194|99,590;1|99,590;3|99,590;99,590|4;50,194|5;50,194|150,974;50,194')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
mol = gto.M(atom=args.mol, basis=args.basis, charge=args.charge, verbose=0)


def setgrid(g, spec):
    if ',' in spec:
        g.atom_grid = tuple(int(x) for x in spec.split(','))
    else:
        g.level = int(spec)


f0 = h0 = None
for s in args.settings.split('|'):
    xc, nlc = s.split(';')
    mf = dft.RKS(mol, xc='wb97m-v')
    setgrid(mf.grids, xc)
    setgrid(mf.nlcgrids, nlc)
    mf.conv_tol = 1e-11
    t = time.perf_counter()
    mf.kernel()
    g = mf.Gradients().kernel()
    hobj = mf.Hessian()
    hobj.cphf_residual_tol = 1e-6
    h = hobj.kernel()
    dt = time.perf_counter() - t
    f = numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
    line = (f'XC {xc:>8s} ({mf.grids.weights.size:7d} pts)  VV10 {nlc:>7s} ({mf.nlcgrids.weights.size:6d} pts)  E {mf.e_tot:.9f}  '
            f'max|g| {abs(g).max():.1e}  {dt:6.1f} s')
    if f0 is None:
        f0, h0 = f, h
        print(line)
        print('   frequencies:', ' '.join(f'{x:.2f}' for x in f), flush=True)
    else:
        df = f - f0
        i = abs(df).argmax()
        hi = abs(f0) > 200
        print(line + f'  max|dH| {abs(h - h0).max():.1e}  max|df| {abs(df).max():.3f} at {f0[i]:.0f}  mean|df| {abs(df).mean():.3f}  '
              f'max|df| f>200 {abs(df[hi]).max():.3f}', flush=True)
print('done')
