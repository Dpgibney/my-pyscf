"""The ground-state gradient of a pure functional through the J engine (part 24) against the Rys path, in one process
at one converged SCF (node1): the gradient once as is, once with j_engine.ej_ip1_available patched to False (the Rys
kernel RYS_per_atom_jk_ip1 with k_factor 0), each timed after a first, untimed call.
Usage: grad_jengine_check.py [--xc pbe] [--basis def2-svp] [--uks] [--save F.npz] (--uks: the caffeine cation)"""
import argparse
import os
import sys
import time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.scf import j_engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='pbe')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--uks', action='store_true')
ap.add_argument('--save', default=None, help='npz: the total density and both paths\' Coulomb gradients (dm, jengine, rys)')
args = ap.parse_args()

if args.uks:
    mol = gto.M(atom=CAFFEINE, basis=args.basis, charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc=args.xc)
else:
    mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.max_cycle = 150
mf.kernel()
if not mf.converged:
    print('SCF not converged', file=sys.stderr)


def grad():
    g = mf.nuc_grad_method()
    g.verbose = 0
    g.kernel()
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    de = g.kernel()
    cupy.cuda.Device().synchronize()
    return np.asarray(de), time.perf_counter() - t0


assert j_engine.ej_ip1_available()
g1, t1 = grad()
available = j_engine.ej_ip1_available
j_engine.ej_ip1_available = lambda: False
g0, t0 = grad()
j_engine.ej_ip1_available = available
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine{" cation UKS" if args.uks else ""} {args.xc} {args.basis}: '
      f'E {mf.e_tot:.10f}; gradient J engine {t1:.2f} s, Rys {t0:.2f} s; max |dg| {abs(g1 - g0).max():.2e} '
      f'(max |g| {abs(g0).max():.3e}); sum over atoms {abs(g1.sum(axis=0)).max():.1e} (Rys {abs(g0.sum(axis=0)).max():.1e})')
if args.save:
    from gpu4pyscf.grad import rhf as rhf_grad
    from gpu4pyscf.scf import jk
    dm = mf.make_rdm1()
    dm_tot = cupy.asarray(dm).sum(axis=0) if dm.ndim == 3 else cupy.asarray(dm)
    ej1 = mf._opt_jengine[mol.omega].get_ej_ip1(dm_tot)
    ej0 = rhf_grad._jk_energy_per_atom(jk._VHFOpt(mol, mf.direct_scf_tol).build(), dm, 1., 0.)
    np.savez(args.save, dm=dm_tot.get(), jengine=ej1, rys=ej0)
    print(f'saved {args.save}: max |dej| {abs(ej1 - ej0).max():.2e}')
