'''The two-electron energy second derivatives (hess_jk_energy) and gradient (jk_energy_per_atom) with the screening
by two density elements against the one by one element, at several thresholds: wall time, orbits computed, and the
largest difference from the one-element bound at 1e-14.
Usage: dd_screen_check.py XYZ SPIN [NPZ with mo_coeff, mo_occ] [--tols 1e-14,1e-12,...] [--skip-old] [--no-grad]'''
import sys, time, numpy
from pyscf import gto, dft, lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rhf as rhf_grad
args = [x for x in sys.argv[1:] if not x.startswith('--')]
opts = [x for x in sys.argv[1:] if x.startswith('--')]
tols = [1e-14, 1e-12, 1e-11, 1e-10]
for o in opts:
    if o.startswith('--tols='):
        tols = [float(x) for x in o[7:].split(',')]
mol = gto.M(atom=args[0], basis='def2-tzvp', spin=int(args[1]), verbose=0)
if len(args) > 2:
    z = numpy.load(args[2]); mo, occ = z['mo_coeff'], z['mo_occ']
else:
    mf = dft.UKS(mol, xc='wb97m-v'); mf.grids.atom_grid = (50, 194); mf.nlcgrids.atom_grid = (30, 110)
    mf.kernel(); mo, occ = mf.mo_coeff, mf.mo_occ
dms = numpy.array([mo[s][:, occ[s] > 0] @ mo[s][:, occ[s] > 0].T for s in range(2)])
dm0 = dms[0] + dms[1]
print(f'{mol.natm} atoms, {mol.nao} AOs, {lib.num_threads()} threads', flush=True)
ref = {}
for omega, label in ((None, 'J + K'), (.3, 'long-range K')):
    for dd in (False, True):
        if not dd and '--skip-old' in opts:
            continue
        for tol in (tols if dd else tols[:1]):
            rhf_hess.JK_ENERGY_DD_SCREEN = dd
            stats = []
            t = time.perf_counter()
            with mol.with_range_coulomb(omega):
                ej, ek = rhf_hess.hess_jk_energy(mol, None if omega else dm0, dms, 2, tol=tol, stats=stats)
            dt = time.perf_counter() - t
            key = label
            if key not in ref:
                ref[key] = (ej, ek)
            print(f'second derivatives, {label:13s} {"two" if dd else "one"} density, tol {tol:.0e}: {dt:7.1f} s, '
                  f'{stats[0]:.3e} orbits, {stats[1]:.3e} integral evaluations; max |d ej| {abs(ej - ref[key][0]).max():.1e}, '
                  f'max |d ek| {abs(ek - ref[key][1]).max():.1e} (max |ej| {abs(ej).max():.2f}, |ek| {abs(ek).max():.2f})', flush=True)
if '--no-grad' not in opts:
    gref = None
    for dd in (False, True):
        rhf_grad.JK_ENERGY_DD_SCREEN = dd
        t = time.perf_counter()
        de = rhf_grad.jk_energy_per_atom(mol, dms, j_factor=1., k_factor=.15)
        de += rhf_grad.jk_energy_per_atom(mol, dms, j_factor=0., k_factor=.85, omega=.3)
        dt = time.perf_counter() - t
        if gref is None:
            gref = de
        print(f'gradient, J + K and long-range K, {"two" if dd else "one"} density: {dt:7.1f} s; max |d de| {abs(de - gref).max():.1e} '
              f'(max |de| {abs(de).max():.2f})', flush=True)
