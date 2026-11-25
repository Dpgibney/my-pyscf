'''The two-electron second derivatives (J and full-range K) of the 57-atom doublet with the screening by one
density element at 1e-14 (the reference) and by two at looser thresholds: time and largest differences.
Usage: dd_big.py XYZ NPZ'''
import sys, time, numpy
from pyscf import gto, lib
from pyscf.hessian import rhf as rhf_hess
mol = gto.M(atom=sys.argv[1], basis='def2-tzvp', spin=1, verbose=0)
z = numpy.load(sys.argv[2]); mo, occ = z['mo_coeff'], z['mo_occ']
dms = numpy.array([mo[s][:, occ[s] > 0] @ mo[s][:, occ[s] > 0].T for s in range(2)])
dm0 = dms[0] + dms[1]
print(f'{mol.natm} atoms, {mol.nao} AOs, {lib.num_threads()} threads; two-electron second derivatives, J and full-range K', flush=True)
ref = None
for dd, tol in ((False, 1e-14), (True, 1e-13), (True, 1e-12), (True, 1e-11)):
    rhf_hess.JK_ENERGY_DD_SCREEN = dd
    stats = []
    t = time.perf_counter()
    ej, ek = rhf_hess.hess_jk_energy(mol, dm0, dms, 2, tol=tol, stats=stats)
    dt = time.perf_counter() - t
    if ref is None:
        ref = (ej, ek)
        numpy.savez('dd_big_ref.npz', ej=ej, ek=ek)
    print(f'{"two" if dd else "one"} density, tol {tol:.0e}: {dt:7.1f} s, {stats[0]:.3e} orbits, {stats[1]:.3e} integral evaluations; '
          f'max |d ej| {abs(ej - ref[0]).max():.1e}, max |d ek| {abs(ek - ref[1]).max():.1e}, max |d (ej - ek)| '
          f'{abs((ej - ek) - (ref[0] - ref[1])).max():.1e} (max |ej| {abs(ej).max():.2f}, |ek| {abs(ek).max():.2f})', flush=True)
