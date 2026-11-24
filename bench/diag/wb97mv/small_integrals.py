'''How much of an exact J + K pass (and of the long-range K pass) goes to shell quartets below a density-weighted
Schwarz bound: the pass with direct_scf_tol at that bound leaves them out, so the difference to 1e-13 is their time.
Those quartets could be evaluated in single precision (relative error 1e-7 of a contribution below the bound).
Usage: small_integrals.py caffeine|XYZ DM.npy|SCF.npz [--charge N] [--tols 1e-13,1e-10,...]'''
import sys, time, numpy
from pyscf import gto, dft
import molecules
a = sys.argv[1:]
charge = int(a[a.index('--charge') + 1]) if '--charge' in a else 0
tols = [float(x) for x in (a[a.index('--tols') + 1] if '--tols' in a else '1e-13,1e-10,1e-8,1e-6,1e-4').split(',')]
mol = gto.M(atom=getattr(molecules, a[0].upper(), None) or a[0], basis='def2-tzvpd', charge=charge, verbose=0)
if a[1].endswith('.npz'):
    d = numpy.load(a[1]); c = d['mo_coeff'][:, d['mo_occ'] > 0]; dm = 2 * c @ c.T
else:
    dm = numpy.load(a[1])
print(f'{mol.natm} atoms, {mol.nao} AOs', flush=True)
ref = None
for tol in tols:
    mf = dft.RKS(mol, xc='wb97m-v'); mf.semi_direct = False; mf.direct_scf_tol = tol
    t = time.perf_counter(); vj, vk = mf.get_jk(mol, dm); t1 = time.perf_counter() - t
    t = time.perf_counter(); vkl = mf.get_k(mol, dm, omega=0.3); t2 = time.perf_counter() - t
    if ref is None:
        ref = (vj, vk, vkl, t1, t2)
    e = numpy.einsum('ij,ji', dm, .5 * (vj - ref[0]) - .25 * .15 * (vk - ref[1]) - .25 * .85 * (vkl - ref[2]))
    print(f'bound {tol:7.0e}: J + K {t1:7.1f} s ({100 * t1 / ref[3]:3.0f}%), long-range K {t2:7.1f} s ({100 * t2 / ref[4]:3.0f}%); '
          f'left out: dE {e:+.1e} Eh', flush=True)
