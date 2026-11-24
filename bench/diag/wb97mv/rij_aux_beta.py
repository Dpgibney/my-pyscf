'''The error of RI-J against the size of the even-tempered fitting basis (df.aug_etb(mol, beta)) at a fixed density:
E_J = 1/2 tr(D J) and the largest element of J against the exact J, with the J-engine's time.
Usage: rij_aux_beta.py caffeine|XYZ DM.npy|SCF.npz [--charge N] [--no-exact] [BETA ...]'''
import sys, time, numpy
from pyscf import gto, dft, df
from pyscf.df import df_jk
import molecules
a = sys.argv[1:]
charge = int(a[a.index('--charge') + 1]) if '--charge' in a else 0
exact = '--no-exact' not in a
betas = [float(x) for x in a[2:] if x[0].isdigit()] or [2.3, 2.0, 1.8, 1.6, 1.5]
mol = gto.M(atom=getattr(molecules, a[0].upper(), None) or a[0], basis='def2-tzvpd', charge=charge, verbose=0)
if a[1].endswith('.npz'):
    d = numpy.load(a[1]); c = d['mo_coeff'][:, d['mo_occ'] > 0]; dm = 2 * c @ c.T
else:
    dm = numpy.load(a[1])
print(f'{mol.natm} atoms, {mol.nao} AOs', flush=True)
vj0 = None
if exact:
    mf = dft.RKS(mol); mf.semi_direct = False
    t = time.perf_counter(); vj0 = mf.get_j(mol, dm); print(f'exact J: {time.perf_counter() - t:.1f} s', flush=True)
for name in ['def2-universal-jfit'] + betas:
    aux = name if isinstance(name, str) else df.aug_etb(mol, name)
    dfobj = df.DF(mol, auxbasis=aux)
    try:
        t = time.perf_counter(); vj = df_jk.get_j(dfobj, dm, hermi=1); t1 = time.perf_counter() - t
        t = time.perf_counter(); vj = df_jk.get_j(dfobj, dm, hermi=1); t2 = time.perf_counter() - t
    except Exception as e:
        print(f'{name}: {type(e).__name__} {e}', flush=True); continue
    eng = getattr(dfobj, '_rij_engine', None)
    err = '' if vj0 is None else f', E_J error {.5 * numpy.einsum("ij,ji", dm, vj - vj0):+.2e} Eh, max|dJ| {abs(vj - vj0).max():.1e}'
    print(f'{"jfit" if isinstance(name, str) else "aug_etb beta %.1f" % name:18s} {dfobj.auxmol.nao:6d} functions, J {t2:6.2f} s (first call {t1:.1f} s), '
          f'metric by {getattr(eng, "j2c_type", "?")}{err}', flush=True)
