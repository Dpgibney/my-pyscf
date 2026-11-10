'''RI-J through the J-engine by auxiliary basis, at a fixed converged density: size, time per build and error of
E_J = 1/2 tr(D J) and of J against the exact J (direct, no cache). Usage: rij_aux.py MOL NPZ'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, df, lib
from pyscf.df import df_jk, addons
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('npz')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
z = numpy.load(args.npz)
mf = dft.RKS(mol, xc=args.xc)
mf.mo_coeff, mf.mo_occ = z['mo_coeff'], z['mo_occ']
dm = mf.make_rdm1()
mf.semi_direct = False
t = time.perf_counter()
vj = mf.get_j(mol, dm, 1)
t_exact = time.perf_counter() - t
ej = .5 * numpy.einsum('ij,ji', dm, vj)
print(f'{args.mol}/{args.basis}: nao {mol.nao}; exact direct J {t_exact:.1f} s, E_J {ej:.8f}', flush=True)

cands = [('def2-universal-jfit', 'def2-universal-jfit'),
         ('def2-universal-jkfit', 'def2-universal-jkfit'),
         ('def2-tzvpp-jkfit', 'def2-tzvpp-jkfit'),
         ('def2-qzvpp-jkfit', 'def2-qzvpp-jkfit'),
         ('aug-cc-pvtz-jkfit', 'aug-cc-pvtz-jkfit'),
         ('aug-cc-pvqz-jkfit', 'aug-cc-pvqz-jkfit'),
         ('aug_etb beta=2.0', lambda: addons.aug_etb(mol, beta=2.0)),
         ('aug_etb beta=1.7', lambda: addons.aug_etb(mol, beta=1.7)),
         ('autoaux', lambda: addons.autoaux(mol)),
         ('autoabs', lambda: addons.autoabs(mol))]
for name, aux in cands:
    try:
        if callable(aux):
            aux = aux()
        mfd = dft.RKS(mol, xc=args.xc).RIJCOSX(auxbasis=aux)
        w = mfd.with_df
        w.build(level=0)
        t = time.perf_counter()
        j = df_jk.get_j(w, dm, 1, 1e-13)
        t1 = time.perf_counter() - t
        t = time.perf_counter()
        j = df_jk.get_j(w, dm, 1, 1e-13)
        t2 = time.perf_counter() - t
        e = .5 * numpy.einsum('ij,ji', dm, j)
        am = w.auxmol
        lmax = max(am.bas_angular(i) for i in range(am.nbas))
        print(f'  {name:22s} naux {am.nao:6d} lmax {lmax}  first {t1:6.2f} s  build {t2:6.2f} s  dE_J {e - ej:+.3e}  '
              f'max|dJ| {abs(j - vj).max():.2e}', flush=True)
    except Exception as err:
        print(f'  {name}: {type(err).__name__}: {str(err)[:200]}', flush=True)
print('done')
