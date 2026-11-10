'''At a converged exact wB97M-V density: the exact J, full-range K and long-range K (direct, no semi-direct cache)
against SGX's K of both operators by grid level and RI-J by auxiliary basis. Energies are the terms of the
functional: E_J = 1/2 tr(D J), E_K = -1/4 tr(D K); wB97M-V takes hyb K_full + (alpha - hyb) K_LR.
Usage: k_accuracy.py MOL NPZ [--basis def2-tzvpd] [--levels 0 1 2 3]'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.df import df_jk
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('npz')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3])
ap.add_argument('--aux', nargs='+', default=['def2-universal-jfit', 'def2-universal-jkfit'])
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
z = numpy.load(args.npz)
mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
mf = dft.RKS(mol, xc=args.xc)
mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo_coeff, mo_occ, z['mo_energy']
dm = mf.make_rdm1()
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
print(f'{args.mol}/{args.basis}: nao {mol.nao}; omega {omega} alpha {alpha} hyb {hyb}', flush=True)

mf.semi_direct = False


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


(vj, vk), t_jk = timed(mf.get_jk, mol, dm, 1)
vklr, t_klr = timed(mf.get_k, mol, dm, 1, omega=omega)
vj1, t_j = timed(mf.get_j, mol, dm, 1)
(_, vk1), t_k = timed(mf.get_jk, mol, dm, 1, with_j=False)
ej = .5 * numpy.einsum('ij,ji', dm, vj)
ek = -.25 * numpy.einsum('ij,ji', dm, vk)
eklr = -.25 * numpy.einsum('ij,ji', dm, vklr)
print(f'exact direct (no cache): J+K {t_jk:.1f} s, LR-K {t_klr:.1f} s, J alone {t_j:.1f} s, K alone {t_k:.1f} s')
print(f'  E_J {ej:.8f}  E_K(full) {ek:.8f}  E_K(LR) {eklr:.8f}; max|J| {abs(vj).max():.3f} max|K| {abs(vk).max():.3f} '
      f'max|K_LR| {abs(vklr).max():.3f}; J alone vs J+K {abs(vj1 - vj).max():.1e}', flush=True)

print('\ndirect_scf_tol scan (exact J/K, no cache): time and error against 1e-13')
for tol in (1e-12, 1e-11, 1e-10, 1e-9):
    mft = dft.RKS(mol, xc=args.xc)
    mft.direct_scf_tol = tol
    mft.semi_direct = False
    (j_t, k_t), t1 = timed(mft.get_jk, mol, dm, 1)
    klr_t, t2 = timed(mft.get_k, mol, dm, 1, omega=omega)
    dej = .5 * numpy.einsum('ij,ji', dm, j_t) - ej
    dek = -.25 * numpy.einsum('ij,ji', dm, k_t) - ek
    deklr = -.25 * numpy.einsum('ij,ji', dm, klr_t) - eklr
    print(f'  tol {tol:.0e}: J+K {t1:6.1f} s  LR-K {t2:6.1f} s  dE_J {dej:+.2e} dE_K {dek:+.2e} dE_KLR {deklr:+.2e} '
          f'max|dJ| {abs(j_t - vj).max():.1e} max|dK| {abs(k_t - vk).max():.1e} max|dK_LR| {abs(klr_t - vklr).max():.1e}', flush=True)

print('\nSGX K (RIJCOSX object, P-junction screening): errors against exact')
print(f'{"level":>5s} {"pair_tol":>8s} {"points":>8s} | {"t full":>7s} {"dE_K full":>11s} {"max|dK|":>9s} | {"t LR":>7s} '
      f'{"dE_K LR":>11s} {"max|dK|":>9s} | {"wB97M-V dE_x":>12s}')
mfx = dft.RKS(mol, xc=args.xc).RIJCOSX()
mfx.mo_coeff, mfx.mo_occ = mo_coeff, mo_occ
sgx = mfx.with_df
for level in args.levels:
    for ptol in ((None, 1e-9) if level == 1 else (1e-9,)):
        sgx.sgx_tol_pair = ptol
        sgx.grids_level_i = sgx.grids_level_f = level
        sgx._rsh_df = {}
        sgx.build(level=level)
        sgx._full_dm = None
        # first calls include the P-junction setup of each operator; time the second
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True)
        (_, k1), t1 = timed(sgx.get_jk, dm, 1, None, with_j=False, with_k=True)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=omega)
        (_, k2), t2 = timed(sgx.get_jk, dm, 1, None, with_j=False, with_k=True, omega=omega)
        e1 = -.25 * numpy.einsum('ij,ji', dm, k1)
        e2 = -.25 * numpy.einsum('ij,ji', dm, k2)
        tot = hyb * (e1 - ek) + (alpha - hyb) * (e2 - eklr)
        print(f'{level:5d} {str(ptol):>8s} {sgx.grids.weights.size:8d} | {t1:7.2f} {e1 - ek:+11.3e} {abs(k1 - vk).max():9.2e} | '
              f'{t2:7.2f} {e2 - eklr:+11.3e} {abs(k2 - vklr).max():9.2e} | {tot:+12.3e}', flush=True)

print('\nRI-J (J-engine) by auxiliary basis: errors against exact')
for aux in args.aux:
    try:
        mfd = dft.RKS(mol, xc=args.xc).RIJCOSX(auxbasis=aux)
        w = mfd.with_df
        w.build(level=0)
        df_jk.get_j(w, dm, 1, 1e-13)
        j, t = timed(df_jk.get_j, w, dm, 1, 1e-13)
        e = .5 * numpy.einsum('ij,ji', dm, j)
        print(f'  {aux:24s} naux {w.auxmol.nao:6d}  t {t:6.2f} s  dE_J {e - ej:+.3e}  max|dJ| {abs(j - vj).max():.2e}', flush=True)
    except Exception as err:
        print(f'  {aux}: {type(err).__name__}: {err}')
print('done')
