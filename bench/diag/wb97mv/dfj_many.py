'''J of many CPHF-like density matrices (nuclear perturbations: 3*natm symmetric dms from make_h1-like trial
vectors): exact direct J, RI-J through the J-engine (one density at a time, as df_jk.get_j does now), and what
"integrals once, contractions as matrix products" costs, measured with the stored 3-index tensor of PySCF's DF
(tensor build = the integrals once; its two products = the contractions of all densities).
Usage: dfj_many.py MOL NPZ [--aux aug_etb] [--nset 72]'''
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
ap.add_argument('--aux', nargs='+', default=['def2-universal-jfit', 'aug_etb'])
ap.add_argument('--nset', type=int, default=None)
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
z = numpy.load(args.npz)
mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
nao, nmo = mo_coeff.shape
mocc = mo_coeff[:, mo_occ > 0]
nocc = mocc.shape[1]
nset = args.nset or 3 * mol.natm
# CPHF-like trial densities: C x O^T + h.c. with x of unit norm in the virtual-occupied block
rng = numpy.random.default_rng(7)
x = numpy.zeros((nset, nmo, nocc))
x[:, nocc:] = rng.standard_normal((nset, nmo - nocc, nocc))
x /= numpy.linalg.norm(x.reshape(nset, -1), axis=1)[:, None, None]
a = lib.einsum('pa,kai->kpi', mo_coeff, x)
dms = numpy.matmul(a, mocc.T)
dms = dms + dms.transpose(0, 2, 1)
print(f'{args.mol}/{args.basis}: nao {nao}, {nset} densities', flush=True)

mf = dft.RKS(mol, xc=args.xc)
mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ
mf.semi_direct = False
t = time.perf_counter()
vj = mf.get_j(mol, dms, 1)
print(f'exact direct J, {nset} dms: {time.perf_counter() - t:.1f} s', flush=True)
t = time.perf_counter()
vj1 = mf.get_j(mol, dms[0], 1)
print(f'exact direct J, 1 dm: {time.perf_counter() - t:.1f} s', flush=True)
t = time.perf_counter()
vjk = mf.get_jk(mol, dms, 1)
print(f'exact direct J+K, {nset} dms: {time.perf_counter() - t:.1f} s', flush=True)
t = time.perf_counter()
vklr = mf.get_k(mol, dms, 1, omega=0.3)
print(f'exact direct LR-K, {nset} dms: {time.perf_counter() - t:.1f} s', flush=True)
vjk = vklr = None

for aux in args.aux:
    name = aux
    if aux.startswith('aug_etb'):
        aux = addons.aug_etb(mol, beta=float(aux.split(':')[1]) if ':' in aux else 2.0)
    elif aux == 'autoaux':
        aux = addons.autoaux(mol)
    w = dft.RKS(mol, xc=args.xc).RIJCOSX(auxbasis=aux).with_df
    w.build(level=0)
    df_jk.get_j(w, dms[0], 1, 1e-13)
    t = time.perf_counter()
    j = df_jk.get_j(w, dms, 1, 1e-13)
    t_eng = time.perf_counter() - t
    err = abs(j - vj).max() / abs(vj).max()
    naux = w.auxmol.nao
    print(f'RI-J {name} (naux {naux}): J-engine, one density at a time: {t_eng:.1f} s for {nset} dms '
          f'({t_eng / nset:.2f} s each); max|dJ|/max|J| {err:.1e}', flush=True)
    # integrals once + products: PySCF's stored tensor
    try:
        d = df.DF(mol, auxbasis=aux)
        d.max_memory = 12000
        t = time.perf_counter()
        d.build()
        t_build = time.perf_counter() - t
        t = time.perf_counter()
        j2 = df_jk.get_jk(d, dms, 1, with_j=True, with_k=False)[0]
        t_gemm = time.perf_counter() - t
        print(f'    stored tensor: build (integrals once + Cholesky) {t_build:.1f} s, J of {nset} dms by products '
              f'{t_gemm:.2f} s; vs engine max|dJ| {abs(j2 - j).max():.1e}', flush=True)
        d = None
    except Exception as e:
        print('    stored tensor failed:', type(e).__name__, str(e)[:200])
print('done')
