import time, numpy, sys
from pyscf import gto, dft, df, lib
from pyscf.df import df_jk, addons
from molecules import CAFFEINE, n_alkane
def test(name, mol, aux, nset, seed=3):
    nao = mol.nao
    rng = numpy.random.default_rng(seed)
    dms = rng.standard_normal((nset, nao, nao)) * rng.uniform(1e-3, 1, nset)[:, None, None]
    dms = dms + dms.transpose(0, 2, 1)
    w = dft.RKS(mol, xc='pbe').RIJCOSX(auxbasis=aux).with_df
    w.build(level=0)
    df_jk.get_j(w, dms[0], 1, 1e-13)
    with lib.temporary_env(df_jk, RIJ_MANY_MIN=10**9):
        t = time.perf_counter(); ref = df_jk.get_j(w, dms, 1, 1e-13); t0 = time.perf_counter() - t
    t = time.perf_counter(); j = df_jk.get_j(w, dms, 1, 1e-13); t1 = time.perf_counter() - t
    err = max(abs(j[k] - ref[k]).max() / abs(ref[k]).max() for k in range(nset))
    print(f'{name} naux {w.auxmol.nao} nset {nset}: one at a time {t0:.2f} s, together {t1:.2f} s, max rel diff {err:.1e}', flush=True)
for basis in ('cc-pvdz', 'def2-qzvp'):
    mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; C 2.5 0.3 0.1; Br 4.4 0.3 0.1', basis=basis, spin=1, verbose=0)
    for aux in ('def2-universal-jfit', 'def2-universal-jkfit'):
        test(f'CH3Br-like/{basis} {aux}', mol, aux, 5)
    test(f'{basis} aug_etb', mol, addons.aug_etb(mol), 3)
if len(sys.argv) > 1:
    mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
    test('caffeine/def2-TZVPD jfit', mol, 'def2-universal-jfit', 72)
    test('caffeine/def2-TZVPD aug_etb', mol, addons.aug_etb(mol), 72)
    mol = gto.M(atom=n_alkane(20), basis='def2-tzvp', verbose=0)
    test('n-C20H42/def2-TZVP jfit', mol, 'def2-universal-jfit', 186)
