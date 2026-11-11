'''The RI-J gradient by the J-engine (df/grad/rhf.py _get_j_engine) against the libcint path of get_j
(df_jk.RIJ_ENGINE = False): the derivative J matrices, the auxiliary-center terms, and the time of each.'''
import time, numpy
from pyscf import gto, dft, df, scf, lib
from pyscf.df import df_jk
from pyscf.df.grad import rhf as dfgrad
def both(mf, dm):
    g = mf.Gradients()
    out = []
    for engine in (True, False):
        df_jk.RIJ_ENGINE = engine
        t = time.perf_counter(); v = dfgrad.get_j(g, mf.mol, dm); out.append((v, time.perf_counter() - t))
    df_jk.RIJ_ENGINE = True
    return out
mols = {
 'H2PO4-': gto.M(atom='bench/diag/wb97mv/h2po4_opt.xyz', charge=-1, verbose=0),
 'glycine': gto.M(atom='bench/diag/wb97mv/glycine_opt.xyz', verbose=0),
}
for name, mol0 in mols.items():
    for basis, aux in (('def2-svp', 'def2-universal-jfit'), ('def2-tzvpd', 'aug_etb'), ('def2-qzvp', 'def2-universal-jfit')):
        mol = mol0.copy(); mol.basis = basis; mol.build()
        auxbasis = df.aug_etb(mol) if aux == 'aug_etb' else aux
        mf = dft.RKS(mol, xc='pbe').density_fit(auxbasis=auxbasis); mf.kernel()
        dm = mf.make_rdm1()
        (a, ta), (b, tb) = both(mf, dm)
        e = numpy.einsum('xij,ij->x', a, dm)
        print(f'{name:8s} {basis:11s} {aux:20s} nao {mol.nao:4d} naux {mf.with_df.auxmol.nao:5d}: vj max|engine - libcint| '
              f'{abs(a - b).max():.1e} (max {abs(b).max():.2f}), aux part {abs(a.aux - b.aux).max():.1e} (max {abs(b.aux).max():.2f}); '
              f'{ta:.2f} s against {tb:.2f} s', flush=True)
# two densities (UKS), and without the auxiliary response
mol = gto.M(atom='bench/diag/wb97mv/h2po4_opt.xyz', charge=0, spin=1, basis='def2-tzvp', verbose=0)
mf = dft.UKS(mol, xc='pbe').density_fit(); mf.kernel(); dm = mf.make_rdm1()
(a, ta), (b, tb) = both(mf, dm)
print(f'UKS H2PO4 def2-tzvp: vj {a.shape} max|d| {abs(a - b).max():.1e}, aux {a.aux.shape} {abs(a.aux - b.aux).max():.1e}')
g = mf.Gradients(); g.auxbasis_response = False
v = dfgrad.get_j(g, mol, dm[0]); df_jk.RIJ_ENGINE = False; w = dfgrad.get_j(g, mol, dm[0]); df_jk.RIJ_ENGINE = True
print(f'no auxiliary response: max|d| {abs(v - w).max():.1e}, tagged {hasattr(v, "aux")}')
# whole gradients, RIJCOSX and density fitting, engine against libcint
for label, make in (('RKS RIJCOSX', lambda m: dft.RKS(m, xc='b3lyp').RIJCOSX()), ('RHF density_fit', lambda m: scf.RHF(m).density_fit()),
                    ('UKS RIJCOSX', lambda m: dft.UKS(m, xc='pbe0').RIJCOSX())):
    mol = gto.M(atom='bench/diag/wb97mv/h2po4_opt.xyz', charge=0 if 'UKS' in label else -1, spin=1 if 'UKS' in label else 0,
                basis='def2-tzvp', verbose=0)
    mf = make(mol); mf.kernel()
    ga = mf.Gradients().kernel(); df_jk.RIJ_ENGINE = False; gb = mf.Gradients().kernel(); df_jk.RIJ_ENGINE = True
    print(f'{label:16s} gradient max|engine - libcint| {abs(ga - gb).max():.1e} (max {abs(gb).max():.2e})', flush=True)
