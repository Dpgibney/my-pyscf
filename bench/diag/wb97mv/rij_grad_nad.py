'''The RI-J gradient (df/grad/rhf.py get_j) of NAD+ (1916 AOs, aug_etb: 9944 functions) from a saved density matrix
(PREFIX_dm.npy of prod_scf_grad.py --rijcosx): by the J-engine, twice (the first with the engine's data), and by the
passes over libcint's integrals (df_jk.RIJ_ENGINE = False). Usage: rij_grad_nad.py DM.npy'''
import sys, time, numpy
from pyscf import gto, dft, df, lib
from pyscf.df import df_jk
from pyscf.df.grad import rhf as dfgrad
mol = gto.M(atom='nad+_mmff.xyz', basis='def2-tzvpd', charge=-1, verbose=0)
dm = numpy.load(sys.argv[1])
mf = dft.RKS(mol, xc='wb97m-v').RIJCOSX(auxbasis=df.aug_etb(mol))
g = mf.Gradients()
def rss():
    return [int(l.split()[1]) / 1048576 for l in open('/proc/self/status') if l.startswith('VmHWM')][0]
res = []
for engine in (True, True, False):
    df_jk.RIJ_ENGINE = engine
    t = time.perf_counter(); v = dfgrad.get_j(g, mol, dm); dt = time.perf_counter() - t
    print(f'{"J-engine" if engine else "libcint"}: {dt:.1f} s, peak resident so far {rss():.1f} GB', flush=True)
    res.append(v)
a, b = res[1], res[2]
aos = mol.aoslice_by_atom()
def force(v):
    return numpy.array([numpy.einsum('xij,ij->x', v[:, p0:p1], dm[p0:p1]) * 2 for p0, p1 in aos[:, 2:]]) + v.aux[0, 0] if v.aux.ndim == 4 else None
print(f'vj max|engine - libcint| {abs(a - b).max():.1e} (max {abs(b).max():.1f}), aux part {abs(a.aux - b.aux).max():.1e} '
      f'(max {abs(b.aux).max():.2e}); J gradient per atom max|d| {abs(force(a) - force(b)).max():.1e} (max {abs(force(b)).max():.2f})')
