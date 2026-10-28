"""Fused/rerouted sparse paths vs the unfused ones and vs dense products, on real alkane AO data."""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE, NBINS

mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids.build(with_non0tab=True)
ni = mf._numint
ao_loc = mol.ao_loc_nr()
nao, nbas = mol.nao, mol.nbas
nblk = 1200
b0 = (mf.grids.weights.size // BLKSIZE) // 2 - nblk // 2
ip0, ip1 = b0 * BLKSIZE, (b0 + nblk) * BLKSIZE
ngrids = ip1 - ip0
mask = np.ascontiguousarray(mf.grids.non0tab[b0:b0 + nblk])
ao = ni.eval_ao(mol, mf.grids.coords[ip0:ip1], deriv=2, non0tab=mask, cutoff=mf.grids.cutoff)
ao1 = np.ascontiguousarray(ao[:4].transpose(0, 2, 1)).transpose(0, 2, 1)  # deriv=1 layout
dm = mf.get_init_guess(key='minao')
rng = np.random.default_rng(3)
wv = rng.standard_normal((5, ngrids)) * mf.grids.weights[ip0:ip1]
pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
nbins = 100
rel = lambda a, b: abs(a - b).max() / abs(b).max()
ok = True

def check(label, err, tol=1e-13):
    global ok
    ok &= err < tol
    print(f'{"ok  " if err < tol else "FAIL"} {label:52s} {err:.1e}')

# eval_rho1: hermi=1 (fused kernel) vs hermi=0 (unfused kernels; same result for symmetric dm)
for xctype, a in (('LDA', ao1[0]), ('GGA', ao1), ('MGGA', ao1)):
    new = numint.eval_rho1(mol, a, dm, mask, xctype, hermi=1, with_lapl=False)
    old = numint.eval_rho1(mol, a, dm, mask, xctype, hermi=0, with_lapl=False)
    dense = numint.eval_rho(mol, a, dm, None, xctype, hermi=1, with_lapl=False)
    check(f'eval_rho1 {xctype} fused vs unfused', rel(new, old))
    check(f'eval_rho1 {xctype} fused vs dense eval_rho', rel(new, dense))
# eval_rho2 through _dot_ao_dm with the screen index vs without
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
mo_occ = mf.get_occ(mo_e, mo_c)
for xctype, a in (('LDA', ao1[0]), ('GGA', ao1), ('MGGA', ao1)):
    new = numint.eval_rho2(mol, a, mo_c, mo_occ, mask, xctype, with_lapl=False)
    dense = numint.eval_rho2(mol, a, mo_c, mo_occ, None, xctype, with_lapl=False)
    check(f'eval_rho2 {xctype} screened vs dense', rel(new, dense))
# GGA XC matrix: fused scale+dot vs _scale_ao_sparse + _dot_ao_ao_sparse
out_new = np.zeros((nao, nao))
numint._dot_ao_aow_sparse(ao1, wv[:4], nbins, mask, pair_mask, ao_loc, out_new)
aow = numint._scale_ao_sparse(ao1, wv[:4], mask, ao_loc)
out_old = numint._dot_ao_ao_sparse(ao1[0], aow, None, nbins, mask, pair_mask, ao_loc)
check('_dot_ao_aow_sparse vs scale + dot', rel(out_new, out_old))
check('_dot_ao_aow_sparse vs dense', rel(out_new, ao1[0].T @ numint._scale_ao(ao1, wv[:4])))
# tau term: one stacked dsyrk vs three products
t_new = numint._tau_dot_sparse(ao1, ao1, wv[4], nbins, mask, pair_mask, ao_loc,
                               out=np.zeros((nao, nao)))
t_ref = sum((ao1[c].T * wv[4]) @ ao1[c] for c in (1, 2, 3))
check('_tau_dot_sparse stacked dsyrk vs dense', rel(t_new, t_ref))
# _dot_ao_ao / _dot_ao_dm with a screen index (gradients, Hessians, eval_rho2)
shls = (0, nbas)
for c in (1, 4):
    aow = numint._scale_ao(ao[0], wv[0])
    v = numint._dot_ao_ao(mol, ao[c], aow, mask, shls, ao_loc)
    check(f'_dot_ao_ao(ao[{c}], aow) screened vs dense', rel(v, ao[c].T @ aow))
v = numint._dot_ao_ao(mol, ao[1], ao[1], mask, shls, ao_loc, hermi=1)
check('_dot_ao_ao hermi=1 same ao (dsyrk) vs dense', rel(v, ao[1].T @ ao[1]))
cocc = mo_c[:, mo_occ > 0]
v = numint._dot_ao_dm(mol, ao[2], cocc, mask, shls, ao_loc)
check('_dot_ao_dm(ao, C_occ) screened vs dense', rel(v, ao[2] @ cocc))
v = numint._dot_ao_dm(mol, ao[0], dm, mask, shls, ao_loc)
check('_dot_ao_dm(ao, dm) screened vs dense', rel(v, ao[0] @ dm))
print('ALL OK' if ok else 'SOME FAILED')
