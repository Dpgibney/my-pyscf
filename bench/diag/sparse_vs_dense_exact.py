"""Sparse-path Vxc and gradient XC term vs the same build forced dense (no screening at all)."""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build(with_non0tab=True)
mf = dft.RKS(mol)
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
ni = numint.NumInt()
xc = 'PBE'
v_sp = ni.nr_rks(mol, grids, xc, dm)[2]
g_sp = rks_grad.get_vxc(ni, mol, grids, xc, dm)[1]
orig = numint._sparse_enough
numint._sparse_enough = lambda *a, **k: False   # every block takes the dense path
v_de = ni.nr_rks(mol, grids, xc, dm)[2]
g_de = rks_grad.get_vxc(ni, mol, grids, xc, dm)[1]
rel = lambda a, b: abs(a - b).max() / abs(b).max()
print(f'sparse path vs dense: Vxc {rel(v_sp, v_de):.1e}, gradient XC term {rel(g_sp, g_de):.1e}')
