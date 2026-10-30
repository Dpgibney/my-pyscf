'''_get_vxc_deriv2_dm's t[B,d,c] (hessian/rks.py:578-586) vs _make_dR_rho1 (rks.py:300-342) on one
real AO block: t[:,:,0] == dR_rho1[:,:,0], t[:,:,1:4] == dR_rho1[:,:,1:4]/2, t[:,:,4] == dR_rho1[:,:,4].'''
import sys
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto
from pyscf.dft import numint, gen_grid
from pyscf.hessian import rks as rks_hess
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
grids = gen_grid.Grids(mol); grids.build()
ao = numint.eval_ao(mol, grids.coords[:20000], deriv=2)
rng = numpy.random.default_rng(0)
dm0 = rng.standard_normal((mol.nao, mol.nao)); dm0 = dm0 + dm0.T
ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, None, (0, mol.nbas), mol.ao_loc_nr()) for i in range(4)]
aoslices = mol.aoslice_by_atom()
for xctype, ncomp in (('GGA', 4), ('MGGA', 5)):
    t = numpy.empty((mol.natm, 3, ncomp, ao.shape[1]))
    for d in range(3):
        t[:,d,0] = rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
        for c in range(1, 4):
            t[:,d,c] = rks_hess._atom_pair_sum(aoslices, ao[rks_hess._D2[d][c-1]], ao_dm0[0])
            t[:,d,c] += rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
        if xctype == 'MGGA':
            t[:,d,4] = sum(rks_hess._atom_pair_sum(aoslices, ao[rks_hess._D2[x][d]], ao_dm0[1+x]) for x in range(3))
    r = numpy.array([rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype) for ia in range(mol.natm)])
    scale = numpy.array([1, .5, .5, .5, 1][:ncomp])
    print(xctype, 'max|t - dR_rho1*scale| / max|t| =', abs(t - r*scale[:,None]).max() / abs(t).max())
