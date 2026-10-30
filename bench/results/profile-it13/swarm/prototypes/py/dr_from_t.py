import sys, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess
mol = gto.M(atom=molecules.CAFFEINE, basis='def2-svp', verbose=0)
aoslices = mol.aoslice_by_atom(); natm = mol.natm
grids = dft.gen_grid.Grids(mol); grids.build()
ao = numint.eval_ao(mol, grids.coords[100000:120000], deriv=2)
rng = numpy.random.default_rng(1)
dm0 = rng.standard_normal((mol.nao, mol.nao)); dm0 = dm0 + dm0.T
ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, None, (0, mol.nbas), mol.ao_loc_nr()) for i in range(4)]
D2 = rks_hess._D2
for xctype, ncomp in (('GGA', 4), ('MGGA', 5)):
    t = numpy.empty((natm, 3, ncomp, ao.shape[1]))
    for d in range(3):
        t[:,d,0] = rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
        for c in range(1, 4):
            t[:,d,c] = rks_hess._atom_pair_sum(aoslices, ao[D2[d][c-1]], ao_dm0[0])
            t[:,d,c] += rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
        if xctype == 'MGGA':
            t[:,d,4] = sum(rks_hess._atom_pair_sum(aoslices, ao[D2[x][d]], ao_dm0[1+x]) for x in range(3))
    dR_t = t.copy(); dR_t[:,:,1:4] *= 2
    dR = numpy.array([rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype) for ia in range(natm)])
    diff = abs(dR - dR_t).max(); scale = abs(dR).max()
    print(f'{xctype}: max |dR_rho1 - dR(t)| = {diff:.2e} (scale {scale:.2e}), bitwise equal: {numpy.array_equal(dR, dR_t)}')
