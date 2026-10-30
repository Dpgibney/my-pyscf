'''Time the numpy (einsum) parts of the GGA Hessian XC terms that run per atom and grid block, on one
caffeine block (67200 points, nao 246, 24 atoms) with synthetic AO values: _make_dR_rho1 + the wv
einsum (both in _get_vxc_deriv1 and _get_vxc_deriv2_dm), and the t[] _atom_pair_sum calls
(_get_vxc_deriv2_dm only). Single thread (einsum is single-threaded anyway).'''
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, lib
from pyscf.hessian import rks as rks_hess

mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
nao, natm = mol.nao, mol.natm
ng = 67200
rng = numpy.random.default_rng(0)
from pyscf.dft import numint, gen_grid
grids = gen_grid.Grids(mol); grids.level = 3; grids.build()
coords = grids.coords[100000:100000+ng]
ao = numint.eval_ao(mol, coords, deriv=2)       # real layout of block_loop's AO blocks
dm0 = rng.standard_normal((nao, nao)); dm0 = dm0 + dm0.T
ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, None, (0, mol.nbas), mol.ao_loc_nr()) for i in range(4)]
print('ao[0] strides', ao[0].strides, 'ao_dm0 strides', ao_dm0[0].strides)
wf = rng.standard_normal((4, 4, ng))
aoslices = mol.aoslice_by_atom()

t0 = time.perf_counter()
wv = numpy.empty((natm, 3, 4, ng))
tm = 0.
for ia in range(natm):
    c0 = time.perf_counter()
    dR_rho1 = rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, 'GGA')
    tm += time.perf_counter() - c0
    wv[ia] = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
t1 = time.perf_counter()
t = numpy.empty((natm, 3, 4, ng))
for d in range(3):
    t[:,d,0] = rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
    for c in range(1, 4):
        t[:,d,c] = rks_hess._atom_pair_sum(aoslices, ao[rks_hess._D2[d][c-1]], ao_dm0[0])
        t[:,d,c] += rks_hess._atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
t2 = time.perf_counter()
r = rks_hess._atom_pair_dot(wv, t)
t3 = time.perf_counter()
print(f'per block: _make_dR_rho1 x{natm} {tm:.3f} s, + wv einsum {t1-t0-tm:.3f} s; '
      f'_atom_pair_sum (t) {t2-t1:.3f} s; _atom_pair_dot {t3-t2:.3f} s')

# a reformulation of rho1: all atoms at once with one elementwise product per component pair and a
# segmented sum over the AO axis (numpy.add.reduceat), as a cheap stand-in for a C kernel
t4 = time.perf_counter()
starts = aoslices[:,2]
def seg(x, y):
    return numpy.add.reduceat(x * y, starts, axis=1).T   # [natm, ng]
XX, XY, XZ, YY, YZ, ZZ = 4, 5, 6, 7, 8, 9
D2 = ((XX, XY, XZ), (XY, YY, YZ), (XZ, YZ, ZZ))
rho1 = numpy.empty((natm, 3, 4, ng))
for x in range(3):
    rho1[:,x,0] = seg(ao[1+x], ao_dm0[0])
    for c in range(3):
        rho1[:,x,1+c] = seg(ao[D2[x][c]], ao_dm0[0]) + seg(ao[1+x], ao_dm0[1+c])
rho1 *= 2
wv2 = numpy.einsum('xyg,asxg->asyg', wf, rho1, optimize=True)
t5 = time.perf_counter()
ref = numpy.array([rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, 'GGA') for ia in range(natm)])
print(f'reduceat variant (all atoms): {t5-t4:.3f} s; rho1 max rel diff '
      f'{abs(rho1-ref).max()/abs(ref).max():.1e}; wv max rel diff {abs(wv2-wv).max()/abs(wv).max():.1e}')
