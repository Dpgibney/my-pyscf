import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, lib
from pyscf.hessian import rks as rks_hess
mol = gto.M(atom=molecules.CAFFEINE, basis='def2-svp', verbose=0)
aoslices = mol.aoslice_by_atom()
nao, natm, ng = mol.nao, mol.natm, 67200
rng = numpy.random.default_rng(0)
ao = rng.standard_normal((10, ng, nao))
ao_dm0 = [rng.standard_normal((ng, nao)) for _ in range(4)]
wf = rng.standard_normal((4, 4, ng))
def tm(f, n=2):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r
# current per-block work in _get_vxc_deriv2_dm (GGA): 21 _atom_pair_sum, natm x (_make_dR_rho1 + einsum)
t_aps, ref = tm(lambda: rks_hess._atom_pair_sum(aoslices, ao[1], ao_dm0[0]))
def mk():
    out = []
    for ia in range(natm):
        r = rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, 'GGA')
        out.append(numpy.einsum('xyg,sxg->syg', wf, r))
    return out
t_mk, _ = tm(mk, 1)
print(f'caffeine block ng={ng}: _atom_pair_sum {t_aps*1e3:.1f} ms (x21 per GGA block = {21*t_aps:.2f} s); natm x (_make_dR_rho1 + einsum xyg,sxg->syg) {t_mk:.2f} s')
# alternative: elementwise product + segmented column sum by GEMM with the atom indicator matrix
M = numpy.zeros((nao, natm))
for ia, (p0, p1) in enumerate(aoslices[:, 2:]):
    M[p0:p1, ia] = 1
def alt():
    return lib.dot(M.T, (ao[1] * ao_dm0[0]).T)
t_alt, r = tm(alt)
print(f'  alternative (x*y) @ M: {t_alt*1e3:.1f} ms, max rel diff {abs(r-ref).max()/abs(ref).max():.1e}')
def alt2():
    return numpy.add.reduceat(ao[1] * ao_dm0[0], aoslices[:, 2], axis=1).T
t_alt2, r2 = tm(alt2)
print(f'  alternative add.reduceat: {t_alt2*1e3:.1f} ms, max rel diff {abs(r2-ref).max()/abs(ref).max():.1e}')
nblk = 5
print(f'per Hessian (5 blocks, _get_vxc_deriv2_dm + _get_vxc_deriv1): pair sums {nblk*21*t_aps:.2f} s, dR_rho1 path {2*nblk*t_mk:.2f} s')
