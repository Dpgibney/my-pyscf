import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, lib, dft
from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess
mol = gto.M(atom=molecules.CAFFEINE, basis='def2-svp', verbose=0)
aoslices = mol.aoslice_by_atom()
nao, natm, ng = mol.nao, mol.natm, 67200
grids = dft.gen_grid.Grids(mol); grids.build(with_non0tab=True)
coords = grids.coords[:ng]
ao = numint.eval_ao(mol, coords, deriv=2)        # layout as in block_loop: each comp (ng,nao), grid index contiguous
print('ao comp flags F:', ao[0].flags.f_contiguous, 'shape', ao.shape)
rng = numpy.random.default_rng(0)
dm0 = rng.standard_normal((nao, nao)); dm0 = dm0 + dm0.T
ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, None, (0, mol.nbas), mol.ao_loc_nr()) for i in range(4)]
print('ao_dm0 flags F:', ao_dm0[0].flags.f_contiguous)
wf = rng.standard_normal((4, 4, ng))
def tm(f, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r
t_aps, ref = tm(lambda: rks_hess._atom_pair_sum(aoslices, ao[1], ao_dm0[0]))
def mk():
    out = []
    for ia in range(natm):
        r = rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, 'GGA')
        out.append(numpy.einsum('xyg,sxg->syg', wf, r))
    return out
t_mk, _ = tm(mk, 2)
def mk_only():
    for ia in range(natm):
        rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices, 'GGA')
t_mko, _ = tm(mk_only, 2)
print(f'_atom_pair_sum {t_aps*1e3:.2f} ms/call; natm x _make_dR_rho1 {t_mko*1e3:.0f} ms; + einsum xyg,sxg->syg {(t_mk-t_mko)*1e3:.0f} ms (per 67200-pt block)')
# all-atoms alternative for one pair sum: elementwise product then segmented column sums
starts = aoslices[:, 2]
def alt2():
    return numpy.add.reduceat(ao[1] * ao_dm0[0], starts, axis=1).T
t_alt2, r2 = tm(alt2)
print(f'  alt reduceat {t_alt2*1e3:.2f} ms, rel diff {abs(r2-ref).max()/abs(ref).max():.1e}')
M = numpy.zeros((natm, nao))
for ia, (p0, p1) in enumerate(aoslices[:, 2:]):
    M[ia, p0:p1] = 1
def alt3():
    return lib.dot(M, (ao[1] * ao_dm0[0]).T)
t_alt3, r3 = tm(alt3)
print(f'  alt product+GEMM {t_alt3*1e3:.2f} ms, rel diff {abs(r3-ref).max()/abs(ref).max():.1e}')
nblk = 5
print(f'per caffeine GGA Hessian (5 blocks): 21 pair sums/block in deriv2_dm = {nblk*21*t_aps:.2f} s; dR_rho1+einsum in deriv2_dm and deriv1 = {2*nblk*t_mk:.2f} s')
