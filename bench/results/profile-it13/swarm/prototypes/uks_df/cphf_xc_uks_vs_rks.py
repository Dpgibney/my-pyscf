'''CPHF-shaped XC response, 1 thread: RKS benzene (dm path nr_rks_fxc vs the It13 factored path
rks_fxc_mo) and UKS benzene+ (nr_uks_fxc dm path, the only one), 3*natm sets each.'''
import sys, time
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint

BENZENE = '''C 0.000 1.396 0.000; C 1.209 0.698 0.000; C 1.209 -0.698 0.000;
C 0.000 -1.396 0.000; C -1.209 -0.698 0.000; C -1.209 0.698 0.000;
H 0.000 2.479 0.000; H 2.147 1.240 0.000; H 2.147 -1.240 0.000;
H 0.000 -2.479 0.000; H -2.147 -1.240 0.000; H -2.147 1.240 0.000'''
xc = sys.argv[1] if len(sys.argv) > 1 else 'b3lyp'
rng = numpy.random.default_rng(0)

def best(f, n=2):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r

# RKS
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc=xc).run()
ni, grids = mf._numint, mf.grids
C, occ = mf.mo_coeff, mf.mo_occ
mocc = C[:, occ > 0]
nao, nmo = C.shape; nocc = mocc.shape[1]
nset = 3 * mol.natm
x = rng.standard_normal((nset, nmo, nocc)) * .1
dm1 = numpy.asarray([C @ xi @ mocc.T for xi in x]); dm1 = dm1 + dm1.transpose(0, 2, 1)
rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, C, occ, 0)
mm = 16000
t_dm, v1 = best(lambda: ni.nr_rks_fxc(mol, grids, xc, None, dm1, 0, 1, rho0, vxc, fxc, max_memory=mm))
xctype = ni._xc_type(xc)
fused = numint._fused_xc(ni, mol, grids, nao, xctype, mm, (C,), match_block_loop=False)
def lowrank():
    out = numpy.zeros((nset, nmo, nocc))
    a = lib.einsum('pa,kai->kpi', C, x)
    fused.rks_fxc_mo(ni, xctype, fxc, a, mocc, C, mocc, out)
    return out
t_lr, vmo = best(lowrank)
ref = numpy.asarray([C.T @ v @ mocc for v in v1])
print(f'RKS benzene {xc}, {nset} sets: dm path {t_dm:.2f} s, factored {t_lr:.2f} s '
      f'(cheaper by model: {fused.lowrank_cheaper(xctype, nocc, nocc, nset)}); rel diff {abs(vmo-ref).max()/abs(ref).max():.1e}')

# UKS
molu = gto.M(atom=BENZENE, basis='def2-svp', charge=1, spin=1, verbose=0)
mfu = dft.UKS(molu, xc=xc).run()
Cu, occu = mfu.mo_coeff, mfu.mo_occ
rho0, vxc, fxc = ni.cache_xc_kernel(molu, mfu.grids, xc, Cu, occu, 1)
dmu = []
for s in range(2):
    mo = Cu[s][:, occu[s] > 0]
    xs = rng.standard_normal((nset, nmo, mo.shape[1])) * .1
    d = numpy.asarray([Cu[s] @ xi @ mo.T for xi in xs]); dmu.append(d + d.transpose(0, 2, 1))
dmu = numpy.asarray(dmu)
t_u, _ = best(lambda: mfu._numint.nr_uks_fxc(molu, mfu.grids, xc, None, dmu, 0, 1, rho0, vxc, fxc, max_memory=mm))
print(f'UKS benzene+ {xc}, 2x{nset} sets: dm path {t_u:.2f} s = {t_u/t_dm:.2f}x the RKS dm path')
