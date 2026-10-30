"""1-thread probe: grid pipeline pieces (eval_ao, eval_rho) and fused-kernel group sizes."""
import sys, time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE

def tm(f, n=3):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()
dm = dft.RKS(mol).get_init_guess(key='minao')
nao = mol.nao
N = 50000
coords = grids.coords[:N]
for deriv in ():
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    buf = np.empty(comp*N*nao)
    t_fresh = tm(lambda: numint.eval_ao(mol, coords, deriv=deriv))
    t_buf = tm(lambda: numint.eval_ao(mol, coords, deriv=deriv, out=buf))
    gb = comp*N*nao*8/1e9
    print('eval_ao deriv=%d: fresh %.3f s, reused out %.3f s (%.2f GB)' % (deriv, t_fresh, t_buf, gb))

ao = numint.eval_ao(mol, coords, deriv=1)
# exact zeros per (56-point block, shell), any component
ao_loc = mol.ao_loc_nr()
nblk = N // BLKSIZE
z = np.abs(ao[:, :nblk*BLKSIZE]).reshape(4, nblk, BLKSIZE, nao).max(axis=(0, 2))  # [nblk, nao]
shl_nz = np.maximum.reduceat(z, ao_loc[:-1], axis=1) > 0
print('exact-zero (block, shell) fraction: %.4f; AO-weighted nonzero fraction %.4f' %
      (1 - shl_nz.mean(), (shl_nz * np.diff(ao_loc)).sum() / (nblk * nao)))
from pyscf.gto.eval_gto import make_screen_index; sidx = make_screen_index(mol, coords, cutoff=grids.cutoff)
print('screen index (cutoff %.0e) nonzero fraction %.4f' % (grids.cutoff, np.count_nonzero(sidx) / sidx.size))

dmh = (dm + dm.T) * .5
shls = (0, mol.nbas)
t_gemm = tm(lambda: numint._dot_ao_dm(mol, ao[0], dmh, None, shls, ao_loc))
c0 = numint._dot_ao_dm(mol, ao[0], dmh, None, shls, ao_loc)
t_con = tm(lambda: numint._contract_rho(ao[1], c0))
t_rho = tm(lambda: numint.eval_rho(mol, ao, dm, xctype='GGA'))
print('eval_rho GGA %.3f s: GEMM %.3f s, one _contract_rho %.3f s (x4 = %.3f)' %
      (t_rho, t_gemm, t_con, 4 * t_con))
gflop = 2 * N * nao**2 / 1e9
print('GEMM rate 1 thread: %.1f GFLOP/s' % (gflop / t_gemm))

# fused-kernel group sizes on the SCF grid (default level, non0tab)
mf = dft.RKS(mol, xc='PBE')
mf.grids.build(with_non0tab=True)
d = numint._GridAOData(mol, mf.grids)
ns = d.group_naos(mol)
print('SCF grid %d points, %d groups; ns per group: mean %.1f, rms %.1f, max %.0f' %
      (mf.grids.weights.size, ns.size, ns.mean(), np.sqrt((ns**2).mean()), ns.max()))
nocc = mol.nelectron // 2
ng = 224
for label, c in (('MO path (4 GEMM)', 4 * 2 * ng * ns * nocc), ('MO + Z (2 GEMM)', 2 * 2 * ng * ns * nocc),
                 ('dm path (1 GEMM)', 2 * ng * ns**2), ('dot_wv GGA GEMM', 2 * ng * ns**2)):
    print('  %-18s %.1f GFLOP per SCF cycle' % (label, c.sum() / 1e9))
