"""1-thread probe: VV10 (_vv10nlc) cost per pair on caffeine's default NLC grid (level 3)."""
import sys, time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint

mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='wB97M-V')
print('do_nlc', mf.do_nlc(), 'nlcgrids level', mf.nlcgrids.level)
mf.nlcgrids.build(with_non0tab=True)
g = mf.nlcgrids
print('NLC grid points', g.weights.size)
dm = mf.get_init_guess(key='minao')
ni = mf._numint
rho = []
for ao, mask, weight, coords in ni.block_loop(mol, g, mol.nao, 1, max_memory=2000):
    rho.append(numint.eval_rho(mol, ao, dm, mask, xctype='GGA', hermi=1))
rho = np.hstack(rho)
keep = rho[0] >= 1e-8
print('points with rho >= 1e-8: %d (%.3f)' % (keep.sum(), keep.mean()))
nlc_pars = ni.nlc_coeff(mf.xc)[0][0]
nout = 1500
sel = np.nonzero(keep)[0][::max(1, keep.sum() // nout)][:nout]
sub_rho = rho[:, sel]
t0 = time.perf_counter()
numint._vv10nlc(sub_rho, g.coords[sel], rho, g.weights, g.coords, nlc_pars)
dt = time.perf_counter() - t0
npairs = len(sel) * keep.sum()
print('subset: %d outer x %d inner points: %.2f s, %.2f ns/pair' % (len(sel), keep.sum(), dt, dt / npairs * 1e9))
print('full call estimate at 1 thread: %.0f s (%.2e pairs)' % (keep.sum()**2 * dt / npairs, keep.sum()**2))
