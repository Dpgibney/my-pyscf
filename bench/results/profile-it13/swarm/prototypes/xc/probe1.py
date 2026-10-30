"""1-thread probe: libxc per-point cost on real vs random densities; per-call setup overheads."""
import sys, time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint, libxc

def tm(f, n=5):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
print('nao', mol.nao, 'nbas', mol.nbas, 'threads', lib.num_threads())
mf = dft.RKS(mol, xc='B3LYP')
t0 = time.perf_counter(); mf.grids.build(with_non0tab=True)
print('grids.build %.3f s, ngrids %d' % (time.perf_counter() - t0, mf.grids.weights.size))
ni = mf._numint
dm = mf.get_init_guess(key='minao')
# orbitals for a tagged (MO) dm: natural orbitals of the minao dm in the S metric
s = mol.intor('int1e_ovlp')
from scipy.linalg import eigh
w, c = eigh(s @ dm @ s, s)
idx = np.argsort(-w)
w, c = w[idx], c[:, idx]
occ = np.zeros_like(w); occ[:mol.nelectron // 2] = 2
dm_mo = lib.tag_array(c[:, occ > 0] @ c[:, occ > 0].T * 2, mo_coeff=c, mo_occ=occ)

print('get_overlap_cond   %.3f ms' % (1e3 * tm(lambda: mol.get_overlap_cond())))
print('condense_to_shell  %.3f ms' % (1e3 * tm(lambda: mol.condense_to_shell(dm_mo, 'absmax'))))
print('condense plain arr %.3f ms' % (1e3 * tm(lambda: mol.condense_to_shell(np.asarray(dm_mo), 'absmax'))))
print('_rho_setup         %.3f ms' % (1e3 * tm(lambda: numint._rho_setup(ni, mol, dm_mo, 1, mf.grids))))
r = numint._rho_setup(ni, mol, dm_mo, 1, mf.grids)
print('rho1_rho2_ratio %.3f (GGA uses dm path if < 4, MGGA if < 1)' % r[-1])

# one block of 67200 points from the middle of the grid
ng = 67200
i0 = (mf.grids.weights.size // 2 // 56) * 56 - ng // 2
coords = mf.grids.coords[i0:i0 + ng]
ao = ni.eval_ao(mol, coords, deriv=1)
rho_g = numint.eval_rho(mol, ao, dm, xctype='GGA', hermi=1)
rho_m = numint.eval_rho(mol, ao, dm, xctype='MGGA', hermi=1, with_lapl=False)
del ao
print('rho range: min %.2e, median %.2e, frac<1e-10 %.3f' %
      (rho_g[0].min(), np.median(rho_g[0]), np.mean(rho_g[0] < 1e-10)))

rng = np.random.default_rng(1)
rnd = np.empty((5, ng))
rnd[0] = rng.random(ng) * 2 + 1e-4
rnd[1:4] = rng.standard_normal((3, ng)) * .3
rnd[4] = rng.random(ng) * 2 + 1e-3

for xc, rho_real, rr in (('B3LYP', rho_g, rnd[:4]), ('PBE', rho_g, rnd[:4]),
                         ('r2SCAN', rho_m, rnd), ('LDA,VWN', rho_g[0], rnd[0])):
    rr = np.ascontiguousarray(rr)
    t_real = tm(lambda: libxc._eval_xc(xc, rho_real, 0, deriv=1))
    t_rnd = tm(lambda: libxc._eval_xc(xc, rr, 0, deriv=1))
    t_eff = tm(lambda: ni.eval_xc_eff(xc, rho_real, deriv=1, xctype=ni._xc_type(xc)))
    print('%-8s _eval_xc real %.1f ns/pt, random %.1f ns/pt; eval_xc_eff real %.1f ns/pt' %
          (xc, t_real / ng * 1e9, t_rnd / ng * 1e9, t_eff / ng * 1e9))
