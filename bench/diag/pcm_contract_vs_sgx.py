"""pcm._int1e_grids_dot through the contracted kernels (pyscf and qcint branches pcm-kernel) against the SGX path, and
their times: argv n, basis, sph|cart. Needs both branches built (LD_LIBRARY_PATH to the qcint build)."""
import sys, time
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import numpy
from molecules import n_alkane
from pyscf import gto, dft, lib
from pyscf.solvent import pcm
n, basis, cart = int(sys.argv[1]), sys.argv[2], sys.argv[3] == 'cart'
mol = gto.M(atom=n_alkane(n), basis=basis, cart=cart, verbose=0)
p = pcm.PCM(mol); p.build()
rng = numpy.random.default_rng(5)
dm = dft.RKS(mol).get_init_guess() + 0.01 * rng.normal(size=(mol.nao, mol.nao))
q = p._get_q(p.v_grids_n - p._get_v(dm[None]))[1]
res = {}
for flag in (False, True):
    pcm.INT1E_GRIDS_CONTRACT = flag
    p._get_v(dm[None]); p._get_vmat(q)
    t0 = time.perf_counter(); v = p._get_v(dm[None]); t1 = time.perf_counter(); V = p._get_vmat(q); t2 = time.perf_counter()
    res[flag] = (v, V, t1 - t0, t2 - t1)
(v0, V0, a0, b0), (v1, V1, a1, b1) = res[False], res[True]
print(f'C{n}H{2*n+2}/{basis}{" cart" if cart else ""}: {mol.nao} AOs, {q.shape[1]} points, {lib.num_threads()} threads: '
      f'_get_v {a0:.3f} -> {a1:.3f} s, _get_vmat {b0:.3f} -> {b1:.3f} s; max |dv| {abs(v1-v0).max():.1e} '
      f'(max |v| {abs(v0).max():.1e}), max |dV| {abs(V1-V0).max():.1e} (max |V| {abs(V0).max():.1e})', flush=True)
