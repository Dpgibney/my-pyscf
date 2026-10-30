"""In-memory J/K (_vhf.incore, s8 ERIs) of n symmetric dms (hermi=1), as UKS/ROKS SCFs pass 2: the
tree's own routing, the one-pass single-dm kernel once per dm, and the BLAS batch kernels; best of REPS
calls each, interleaved, with the max relative difference from the per-dm kernels of nr_incore.c.
Usage: incore_ndm_time.py SYSTEM [REPS] [NDM ...]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE
from pyscf import gto, lib, scf
from pyscf.scf import _vhf

which = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ndms = [int(x) for x in sys.argv[3:]] or [1, 2, 3, 4, 6, 8]
atom, charge = {'benzene+': (BENZENE, 1), 'caffeine+': (CAFFEINE, 1)}[which]
mol = gto.M(atom=atom, basis='def2-svp', charge=charge, spin=1, verbose=0)
nao = mol.nao
t0 = time.perf_counter()
eri = mol.intor('int2e', aosym='s8')
print(f'{which} nao={nao} ERIs {eri.nbytes / 1e9:.2f} GB in {time.perf_counter() - t0:.1f} s, '
      f'threads={lib.num_threads()}', flush=True)
# UHF-like densities: the minao guess per spin, then perturbed copies for n > 2
dm_ab = scf.UHF(mol).get_init_guess(key='minao')
rng = np.random.default_rng(5)
dms = [dm_ab[i % 2] for i in range(max(ndms))]
for i in range(2, len(dms)):
    a = rng.standard_normal((nao, nao)) * 1e-2
    dms[i] = dms[i] + a + a.T
dms = np.asarray(dms)


def per_dm():
    out = [_vhf.incore(eri, d, 1) for d in dms_n]
    return np.asarray([o[0] for o in out]), np.asarray([o[1] for o in out])


def batch():
    with lib.temporary_env(_vhf, INCORE_FUSED_MAX=0):
        return _vhf._incore_s8_batch(eri, dms_n, 1, True, True)


def tree():
    return _vhf.incore(eri, dms_n, 1)


t0 = time.perf_counter()
with lib.temporary_env(_vhf, INCORE_BATCH_MIN=10**9):
    vj_ref, vk_ref = _vhf.incore(eri, dms, 1)
print(f'reference (per-dm kernels of nr_incore.c, {len(dms)} dms) {time.perf_counter() - t0:.1f} s', flush=True)
print(f'{"n_dm":>4} {"tree (s)":>9} {"per dm":>8} {"batch":>8} {"tree J err":>10} {"tree K err":>10} '
      f'{"tree vs per dm":>15}')
for n in ndms:
    dms_n = dms[:n]
    fns = {'tree': tree, 'per_dm': per_dm, 'batch': batch}
    ts = {k: [] for k in fns}
    out = {}
    for rep in range(reps):
        for k in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
            t0 = time.perf_counter()
            out[k] = fns[k]()
            ts[k].append(time.perf_counter() - t0)
    vj0, vk0 = vj_ref[:n], vk_ref[:n]
    ej = abs(out['tree'][0] - vj0).max() / abs(vj0).max()
    ek = abs(out['tree'][1] - vk0).max() / abs(vk0).max()
    same = all(np.array_equal(a, b) for a, b in zip(out['tree'], out['per_dm']))
    print(f'{n:4d} {min(ts["tree"]):9.3f} {min(ts["per_dm"]):8.3f} {min(ts["batch"]):8.3f} '
          f'{ej:10.1e} {ek:10.1e} {"bit for bit" if same else "DIFFER":>15}', flush=True)
