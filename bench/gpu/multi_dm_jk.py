"""J and K for many densities at once, as the Hessian's CPHF builds them (hessian.rhf._get_jk_mo: 3 * natm densities
per iteration), on node1: the J-engine (j_engine._VHFOpt.get_j), the K kernels (jk._VHFOpt.get_k) and the one-pass
J+K kernels (jk._VHFOpt.get_jk), each timed after one untimed call. The densities are the converged density times
(1 + X_n) with symmetric X_n of the density's sparsity (fixed seed). Prints the largest differences between the
J matrices and between the K matrices of the two paths. Source scripts/node1-gpu-env.sh perf|base first.
Usage: multi_dm_jk.py --dm DM.npy [--basis def2-svp] [--n-dm 72] [--repeat 2]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.scf import j_engine, jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--n-dm', type=int, default=72)
ap.add_argument('--repeat', type=int, default=2)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm0 = numpy.load(args.dm)
rng = numpy.random.default_rng(11)
dms = []
for _ in range(args.n_dm):
    x = rng.uniform(-.5, .5, dm0.shape)
    dms.append(dm0 * (1 + x + x.T))
dms = cupy.asarray(numpy.array(dms))
tol = 1e-13
jopt = j_engine._VHFOpt(mol, tol).build()
kopt = jk._VHFOpt(mol, tol).build()
jdms = jopt.apply_coeff_C_mat_CT(dms)
kdms = kopt.apply_coeff_C_mat_CT(dms)
builds = {
    'J-engine J': lambda: jopt.apply_coeff_CT_mat_C(jopt.get_j(jdms, 0)),
    'K kernels K': lambda: kopt.apply_coeff_CT_mat_C(kopt.get_k(kdms, 1, 0, 0., 1., 1.)),
    'one-pass J+K': lambda: [kopt.apply_coeff_CT_mat_C(v) for v in kopt.get_jk(kdms, 1, 0)],
}
res = {}
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis}, nao {mol.nao}, {args.n_dm} densities')
for name, build in builds.items():
    res[name] = build()
    ts = []
    with GpuMonitor() as mon:
        for _ in range(args.repeat):
            cupy.cuda.Device().synchronize()
            t0 = time.perf_counter()
            res[name] = build()
            cupy.cuda.Device().synchronize()
            ts.append(time.perf_counter() - t0)
    print(f'  {name:13s} median {numpy.median(ts):7.3f} s of {args.repeat}, SM {mon.summary.get("sm_mhz")} MHz', flush=True)
vj_md, vk = res['J-engine J'], res['K kernels K']
vj_1, vk_1 = res['one-pass J+K']
print(f'  max |J one-pass - J-engine| = {abs(vj_1 - vj_md).max():.2e} (max |J| {abs(vj_md).max():.2e}); '
      f'max |K one-pass - K| = {abs(vk_1 - vk).max():.2e} (max |K| {abs(vk).max():.2e})')
