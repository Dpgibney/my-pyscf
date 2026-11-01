"""The two-electron derivative builds of GPU4PySCF at a fixed density, for comparing builds (node1):
  ejk1  gradient energy, grad.rhf jk_energy_per_atom (RYS_per_atom_jk_ip1): J + K, B3LYP's J + 0.2 K, J alone,
        and wB97X's short-range K (omega -0.3)
  ejk2  Hessian energy second derivatives, hessian.rhf._partial_ejk_ip2 (RYS_per_atom_jk_ip2_type12/type3)
  jk1   J and K derivative matrices, hessian.rhf._get_jk_ip1 (RYS_build_jk_ip1)
Each is built once untimed, then REPEAT times between device synchronizations (with the GPU clocks); the results go
to OUT.npz, and --compare REF.npz prints the largest differences. DM.npy is a density in the original AO basis
(jk_at_dm.py --save-dm). Source scripts/node1-gpu-env.sh perf|base first.
--parts takes families (ejk1, ejk2, jk1) or single builds (ejk1_b3lyp, ejk1_srk, ...).
Usage: deriv_at_dm.py --dm DM.npy [--parts ejk1,ejk2,jk1] [--basis def2-svp] [--repeat 3] [--out OUT.npz]
                      [--compare REF.npz]"""
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
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--parts', default='ejk1,ejk2,jk1')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--out', default=None)
ap.add_argument('--compare', default=None)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = cupy.asarray(numpy.load(args.dm))
g = dft.RKS(mol, xc='b3lyp').nuc_grad_method()
builds = {
    'ejk1_jk': lambda: g.jk_energy_per_atom(dm, 1., 1.),
    'ejk1_b3lyp': lambda: g.jk_energy_per_atom(dm, 1., .2),
    'ejk1_j': lambda: g.jk_energy_per_atom(dm, 1., 0.),
    'ejk1_srk': lambda: g.jk_energy_per_atom(dm, 0., .5, omega=-.3),
    'ejk2': lambda: rhf_hess._partial_ejk_ip2(mol, dm, j_factor=1., k_factor=.2),
    'jk1': lambda: cupy.stack(rhf_hess._get_jk_ip1(mol, dm)),
}
parts = args.parts.split(',')
res = {}
for name, build in builds.items():
    if name not in parts and name.split('_')[0] not in parts:
        continue
    v = build()
    ts = []
    with GpuMonitor() as mon:
        for _ in range(args.repeat):
            cupy.cuda.Device().synchronize()
            t0 = time.perf_counter()
            v = build()
            cupy.cuda.Device().synchronize()
            ts.append(time.perf_counter() - t0)
    res[name] = cupy.asnumpy(v)
    print(f'{os.path.dirname(gpu4pyscf.__file__)}: {name:11s} median {numpy.median(ts):7.3f} s of {args.repeat}, '
          f"SM {mon.summary.get('sm_mhz')} MHz", flush=True)
if args.out:
    numpy.savez(args.out, **res)
if args.compare:
    ref = numpy.load(args.compare)
    for name, v in res.items():
        print(f'  {name:11s} max |d - ref| = {abs(v - ref[name]).max():.2e} (max |ref| {abs(ref[name]).max():.2e})')
