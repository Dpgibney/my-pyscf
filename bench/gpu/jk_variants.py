"""J/K builds of GPU4PySCF compared at one converged density (node1): the MD J-engine (get_j, what RKS uses for J),
the Rys K (get_k, what RKS uses for hybrids) and the Rys J+K in one pass (get_jk). Median of REPEAT calls after one
untimed call each, synchronized; the J and K of get_jk are checked against get_j and get_k.
Usage: jk_variants.py [--mol caffeine] [--basis def2-svp] [--xc b3lyp] [--repeat 3]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks as gpu_rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--repeat', type=int, default=3)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = gpu_rks.RKS(mol, xc=args.xc).run()
dm = cupy.asarray(mf.make_rdm1())
print(f'{args.mol} {args.basis} {args.xc}: nao {mol.nao}, E {mf.e_tot:.10f}')


def bench(f):
    out = f()
    ts = []
    for _ in range(args.repeat):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = f()
        cupy.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    return out, float(numpy.median(ts))


vj, t_j = bench(lambda: mf.get_j(mol, dm))
vk, t_k = bench(lambda: mf.get_k(mol, dm))
(vj2, vk2), t_jk = bench(lambda: mf.get_jk(mol, dm))
print(f'  get_j  (MD J-engine)      {t_j:7.3f} s')
print(f'  get_k  (Rys K)            {t_k:7.3f} s')
print(f'  get_j + get_k             {t_j + t_k:7.3f} s')
print(f'  get_jk (Rys J+K, 1 pass)  {t_jk:7.3f} s')
print(f'  max |J(get_jk) - J(get_j)| = {float(abs(vj2 - vj).max()):.1e}, '
      f'max |K(get_jk) - K(get_k)| = {float(abs(vk2 - vk).max()):.1e}')
