"""Where a GPU4PySCF nuclear gradient spends its time (node1): the SCF runs first, untimed; the gradient is then timed
twice (the first call also builds the gradient's integral screening), and in the second call the two-electron
part (jk_energy_per_atom, the Rys derivative kernels) is timed between device synchronizations. Source
scripts/node1-gpu-env.sh perf|base first. Prints and appends a JSON line to --out.
Usage: grad_parts.py [--xc b3lyp] [--basis def2-svp] [--mol caffeine] [--out FILE]"""
import argparse
import json
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft, scf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--out', default=None)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = scf.RHF(mol) if args.xc == 'hf' else dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
g = mf.nuc_grad_method()


def sync_time(f):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    r = f()
    cupy.cuda.Device().synchronize()
    return r, time.perf_counter() - t0


de1, t_first = sync_time(g.kernel)
cls = type(g)
orig = cls.jk_energy_per_atom
t_jk = []


def timed(self, *a, **kw):
    r, t = sync_time(lambda: orig(self, *a, **kw))
    t_jk.append(t)
    return r


cls.jk_energy_per_atom = timed
with GpuMonitor() as mon:
    de, t_total = sync_time(g.kernel)
cls.jk_energy_per_atom = orig
r = {'tree': os.path.dirname(gpu4pyscf.__file__), 'mol': args.mol, 'basis': args.basis, 'xc': args.xc,
     'scf_cycles': int(mf.cycles), 'grad_first_s': round(t_first, 3), 'grad_s': round(t_total, 3),
     'jk_s': [round(t, 3) for t in t_jk], 'max_abs_grad': float(abs(de).max()),
     'grad': [[float(x) for x in row] for row in de], 'gpu': mon.summary}
print(f"{args.mol} {args.basis} {args.xc}: gradient {t_total:.2f} s (first call {t_first:.2f} s), two-electron part "
      f"{sum(t_jk):.2f} s in {len(t_jk)} calls ({100 * sum(t_jk) / t_total:.0f}%), SM {mon.summary.get('sm_mhz')}")
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(r) + '\n')
