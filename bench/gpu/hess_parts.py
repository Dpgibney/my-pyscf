"""Time a GPU4PySCF analytical Hessian (node1): the SCF runs first, untimed, then the Hessian once between device
synchronizations, with the GPU clocks. Writes the Hessian to --save (npy) and, with --compare REF.npy, prints the
largest difference. --uks runs the UKS Hessian, of the cation (doublet) unless --charge/--spin say otherwise. Source
scripts/node1-gpu-env.sh perf|base first.
Usage: hess_parts.py [--xc b3lyp] [--basis def2-svp] [--mol caffeine] [--uks [--charge 1 --spin 1]] [--save H.npy]
                     [--compare REF.npy]"""
import argparse
import os
import sys
import time
import numpy
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
ap.add_argument('--uks', action='store_true')
ap.add_argument('--charge', type=int, default=1)
ap.add_argument('--spin', type=int, default=1)
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol]
if args.uks:
    mol = gto.M(atom=atom, basis=args.basis, charge=args.charge, spin=args.spin, verbose=0)
    mf = scf.UHF(mol) if args.xc == 'hf' else dft.UKS(mol, xc=args.xc)
else:
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    mf = scf.RHF(mol) if args.xc == 'hf' else dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
hobj = mf.Hessian()
with GpuMonitor() as mon:
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    h = hobj.kernel()
    cupy.cuda.Device().synchronize()
    t = time.perf_counter() - t0
h = cupy.asnumpy(h) if isinstance(h, cupy.ndarray) else numpy.asarray(h)
label = f'{args.mol} UKS (charge {args.charge}, spin {args.spin})' if args.uks else args.mol
print(f"{os.path.dirname(gpu4pyscf.__file__)}: {label} {args.basis} {args.xc} Hessian {t:.2f} s "
      f"(SCF {mf.cycles} cycles), SM {mon.summary.get('sm_mhz')} MHz", flush=True)
if args.save:
    numpy.save(args.save, h)
if args.compare:
    ref = numpy.load(args.compare)
    print(f'  max |H - ref| = {abs(h - ref).max():.2e} (max |ref| {abs(ref).max():.2e})')
