"""J or K of GPU4PySCF at a fixed density, saved for comparing builds (node1). With --save-dm it first converges the
SCF and writes the density; later runs (other trees: source scripts/node1-gpu-env.sh perf|base) load it, build the
matrix once untimed and REPEAT times timed, and write it to OUT.npy; --compare REF.npy prints the largest difference.
Usage: jk_at_dm.py --dm DM.npy [--save-dm] [--part k|j|jk] [--xc b3lyp] [--out OUT.npy] [--compare REF.npy]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--save-dm', action='store_true')
ap.add_argument('--part', default='k', choices=['k', 'j', 'jk'])
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--out', default=None)
ap.add_argument('--compare', default=None)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
if args.save_dm:
    mf.kernel()
    numpy.save(args.dm, cupy.asnumpy(mf.make_rdm1()))
dm = cupy.asarray(numpy.load(args.dm))
build = {'k': lambda: mf.get_k(mol, dm, 1), 'j': lambda: mf.get_j(mol, dm, 1),
         'jk': lambda: cupy.stack(mf.get_jk(mol, dm, 1))}[args.part]
v = build()
ts = []
for _ in range(args.repeat):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    v = build()
    cupy.cuda.Device().synchronize()
    ts.append(time.perf_counter() - t0)
v = cupy.asnumpy(v)
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.part} median {numpy.median(ts):.3f} s of {args.repeat}')
if args.out:
    numpy.save(args.out, v)
if args.compare:
    ref = numpy.load(args.compare)
    print(f'  max |{args.part} - ref| = {abs(v - ref).max():.2e} (max |ref| {abs(ref).max():.2e})')
