"""The J engine's kernels under compute-sanitizer (node1). With CuPy's memory pool off, every array is its own
cudaMalloc of its exact size, so memcheck reports the reads past an array's end that the pool would hide. Builds J
once for the density in DM.npy (NDM copies of it for the multi-density kernels).
Usage: compute-sanitizer --tool memcheck python j_memcheck.py --dm DM.npy [--basis def2-svp] [--ndm N]"""
import argparse
import os
import sys
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.scf import j_engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--ndm', type=int, default=1)
args = ap.parse_args()

cupy.cuda.set_allocator(cupy.cuda.memory._malloc)
mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = numpy.load(args.dm)
if args.ndm > 1:
    dm = numpy.stack([dm] * args.ndm)
vj = j_engine.get_j(mol, cupy.asarray(dm))
cupy.cuda.Device().synchronize()
print(f'J built: {args.basis}, {args.ndm} density(ies), max |J| {float(abs(vj).max()):.6e}')
