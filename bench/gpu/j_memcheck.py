"""The J engine's kernels under compute-sanitizer (node1). With CuPy's memory pool off, every array is its own
cudaMalloc of its exact size, so memcheck reports the reads past an array's end that the pool would hide. Builds J
once for the density in DM.npy (NDM copies of it for the multi-density kernels); with --grad, the J engine's nuclear
gradient at that density instead (MD_build_j_ip1, Rt_ip1_dot_dm; part 24).
Usage: compute-sanitizer --tool memcheck python j_memcheck.py --dm DM.npy [--basis def2-svp] [--ndm N] [--grad]"""
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
ap.add_argument('--grad', action='store_true')
args = ap.parse_args()

cupy.cuda.set_allocator(cupy.cuda.memory._malloc)
mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = numpy.load(args.dm)
if args.ndm > 1:
    dm = numpy.stack([dm] * args.ndm)
if args.grad:
    ej = j_engine._VHFOpt(mol, 1e-13).build().get_ej_ip1(cupy.asarray(dm))
    print(f'J gradient built: {args.basis}, max |g| {abs(ej).max():.6e}')
else:
    vj = j_engine.get_j(mol, cupy.asarray(dm))
    cupy.cuda.Device().synchronize()
    print(f'J built: {args.basis}, {args.ndm} density(ies), max |J| {float(abs(vj).max()):.6e}')
