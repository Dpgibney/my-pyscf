"""One J-engine build of GPU4PySCF at a saved density, with only the launches of the chosen classes inside the
profiler range (node1). Run it as
    ncu --profile-from-start off --section SourceCounters ... python ncu_j_class.py --dm DM.npy --classes '(ps|ss)'
Each MD_build_j call of a listed class is bracketed with cudaProfilerStart/Stop, so ncu profiles exactly its kernel
(one launch per class and build). The classes are written as in the J engine's debug log, e.g. (fs|dp); 'all'
profiles every class.
Usage: ncu_j_class.py --dm DM.npy --classes '(ps|ss),(fs|dp)' [--basis def2-tzvp] [--ndm 1]"""
import argparse
import os
import sys
import ctypes
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.scf import j_engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--ndm', type=int, default=1)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc='b3lyp')
dm = numpy.load(args.dm)
if args.ndm > 1:
    dm = numpy.stack([dm] * args.ndm)
dm = cupy.asarray(dm)


def run():
    return mf.get_j(mol, dm, 1)


run()   # optimizer, screening data, module loading
smol = mf._opt_jengine[mol.omega].sorted_mol
targets = set(args.classes.split(','))
seen = {}
orig = j_engine.libvhf_md.MD_build_j


def call(*a):
    shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
    ls = [lib.param.ANGULAR[smol._bas[shls[2 * n], gto.ANG_OF]] for n in range(4)]
    c = f'({ls[0]}{ls[1]}|{ls[2]}{ls[3]})'
    if (c in targets or 'all' in targets) and c not in seen:
        seen[c] = 1
        cupy.cuda.Device().synchronize()
        cupy.cuda.runtime.profilerStart()
        err = orig(*a)
        cupy.cuda.Device().synchronize()
        cupy.cuda.runtime.profilerStop()
        print(f'profiled {c}', file=sys.stderr)
        return err
    return orig(*a)


j_engine.libvhf_md.MD_build_j = call
run()
print(f'{args.basis}: profiled {list(seen)}', file=sys.stderr)
