"""One K build of GPU4PySCF at a saved density, with only the launches of the chosen shell-group classes inside the
profiler range (node1). Run it as
    ncu --profile-from-start off --section SourceCounters ... python ncu_class.py --dm DM.npy --classes '(fp|pp)'
Each RYS_build_k call of a listed class is bracketed with cudaProfilerStart/Stop, so ncu profiles exactly those
kernels (one per call). The classes are written as in ncu_parts.py, e.g. (fp|pp); --max-calls limits the profiled
calls per class (the largest group pairs come first in the K build's loop order).
Usage: ncu_class.py --dm DM.npy --classes '(fp|pp),(fd|ps)' [--basis def2-tzvp] [--max-calls 1]"""
import argparse
import ctypes
import os
import sys
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.scf import jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--max-calls', type=int, default=1)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc='b3lyp')
dm = cupy.asarray(numpy.load(args.dm))
mf.get_k(mol, dm, 1)   # optimizer, screening data, module loading
opt = next(v for v in mf._opt_gpu.values() if v is not None)
smol = opt.sorted_mol
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]
targets = set(args.classes.split(','))
seen = {}
orig = jk.libvhf_rys.RYS_build_k


def call(*a):
    shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
    g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
    c = f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]})'
    if c in targets and seen.get(c, 0) < args.max_calls:
        seen[c] = seen.get(c, 0) + 1
        cupy.cuda.Device().synchronize()
        cupy.cuda.runtime.profilerStart()
        err = orig(*a)
        cupy.cuda.Device().synchronize()
        cupy.cuda.runtime.profilerStop()
        print(f'profiled {c} groups {g}', file=sys.stderr)
        return err
    return orig(*a)


jk.libvhf_rys.RYS_build_k = call
mf.get_k(mol, dm, 1)
print(f'{args.basis}: profiled {seen}', file=sys.stderr)
