"""One run of the Hessian's two-electron second derivatives (hessian.rhf._partial_ejk_ip2, B3LYP's factors) at a saved
density, with only the launches of the chosen shell-group classes inside the profiler range (node1). Run it as
    ncu --profile-from-start off --section SourceCounters ... python ncu_ejk2_class.py --dm DM.npy --classes '(dp|ps)'
Each RYS_per_atom_jk_ip2_type12 or _type3 call of a listed class is bracketed with cudaProfilerStart/Stop, so ncu
profiles exactly those kernels (one per call). The classes are written as in ncu_parts.py, e.g. (dp|ps); --kinds
picks the kernels (t12, t3 or both), and --max-calls limits the profiled calls per class and kind (the largest group
pairs come first in the loop order). DM.npy is a density in the original AO basis (jk_at_dm.py --save-dm).
Usage: ncu_ejk2_class.py --dm DM.npy --classes '(dp|ps),(dp|dp)' [--kinds t12,t3] [--basis def2-svp]
                         [--max-calls 1]"""
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
from gpu4pyscf.hessian import rhf as rhf_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--kinds', default='t12,t3')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--max-calls', type=int, default=1)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = cupy.asarray(numpy.load(args.dm))
opt = rhf_hess._VHFOpt(mol).build()
smol = opt.sorted_mol
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]
targets = set(args.classes.split(','))
kinds = set(args.kinds.split(','))
seen = {}
lib_rys = rhf_hess.libvhf_rys


def bracket(kind, orig):
    def call(*a):
        shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
        g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
        c = f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]})'
        if kind in kinds and c in targets and seen.get((c, kind), 0) < args.max_calls:
            seen[c, kind] = seen.get((c, kind), 0) + 1
            cupy.cuda.Device().synchronize()
            cupy.cuda.runtime.profilerStart()
            err = orig(*a)
            cupy.cuda.Device().synchronize()
            cupy.cuda.runtime.profilerStop()
            print(f'profiled {c} {kind} groups {g}', file=sys.stderr)
            return err
        return orig(*a)
    return call


lib_rys.RYS_per_atom_jk_ip2_type12 = bracket('t12', lib_rys.RYS_per_atom_jk_ip2_type12)
lib_rys.RYS_per_atom_jk_ip2_type3 = bracket('t3', lib_rys.RYS_per_atom_jk_ip2_type3)
rhf_hess._partial_ejk_ip2(mol, dm, opt, j_factor=1., k_factor=.2)
cupy.cuda.Device().synchronize()
print(f'profiled calls: {sorted(seen.items())}', file=sys.stderr)
