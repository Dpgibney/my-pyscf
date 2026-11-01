"""One K build of GPU4PySCF at a saved density, with only the launches of the chosen shell-group classes inside the
profiler range (node1). Run it as
    ncu --profile-from-start off --section SourceCounters ... python ncu_class.py --dm DM.npy --classes '(fp|pp)'
Each RYS_build_k call of a listed class is bracketed with cudaProfilerStart/Stop, so ncu profiles exactly those
kernels (one per call). The classes are written as in ncu_parts.py, e.g. (fp|pp); --max-calls limits the profiled
calls per class (the largest group pairs come first in the K build's loop order). A class written with its shell
groups, e.g. (ps|ss)@4.2.2.2, selects the call of those groups instead of the first.
With --dms DMS.npz (cphf_jk_dms.py) it instead runs the Hessian CPHF's one-pass J+K build at those densities
(_VHFOpt.get_jk, as hessian.rhf._get_jk_mo) and brackets its RYS_build_jk calls.
Usage: ncu_class.py --dm DM.npy --classes '(fp|pp),(fd|ps)' [--basis def2-tzvp] [--max-calls 1]
       ncu_class.py --dms DMS.npz --classes '(ps|ss)@4.2.2.2' --basis def2-svp"""
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
ap.add_argument('--dm', default=None)
ap.add_argument('--dms', default=None)
ap.add_argument('--classes', required=True)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--max-calls', type=int, default=1)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc='b3lyp')
if args.dms:
    opt = jk._VHFOpt(mol, mf.direct_scf_tol).build()
    dms = opt.apply_coeff_C_mat_CT(cupy.asarray(numpy.load(args.dms)['dms']))
    launcher = 'RYS_build_jk'

    def run():
        return opt.get_jk(dms, 1, 0)
else:
    dm = cupy.asarray(numpy.load(args.dm))
    launcher = 'RYS_build_k'

    def run():
        return mf.get_k(mol, dm, 1)
run()   # optimizer, screening data, module loading
if not args.dms:
    opt = next(v for v in mf._opt_gpu.values() if v is not None)
smol = opt.sorted_mol
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]
targets = set(args.classes.split(','))
picked = {t.split('@')[0] for t in targets if '@' in t}
seen = {}
orig = getattr(jk.libvhf_rys, launcher)


def call(*a):
    shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
    g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
    c = f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]})'
    if c in picked:
        c = c + '@' + '.'.join(map(str, g))
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


setattr(jk.libvhf_rys, launcher, call)
run()
print(f'{args.basis}: profiled {seen}', file=sys.stderr)
