"""The Hessian CPHF's J/K at saved response densities (node1): the one-pass Rys J+K build that hessian.rhf._get_jk_mo
runs (_VHFOpt.get_jk, RYS_build_jk), with every launch timed between device synchronizations and grouped by
shell-group class, for the first N of the densities for each N of --ndm. DMS.npz comes from cphf_jk_dms.py.
Usage: cphf_jk_parts.py --dms DMS.npz [--ndm 72,24,8,1] [--basis def2-svp] [--top 25]"""
import argparse
import ctypes
import os
import sys
import time
from collections import defaultdict
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.scf import jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dms', required=True)
ap.add_argument('--ndm', default='72,24,8,1')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--top', type=int, default=25)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
data = numpy.load(args.dms)
dms_all = cupy.asarray(data['dms'])
mf = dft.RKS(mol, xc='b3lyp')
kopt = jk._VHFOpt(mol, mf.direct_scf_tol).build()   # as hessian.rhf._get_jk_mo
smol = kopt.sorted_mol
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]
times = defaultdict(float)
counts = defaultdict(int)
orig = jk.libvhf_rys.RYS_build_jk


def call(*a):
    shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
    g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
    c = f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]})'
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    err = orig(*a)
    cupy.cuda.Device().synchronize()
    times[c] += time.perf_counter() - t0
    counts[c] += 1
    return err


jk.libvhf_rys.RYS_build_jk = call
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis}, nao {mol.nao}, {len(dms_all)} densities saved')
for n in (int(x) for x in args.ndm.split(',')):
    dms = kopt.apply_coeff_C_mat_CT(dms_all[:n])
    kopt.get_jk(dms, 1, 0)   # first call: modules, memory pool
    times.clear()
    counts.clear()
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    kopt.get_jk(dms, 1, 0)
    cupy.cuda.Device().synchronize()
    total = time.perf_counter() - t0
    tk = sum(times.values())
    print(f'  {n} densities: {total:.3f} s, kernels {tk:.3f} s ({tk / n * 1e3:.1f} ms per density)')
    for c, t in sorted(times.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f'    {t:7.3f} s {100 * t / tk:5.1f}%  {c} ({counts[c]} launches)')
