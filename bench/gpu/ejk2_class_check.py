"""The Hessian's two-electron second derivatives (hessian.rhf._partial_ejk_ip2, B3LYP's factors) class by class at a
saved density (node1): for each listed shell-group class, one run of the stage in which only that class's launches
of RYS_per_atom_jk_ip2_type12 and _type3 execute (the others return at once), so the result is that class's
contribution. Each launch is timed between device synchronizations. With --kinds t12 or t3 only that kernel runs.
The contributions go to --out (an .npz keyed by class and kind), and --compare REF.npz prints, per class, the largest
difference and the time against REF. --all adds one run of the whole stage. With --ref-off the second run of each
class sets GPU4PYSCF_RYS_GEN_IP2_OFF=1 (builds with --gen-ip2-env), so the generated and the other kernels are
compared within one tree. DM.npy is a density in the original AO basis (jk_at_dm.py --save-dm).
Usage: ejk2_class_check.py --dm DM.npy --classes '(dp|ps),...' [--kinds t12,t3] [--basis def2-svp] [--uks-dm]
                           [--repeat 1] [--out OUT.npz] [--compare REF.npz] [--ref-off] [--all]"""
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
from gpu4pyscf.hessian import rhf as rhf_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--kinds', default='t12,t3')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--uks-dm', action='store_true', help='use DM as both spins (n_dm = 2)')
ap.add_argument('--repeat', type=int, default=1)
ap.add_argument('--out', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--ref-off', action='store_true')
ap.add_argument('--all', action='store_true')
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = cupy.asarray(numpy.load(args.dm))
if args.uks_dm:
    dm = cupy.stack((dm * .5, dm * .5))
opt = rhf_hess._VHFOpt(mol).build()
smol = opt.sorted_mol
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]
lib_rys = rhf_hess.libvhf_rys
state = {'target': None, 'kinds': set(args.kinds.split(','))}
times = defaultdict(float)


def wrap(kind, orig):
    def call(*a):
        shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
        g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
        c = f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]})'
        if state['target'] != 'all' and (c != state['target'] or kind not in state['kinds']):
            return 0
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        err = orig(*a)
        cupy.cuda.Device().synchronize()
        times[c, kind] += time.perf_counter() - t0
        return err
    return call


lib_rys.RYS_per_atom_jk_ip2_type12 = wrap('t12', lib_rys.RYS_per_atom_jk_ip2_type12)
lib_rys.RYS_per_atom_jk_ip2_type3 = wrap('t3', lib_rys.RYS_per_atom_jk_ip2_type3)


def run(target):
    state['target'] = target
    best, ejk = None, None
    for _ in range(args.repeat):
        times.clear()
        ejk = cupy.asnumpy(rhf_hess._partial_ejk_ip2(mol, dm, opt, j_factor=1., k_factor=.2))
        t = dict(times)
        best = t if best is None or sum(t.values()) < sum(best.values()) else best
    return ejk, best


ref = numpy.load(args.compare) if args.compare else None
out = {}
classes = args.classes.split(',') + (['all'] if args.all else [])
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis}, n_dm {dm.ndim - 1 if args.uks_dm else 1}')
for c in classes:
    ejk, t = run(c)
    key = c if c == 'all' else f'{c} {args.kinds}'
    out[key] = ejk
    tt = sum(t.values())
    line = f'  {key:22s} {tt:8.3f} s  (' + ', '.join(f'{k} {v:.3f}' for (cc, k), v in sorted(t.items())) + ')'
    if args.ref_off:
        os.environ['GPU4PYSCF_RYS_GEN_IP2_OFF'] = '1'
        ejk0, t0 = run(c)
        del os.environ['GPU4PYSCF_RYS_GEN_IP2_OFF']
        out[key + ' off'] = ejk0
        line += (f'; generated off {sum(t0.values()):8.3f} s ({sum(t0.values()) / max(tt, 1e-9):.2f}x), '
                 f'max |d| {abs(ejk - ejk0).max():.2e} (max {abs(ejk0).max():.2e})')
    if ref is not None and key in ref.files:
        line += f'; against REF max |d| {abs(ejk - ref[key]).max():.2e} (max {abs(ref[key]).max():.2e})'
    print(line, flush=True)
if args.out:
    numpy.savez(args.out, **out)
