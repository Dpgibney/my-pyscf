"""Where a VV10 functional's SCF and gradient NLC term spend their time (node1): runs the SCF with nr_nlc_vxc timed
per call, prints the grid sizes, then profiles grad.rks.get_nlc_exc twice under cProfile.
Usage: nlc_profile.py [--xc b97m_v] [--basis def2-svp] [--mol caffeine|benzene] [--top 18]"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks, numint  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='b97m_v')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--top', type=int, default=18)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
times = []
nr_nlc_vxc = numint.NumInt.nr_nlc_vxc


def timed(self, *a, **k):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    r = nr_nlc_vxc(self, *a, **k)
    cupy.cuda.Device().synchronize()
    times.append(time.perf_counter() - t0)
    return r


numint.NumInt.nr_nlc_vxc = timed
t0 = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t0
print(f'SCF {t_scf:.1f} s, {mf.cycles} cycles; nr_nlc_vxc {len(times)} calls, {sum(times):.2f} s '
      f'(first {times[0]:.2f}, then {sum(times[1:]) / max(len(times) - 1, 1):.2f} s each)')
print(f'grids {mf.grids.coords.shape[0]} points, nlcgrids {mf.nlcgrids.coords.shape[0]} points')
dm = mf.make_rdm1()
xc = mf.xc if mf._numint.libxc.is_nlc(mf.xc) else mf.nlc
for call in (1, 2):
    pr = cProfile.Profile()
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    pr.enable()
    rks_grad.get_nlc_exc(mf._numint, mol, mf.nlcgrids, xc, dm)
    cupy.cuda.Device().synchronize()
    pr.disable()
    print(f'== get_nlc_exc call {call}: {time.perf_counter() - t0:.2f} s')
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(args.top)
    print('\n'.join(x for x in s.getvalue().split('\n') if x.strip()))
