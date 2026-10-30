"""1 thread: is libxc slower when called inside the fused nr_rks than standalone on the same inputs?"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint, libxc

xc = sys.argv[1] if len(sys.argv) > 1 else 'PBE'
fused = int(sys.argv[2]) if len(sys.argv) > 2 else 1
numint.FUSED_XC = bool(fused) if hasattr(numint, 'FUSED_XC') else None
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
dm = mf.get_init_guess(key='minao')
ni = mf._numint
rec = []
orig = ni.eval_xc_eff
def wrapped(xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None, spin=None):
    t0 = time.perf_counter()
    out = orig(xc_code, rho, deriv=deriv, omega=omega, xctype=xctype, verbose=verbose, spin=spin)
    rec.append((time.perf_counter() - t0, np.array(rho, copy=True), deriv, xctype, spin))
    return out
ni.eval_xc_eff = wrapped
ni.nr_rks(mol, mf.grids, xc, dm)   # warm-up (radial cache etc.)
rec.clear()
t0 = time.perf_counter()
for _ in range(2):
    ni.nr_rks(mol, mf.grids, xc, dm)
t_nr = time.perf_counter() - t0
in_ctx = sum(r[0] for r in rec)
npts = sum(r[1].shape[-1] for r in rec)
# standalone on the identical inputs, same order
best = []
for t, rho, deriv, xctype, spin in rec:
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        orig(xc, rho, deriv=deriv, xctype=xctype, spin=spin)
        ts.append(time.perf_counter() - t0)
    best.append((ts[0], min(ts)))
first = sum(b[0] for b in best); mn = sum(b[1] for b in best)
print(f'{xc} fused={fused} FUSED_XC={getattr(numint, "FUSED_XC", None)}: nr_rks x2 {t_nr:.3f} s; '
      f'{len(rec)} eval_xc_eff calls, {npts} points; in nr_rks {in_ctx*1e3:.1f} ms '
      f'({in_ctx/npts*1e9:.0f} ns/pt); standalone same inputs first {first*1e3:.1f} ms, '
      f'best {mn*1e3:.1f} ms ({mn/npts*1e9:.0f} ns/pt); ratio in-context/standalone-best {in_ctx/mn:.2f}')
print('per call in-context ms:', np.round([r[0]*1e3 for r in rec], 1))
print('per call standalone ms:', np.round([b[1]*1e3 for b in best], 1))
