"""1-thread probe: eval_xc_eff split into libxc (eval_xc1) and the xc_deriv transforms, per deriv order."""
import time
import numpy as np
from pyscf.dft import libxc, xc_deriv

def tm(f, n=3):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

rng = np.random.default_rng(1)
npts = 200_000
rho = np.empty((5, npts))
rho[0] = rng.random(npts) * 2 + 1e-4
rho[1:4] = rng.standard_normal((3, npts)) * .3
rho[4] = rng.random(npts) * 2 + 1e-3
rho_u = np.stack([rho * .5, rho * .5])  # spin-polarized input [2,nvar,ngrids]

for spin in (0, 1):
    for xc, nvar in (('PBE', 4), ('B3LYP', 4), ('SCAN', 5), ('r2SCAN', 5)):
        r = np.ascontiguousarray((rho if spin == 0 else rho_u)[..., :nvar, :])
        xctype = libxc.xc_type(xc)
        for d in (1, 2, 3):
            try:
                out = libxc.eval_xc1(xc, r, spin, deriv=d)
            except AssertionError:
                continue
            t_xc = tm(lambda: libxc.eval_xc1(xc, r, spin, deriv=d))
            t_tr = tm(lambda: [xc_deriv.transform_xc(r, out, xctype, spin, o) for o in range(1, d + 1)])
            print('spin=%d %-7s d%d: libxc %.1f ns/pt, transforms %.1f ns/pt (%.0f%% of eval_xc_eff)' %
                  (spin, xc, d, t_xc / npts * 1e9, t_tr / npts * 1e9, 100 * t_tr / (t_xc + t_tr)))
