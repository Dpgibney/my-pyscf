"""lib.dot (NPdgemm) on fixed random matrices for shapes that take its m-, n- and k-splits, transposes and
beta != 0: `save` in one tree, `cmp` in another (bit for bit or the max relative difference), with the
median time of 5 calls per shape. Usage: npdgemm_check.py save|cmp OUT
"""
import sys
import time

import numpy as np
from pyscf import lib

mode, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(4)
# (m, k, n, trans_a, trans_b, beta): c[m,n] = op(a) op(b)
SHAPES = [(50000, 246, 246, 'N', 'N', 0), (60001, 114, 114, 'N', 'T', 0), (246, 246, 50000, 'N', 'N', 0),
          (500, 300, 3001, 'T', 'N', 1), (20000, 494, 494, 'N', 'N', 1), (494, 494, 7, 'N', 'N', 0),
          (300, 40000, 300, 'N', 'N', 0)]
res, times = {}, {}
for i, (m, k, n, ta, tb, beta) in enumerate(SHAPES):
    a = rng.standard_normal((k, m) if ta == 'T' else (m, k))
    b = rng.standard_normal((n, k) if tb == 'T' else (k, n))
    c0 = rng.standard_normal((m, n))
    a1 = a.T if ta == 'T' else a
    b1 = b.T if tb == 'T' else b

    def f():
        return lib.dot(a1, b1, 1., c0.copy() if beta else None, beta)
    res[str(i)] = f()
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    times[str(i)] = np.median(t)
if mode == 'save':
    np.savez(out, **res)
    for i, s in enumerate(SHAPES):
        print(f'{s}: {times[str(i)] * 1e3:.2f} ms')
else:
    ref = np.load(out)
    for i, s in enumerate(SHAPES):
        v, r = res[str(i)], ref[str(i)]
        d = 'bit for bit' if np.array_equal(v, r) else f'{abs(v - r).max() / abs(r).max():.1e}'
        print(f'{s}: {d}, {times[str(i)] * 1e3:.2f} ms')
