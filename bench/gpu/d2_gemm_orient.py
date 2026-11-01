"""GEMM orientations and K splits for part 14's per-block product out[(A,i),(x,j)] = W[(A,i),K] T[(x,j),K] (node1):
W.dot(T.T), T.dot(W.T), four K chunks summed, and a batched matmul over eight K chunks, for the caffeine and
n-C20H42 block shapes and a large one. Usage: d2_gemm_orient.py (source scripts/node1-gpu-env.sh perf first)."""
import time
import cupy

def tm(fn, rep=20):
    fn(); cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    cupy.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / rep * 1e3

for m, n, k in ((72, 510, 16384), (72, 510, 20480), (99, 516, 16384), (72, 510, 4096), (300, 1500, 20480)):
    w = cupy.random.standard_normal((m, k))
    t = cupy.random.standard_normal((n, k))
    gf = 2 * m * n * k / 1e9
    r = {}
    r['W.dot(T.T)'] = tm(lambda: w.dot(t.T))
    r['T.dot(W.T)'] = tm(lambda: t.dot(w.T))
    # split K into 4 and sum (a manual split-K)
    def splitk(p=4):
        kk = k // p
        out = w[:, :kk].dot(t[:, :kk].T)
        for q in range(1, p):
            out += w[:, q*kk:(q+1)*kk].dot(t[:, q*kk:(q+1)*kk].T)
        return out
    r['split-K 4'] = tm(splitk)
    # strided batched over K chunks then sum
    def batched(p=8):
        kk = k // p
        wb = w.reshape(m, p, kk).transpose(1, 0, 2)
        tb = t.reshape(n, p, kk).transpose(1, 2, 0)
        return cupy.matmul(wb, tb).sum(axis=0)
    r['matmul batched K/8 + sum'] = tm(batched)
    print(f'm {m} n {n} k {k}: ' + ', '.join(f'{key} {v:.2f} ms ({gf/v*1e3:.0f} GFLOP/s)' for key, v in r.items()))
