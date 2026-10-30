import time, numpy as np, scipy.linalg
threadpool_info = None
rng = np.random.default_rng(1)
for n in (246, 490):
    a = rng.standard_normal((n, n)); h = a + a.T
    b = rng.standard_normal((n, n)); s = b @ b.T / n + np.eye(n)
    scipy.linalg.eigh(h, s)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); scipy.linalg.eigh(h, s); ts.append(time.perf_counter() - t0)
    print(n, f'{min(ts)*1e3:.2f} ms min, {np.median(ts)*1e3:.2f} median')
try:
    for p in threadpool_info(): print(p.get('internal_api'), p.get('num_threads'), p.get('filepath'))
except Exception as e: print(e)
