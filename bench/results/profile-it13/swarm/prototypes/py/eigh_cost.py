import time, numpy, scipy.linalg, os
print('OMP', os.environ.get('OMP_NUM_THREADS'), 'OPENBLAS', os.environ.get('OPENBLAS_NUM_THREADS'))
rng = numpy.random.default_rng(0)
for n, prof in [(246, '10 ms (0.341 s/34, scf_caffeine_svp_pbe)'), (490, '24 ms (0.489 s/20, scf_c20h42_svp_b3lyp)'),
                (494, '22-24 ms (scf_caffeine_tzvp_*)'), (872, '65 ms (1.421 s/22, scf_c20h42_tzvp_pbe)')]:
    a = rng.standard_normal((n, n)); h = a + a.T
    b = rng.standard_normal((n, n)); s = b @ b.T / n + numpy.eye(n)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); scipy.linalg.eigh(h, s); ts.append(time.perf_counter() - t0)
    ts2 = []
    for _ in range(3):
        t0 = time.perf_counter(); numpy.dot(h, s); ts2.append(time.perf_counter() - t0)
    print(f'n={n}: scipy eigh(h,s) 1 thread min {min(ts)*1e3:.1f} ms median {sorted(ts)[2]*1e3:.1f} ms; 10-thread profile: {prof}; numpy.dot nxn {min(ts2)*1e3:.2f} ms')
