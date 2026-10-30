"""Single-thread GFLOP/s of the per-set GEMM shapes of the fused fxc kernels (dm path and
factored path) against the same work as one GEMM over all sets of a group.
Run with 1 thread (run_in.py ... 1)."""
import ctypes
import time
import numpy as np

blas = ctypes.CDLL('/home/danny/Chemistry/my-pyscf/pyscf/pyscf/lib/deps/lib/libopenblasp-r0.3.34.so')
dgemm = blas.dgemm_
c_int = ctypes.c_int
c_dbl = ctypes.c_double
one, zero = c_dbl(1.), c_dbl(0.)


def gemm(ta, tb, m, n, k, a, lda, b, ldb, c, ldc):
    dgemm(ctypes.c_char_p(ta), ctypes.c_char_p(tb), ctypes.byref(c_int(m)), ctypes.byref(c_int(n)),
          ctypes.byref(c_int(k)), ctypes.byref(one), a.ctypes.data_as(ctypes.c_void_p),
          ctypes.byref(c_int(lda)), b.ctypes.data_as(ctypes.c_void_p), ctypes.byref(c_int(ldb)),
          ctypes.byref(zero), c.ctypes.data_as(ctypes.c_void_p), ctypes.byref(c_int(ldc)))


def bench(fn, flops, reps):
    fn()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return flops / best / 1e9


rng = np.random.default_rng(0)
ng = 224
for label, ns, r, nset in (('benzene', 99, 21, 16), ('caffeine', 163, 51, 16), ('c20', 165, 81, 16)):
    nq = r
    ao = rng.standard_normal(ns * ng)
    dm = rng.standard_normal(nset * ns * ns)
    cb = np.empty(nset * ns * max(ns, ng))
    op = rng.standard_normal(nset * ns * ng)
    print(f'{label}: ns={ns} r=nq={r} nset={nset} ng={ng}')
    # dm path, per set
    def rho_dm():
        for i in range(nset):
            gemm(b'N', b'T', ng, ns, ns, ao, ng, dm[i*ns*ns:], ns, cb, ng)
    def rho_dm_b():
        gemm(b'N', b'T', ng, nset*ns, ns, ao, ng, dm, nset*ns, cb, ng)
    def dot_dm():
        for i in range(nset):
            gemm(b'T', b'N', ns, ns, ng, op[i*ns*ng:], ng, ao, ng, cb, ns)
    def dot_dm_b():
        gemm(b'T', b'N', nset*ns, ns, ng, op, ng, ao, ng, cb, nset*ns)
    f = 2. * ng * ns * ns * nset
    print(f'  dm rho   per set {bench(rho_dm, f, 20):6.1f}  one GEMM {bench(rho_dm_b, f, 20):6.1f} GF/s')
    print(f'  dm dot   per set {bench(dot_dm, f, 20):6.1f}  one GEMM {bench(dot_dm_b, f, 20):6.1f} GF/s')
    # factored path
    A = rng.standard_normal(nset * ns * r)
    tb = rng.standard_normal(r * ng * 4)
    ta = np.empty(nset * r * ng)
    zb = np.empty(nset * ns * ng)
    ket = rng.standard_normal(nset * nq * 2 * ng)
    bra = rng.standard_normal(nset * ns * 2 * ng)
    vb = np.empty(nset * ns * nq)
    def ta_f():
        for i in range(nset):
            gemm(b'N', b'T', ng, r, ns, ao, ng, A[i*ns*r:], r, ta, ng)
    def ta_b():
        gemm(b'N', b'T', ng, nset*r, ns, ao, ng, A, nset*r, ta, ng)
    def z_f():
        for i in range(nset):
            gemm(b'N', b'N', ng, ns, r, tb, ng, A[i*ns*r:], r, zb, ng)
    def z_b():
        gemm(b'N', b'N', ng, nset*ns, r, tb, ng, A, r, zb, ng)
    def v_f():
        for i in range(nset):
            gemm(b'T', b'N', nq, ns, 2*ng, ket[i*nq*2*ng:], 2*ng, bra[i*ns*2*ng:], 2*ng, vb, nq)
    def v_b():
        # ao0^T [ket0_1..ket0_n] and [S_1..S_n]^T FQ0: two GEMMs of k = ng
        gemm(b'T', b'N', nset*nq, ns, ng, ket, ng, ao, ng, vb, nset*nq)
        gemm(b'T', b'N', nq, nset*ns, ng, tb, ng, bra, ng, vb, nq)
    f = 2. * ng * ns * r * nset
    print(f'  lr TA    per set {bench(ta_f, f, 50):6.1f}  one GEMM {bench(ta_b, f, 50):6.1f} GF/s')
    print(f'  lr Z     per set {bench(z_f, f, 50):6.1f}  one GEMM {bench(z_b, f, 50):6.1f} GF/s')
    f = 2. * 2 * ng * ns * nq * nset
    print(f'  lr vQ    per set {bench(v_f, f, 50):6.1f}  two GEMMs {bench(v_b, f, 50):6.1f} GF/s')
