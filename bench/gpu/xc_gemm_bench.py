"""Ways to run the XC part's per-block GEMMs on node1's card (NOTES part 21), at the grid blocks' shapes: K = 4096
grid points and n = the block's AO count (caffeine/def2-SVP blocks hold 128-222 AOs, def2-TZVP blocks 210-442).
1) The Vxc GEMM V = A B^T (A, B: n x K, row-major, as ao_mask[0].dot(aow.T)): cupy's default, then cublasGemmEx with
   each algorithm number 0-23 (CUBLAS_GEMM_ALGOx; -1 is the default). Prints the default time, the best algorithm's
   time and the useful flop rate (2 n^2 K per call); "padded" is what the default's 128x64 tiles cost.
2) The meta-GGA tau term V = sum_j A_j (w A_j)^T over the three AO derivatives, each n x K: as now (three GEMMs into
   one output), as one GEMM of K = 3 x 4096 on the stacked derivatives, and as cublasDsyrkx on the stack (only the
   lower triangle: the result is symmetric), through ctypes on CuPy's cuBLAS handle.
Each time is the median of --repeat calls.
Usage: xc_gemm_bench.py [--n 139,165,181,222,319,442] [--k 4096] [--repeat 15]"""
import argparse
import ctypes
import numpy
import cupy
from cupy_backends.cuda.libs import cublas

ap = argparse.ArgumentParser()
ap.add_argument('--n', default='139,165,181,222,319,442')
ap.add_argument('--k', type=int, default=4096)
ap.add_argument('--repeat', type=int, default=15)
args = ap.parse_args()
K = args.k
handle = cupy.cuda.device.get_cublas_handle()
OP_N, OP_T = 0, 1
R_64F = 1                       # cudaDataType CUDA_R_64F
COMPUTE_64F = 70                # cublasComputeType_t CUBLAS_COMPUTE_64F
libcublas = ctypes.CDLL('libcublas.so.12')
dsyrkx = libcublas.cublasDsyrkx_v2 if hasattr(libcublas, 'cublasDsyrkx_v2') else libcublas.cublasDsyrkx
dsyrkx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int]
dsyrkx.restype = ctypes.c_int
one = numpy.ones(1)
zero = numpy.zeros(1)


def median_time(fn):
    fn()
    ts = []
    for _ in range(args.repeat):
        e0, e1 = cupy.cuda.Event(), cupy.cuda.Event()
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(cupy.cuda.get_elapsed_time(e0, e1) * 1e-3)
    return sorted(ts)[len(ts) // 2]


def gemm_abt(a, b, c, algo, beta=zero):
    '''c (n x n, row-major) = a b^T + beta c for row-major a, b (n x k): column-major c^T = b^T(cm)^T a^T(cm)'''
    n, k = a.shape
    cublas.gemmEx(handle, OP_T, OP_N, n, n, k, one.ctypes.data, b.data.ptr, R_64F, k, a.data.ptr, R_64F, k,
                  beta.ctypes.data, c.data.ptr, R_64F, n, COMPUTE_64F, algo)


print(f'1) Vxc GEMM, K = {K}: ms per call (useful GFLOP/s)')
print(f"{'n':>4s} {'cupy':>14s} {'gemmEx -1':>14s} {'best algo':>20s} {'padded tiles':>13s}")
for n in map(int, args.n.split(',')):
    a = cupy.random.rand(n, K)
    b = cupy.random.rand(n, K)
    c = cupy.empty((n, n))
    flop = 2 * n * n * K
    t_cupy = median_time(lambda: a.dot(b.T, out=c))
    ref = a.dot(b.T)
    times = {}
    for algo in [-1] + list(range(24)):
        try:
            gemm_abt(a, b, c, algo)
            cupy.cuda.Device().synchronize()
        except Exception:
            continue
        if float(abs(c - ref).max()) > 1e-9 * float(abs(ref).max()):
            continue
        times[algo] = median_time(lambda: gemm_abt(a, b, c, algo))
    best = min(times, key=times.get)
    pad = 2 * (-(-n // 128) * 128) * (-(-n // 64) * 64) * K
    print(f'{n:4d} {t_cupy*1e3:7.3f} ({flop/t_cupy/1e9:4.0f}) {times[-1]*1e3:7.3f} ({flop/times[-1]/1e9:4.0f}) '
          f'{best:3d}: {times[best]*1e3:7.3f} ({flop/times[best]/1e9:4.0f}) {pad/flop:12.2f}x')
    print('      all algorithms (ms): ' + ' '.join(f'{k}:{v*1e3:.2f}' for k, v in sorted(times.items())))

print(f'\n2) meta-GGA tau term, three derivatives of K = {K}: ms per call (useful GFLOP/s of the full product)')
print(f"{'n':>4s} {'3 GEMMs':>14s} {'1 GEMM, 3K':>14s} {'dsyrkx, 3K':>14s} {'max |d| syrkx':>14s}")
for n in map(int, args.n.split(',')):
    a3 = cupy.random.rand(3, n, K)
    w = cupy.random.rand(K) - .3
    b3 = a3 * w
    c = cupy.empty((n, n))
    flop = 2 * n * n * 3 * K

    def three():
        gemm_abt(a3[0], b3[0], c, -1)
        gemm_abt(a3[1], b3[1], c, -1, beta=one)
        gemm_abt(a3[2], b3[2], c, -1, beta=one)
    t3 = median_time(three)
    ref = c.copy()
    a_st = cupy.ascontiguousarray(a3.transpose(1, 0, 2).reshape(n, 3 * K))
    b_st = cupy.ascontiguousarray(b3.transpose(1, 0, 2).reshape(n, 3 * K))
    t1 = median_time(lambda: gemm_abt(a_st, b_st, c, -1))

    def syrkx():
        # column-major: c^T(cm) lower = A(cm)^T B(cm) with A(cm) = a_st^T (3K x n): trans = T, k = 3K
        err = dsyrkx(handle, 0, OP_T, n, 3 * K, one.ctypes.data, a_st.data.ptr, 3 * K, b_st.data.ptr, 3 * K,
                     zero.ctypes.data, c.data.ptr, n)
        assert err == 0, err
    c[:] = 0
    ts = median_time(syrkx)
    tri = numpy.tril_indices(n)
    got = cupy.asnumpy(c).T          # row-major c holds the column-major result transposed
    d = abs(got[tri] - cupy.asnumpy(ref)[tri]).max()
    print(f'{n:4d} {t3*1e3:7.3f} ({flop/t3/1e9:4.0f}) {t1*1e3:7.3f} ({flop/t1/1e9:4.0f}) '
          f'{ts*1e3:7.3f} ({flop/ts/1e9:4.0f}) {d:14.1e}')
