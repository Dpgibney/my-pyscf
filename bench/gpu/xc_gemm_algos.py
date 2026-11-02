"""cuBLAS's GEMM algorithms (cublasGemmEx, CUBLAS_GEMM_ALGO0-23; -1 the default heuristic) at every per-block GEMM
shape of the XC code (node1; NOTES part 21), with K = 4096 grid points, nocc orbital columns and n block AOs:
  c^T ao    (nocc x n)(n x K)   the density pass, and C^T aow in the gradient
  c c0      (n x nocc)(nocc x K) dm . ao, and dm . aow in the gradient
  ao aow^T  (n x K)(K x n)      the Vxc pass (GGA, LDA; meta-GGA's rho part)
  tau, 3K   (n x 3K)(3K x n)    the meta-GGA tau term as one GEMM on the stacked derivatives
The operands are row-major like CuPy's; each time is the median of --repeat calls. Prints, per shape, the default's
time, the best algorithm and its time, and the algorithms within 3% of the best.
Usage: xc_gemm_algos.py [--n 139,165,181,222,260,319,400,442] [--nocc 51] [--k 4096] [--repeat 11]"""
import argparse
import numpy
import cupy
from cupy_backends.cuda.libs import cublas

ap = argparse.ArgumentParser()
ap.add_argument('--n', default='139,165,181,222,260,319,400,442')
ap.add_argument('--nocc', type=int, default=51)
ap.add_argument('--k', type=int, default=4096)
ap.add_argument('--repeat', type=int, default=11)
args = ap.parse_args()
handle = cupy.cuda.device.get_cublas_handle()
OP_N, OP_T = 0, 1
R_64F, COMPUTE_64F = 1, 70
one = numpy.ones(1)
zero = numpy.zeros(1)


def gemm(a, ta, b, tb, c, algo):
    '''row-major c (M x N) = op(a) op(b): column-major c^T = op(b)^T op(a)^T'''
    m, n = c.shape
    k = a.shape[0] if ta else a.shape[1]
    lda, ldb = a.shape[1], b.shape[1]
    cublas.gemmEx(handle, OP_T if tb else OP_N, OP_T if ta else OP_N, n, m, k, one.ctypes.data,
                  b.data.ptr, R_64F, ldb, a.data.ptr, R_64F, lda, zero.ctypes.data, c.data.ptr, R_64F, n,
                  COMPUTE_64F, algo)


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


K, nocc = args.k, args.nocc
print(f'{"shape":10s} {"n":>4s} {"default ms":>10s} {"best":>5s} {"ms":>7s} {"x":>5s}  algorithms within 3% of the best')
for n in map(int, args.n.split(',')):
    cases = {
        'c^T ao': (cupy.random.rand(n, nocc), True, cupy.random.rand(n, K), False, (nocc, K)),
        'c c0': (cupy.random.rand(n, nocc), False, cupy.random.rand(nocc, K), False, (n, K)),
        'ao aow^T': (cupy.random.rand(n, K), False, cupy.random.rand(n, K), True, (n, n)),
        'tau, 3K': (cupy.random.rand(n, 3 * K), False, cupy.random.rand(n, 3 * K), True, (n, n)),
    }
    for name, (a, ta, b, tb, shape) in cases.items():
        c = cupy.empty(shape)
        ref = (a.T if ta else a) @ (b.T if tb else b)
        times = {}
        for algo in [-1] + list(range(24)):
            try:
                gemm(a, ta, b, tb, c, algo)
                cupy.cuda.Device().synchronize()
            except Exception:
                continue
            if float(abs(c - ref).max()) > 1e-10 * float(abs(ref).max()):
                continue
            times[algo] = median_time(lambda: gemm(a, ta, b, tb, c, algo))
        best = min(times, key=times.get)
        near = [k for k, v in sorted(times.items()) if v <= 1.03 * times[best]]
        print(f'{name:10s} {n:4d} {times[-1]*1e3:10.3f} {best:5d} {times[best]*1e3:7.3f} '
              f'{times[-1]/times[best]:5.2f}  {near}' + (f'; algorithm 1 {times[1]*1e3:.3f} ms' if 1 in times else ''))
