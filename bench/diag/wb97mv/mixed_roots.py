'''Numerical check: one n-point Rys-type quadrature for the mixed kernel a/r + b erf(w r)/r, obtained by compressing
the 2n-point rule (Coulomb roots + long-range roots) with a Lanczos recurrence on the discrete measure and the
eigenvalues of the n x n Jacobi matrix (Golub-Welsch). Checks the moments of degree 0..2n-1 in t^2 (what a
quartet's 2D integrals need) against the 2n-point rule, in double precision.'''
import ctypes
import numpy
import pyscf
from pyscf import lib

cint = ctypes.CDLL(pyscf.__path__[0] + '/lib/deps/lib/libcint.so')
cint.CINTrys_roots.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]


def rys(n, x):
    u = numpy.zeros(n)
    w = numpy.zeros(n)
    cint.CINTrys_roots(n, x, u.ctypes.data, w.ctypes.data)
    return u / (1 + u), w   # t^2, weights


def doubled(n, x, theta, a, b):
    t1, w1 = rys(n, x)
    t2, w2 = rys(n, x * theta)
    return numpy.hstack([t1, t2 * theta]), numpy.hstack([a * w1, b * numpy.sqrt(theta) * w2])


def compress(nodes, wts, n):
    '''n-point Gauss rule of the discrete measure (nodes, wts): Lanczos with full reorthogonalization'''
    mu0 = wts.sum()
    q = numpy.sqrt(wts / mu0)
    Q = [q]
    alpha = numpy.zeros(n)
    beta = numpy.zeros(n - 1)
    for k in range(n):
        v = nodes * Q[k]
        alpha[k] = Q[k] @ v
        if k == n - 1:
            break
        v = v - alpha[k] * Q[k]
        if k > 0:
            v = v - beta[k - 1] * Q[k - 1]
        for p in Q:   # reorthogonalize
            v = v - (p @ v) * p
        beta[k] = numpy.linalg.norm(v)
        Q.append(v / beta[k])
    J = numpy.diag(alpha) + numpy.diag(beta, 1) + numpy.diag(beta, -1)
    ev, vec = numpy.linalg.eigh(J)
    return ev, mu0 * vec[0] ** 2


a, b = 0.15, 0.85   # wB97M-V: hyb and alpha - hyb
worst = {}
rng = numpy.random.default_rng(1)
for n in range(1, 10):
    err = 0
    wmin = 1
    for x in list(10 ** rng.uniform(-3, 2.2, 400)) + [0., 1e-8, 35., 60., 150.]:
        for theta in list(10 ** rng.uniform(-4, 0, 12)) + [0.999999, 1e-6]:
            nd, wd = doubled(n, x, theta, a, b)
            t, w = compress(nd, wd, n)
            # moments of degree 0..2n-1, relative to the moment of |measure|
            for m in range(2 * n):
                ref = wd @ nd ** m
                got = w @ t ** m
                err = max(err, abs(got - ref) / abs(wd @ abs(nd) ** m))
            wmin = min(wmin, w.min() / w.sum())
            assert t.min() > -1e-14 and t.max() < 1 + 1e-14
    worst[n] = err
    print(f'n = {n}: max relative error of the moments 0..{2*n-1}: {err:.1e}; smallest relative weight {wmin:.1e}')
