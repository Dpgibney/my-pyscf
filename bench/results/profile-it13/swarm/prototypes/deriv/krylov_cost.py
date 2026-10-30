'''Overhead of lib.krylov itself (everything but aop) on CPHF shapes, with a cheap synthetic operator
that converges in about as many cycles as the real CPHF (6-7). Single thread.
usage: krylov_cost.py NRHS NDIM'''
import sys, time
import numpy
from pyscf import lib

nrhs, ndim = int(sys.argv[1]), int(sys.argv[2])
rng = numpy.random.default_rng(0)
r = 64
U, _ = numpy.linalg.qr(rng.standard_normal((ndim, r)))
s = rng.uniform(-0.6, 0.9, r)
d = rng.uniform(0.0, 0.25, ndim)
taop = [0.]
nvec = []
def aop(x):
    c0 = time.perf_counter()
    x = numpy.asarray(x).reshape(-1, ndim)
    nvec.append(len(x))
    y = lib.dot(lib.dot(x, U) * s, U.T) + x * d
    taop[0] += time.perf_counter() - c0
    return y
b = rng.standard_normal((nrhs, ndim))
tol = 1e-9 * nrhs / 3
t0 = time.perf_counter()
x = lib.krylov(aop, b, tol=tol, max_cycle=50, lindep=max(tol**2, 1e-16))
t1 = time.perf_counter()
print(f'nrhs {nrhs} ndim {ndim}: krylov {t1-t0:.2f} s of which aop {taop[0]:.2f} s; '
      f'cycles {len(nvec)}, vectors per cycle {nvec}, total {sum(nvec)}')
res = abs(aop(x) + x - b).max()
print(f'  residual {res:.1e}')

def krylov_blas(aop, b, tol, max_cycle=50, lindep=1e-16):
    '''lib.krylov (nroots > 1, hermi=False, x0=None) with the per-vector loops as GEMMs'''
    from pyscf.lib.linalg_helper import _qr
    x1 = b
    nroots, ndim = x1.shape
    x1, rmat = _qr(x1, numpy.dot, lindep)
    x1 *= rmat.diagonal()[:,None]
    innerprod = list(rmat.diagonal().real ** 2)
    xs = numpy.empty((0, ndim)); ax = numpy.empty((0, ndim))
    for cycle in range(max_cycle):
        axt = aop(x1)
        xs = numpy.vstack([xs, x1]); ax = numpy.vstack([ax, axt])
        w = lib.dot(axt, xs.T) / numpy.asarray(innerprod)
        x1 = axt - lib.dot(w, xs)
        x1, rmat = _qr(x1, numpy.dot, lindep)
        x1 *= rmat.diagonal()[:,None]
        innerprod1 = rmat.diagonal().real ** 2
        if max(innerprod1, default=0.) < lindep or max(innerprod1, default=0.) < tol**2:
            break
        mask = (innerprod1 > lindep) & (innerprod1 > tol**2)
        x1 = x1[mask]
        innerprod.extend(innerprod1[mask])
    h = lib.dot(xs, ax.T)
    h[numpy.diag_indices(len(xs))] += innerprod
    g = lib.dot(xs, b.T)
    c = numpy.linalg.solve(h, g)
    return lib.dot(c.T, xs)

taop[0] = 0.; nvec.clear()
t0 = time.perf_counter()
x2 = krylov_blas(aop, b, tol=tol, lindep=max(tol**2, 1e-16))
t1 = time.perf_counter()
print(f'BLAS-3 variant: {t1-t0:.2f} s of which aop {taop[0]:.2f} s; cycles {len(nvec)}; '
      f'max|x2-x| {abs(x2-x).max():.1e} (|x| {abs(x).max():.1e})')
