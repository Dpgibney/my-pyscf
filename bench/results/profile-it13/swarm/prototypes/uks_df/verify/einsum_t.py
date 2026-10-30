import numpy, time
from pyscf import lib
nao, nocc = 246, 51
rng = numpy.random.default_rng(0)
s1ao = rng.random((3,nao,nao)); mocc = rng.random((nao,nocc)); e = rng.random(nocc)
mo1 = rng.random((3,nao,nocc)); h1 = rng.random((3,nao,nao)); moe1 = rng.random((3,nocc,nocc))
def t(f, n=3):
    f(); t0=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t0)/n*1e3
print('s1oo einsum ms', t(lambda: numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc), 2))
print('s1oo lib.einsum ms', t(lambda: lib.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)))
def pair():
    dm1 = numpy.einsum('ypi,qi->ypq', mo1, mocc)
    a = numpy.einsum('xpq,ypq->xy', h1, dm1)
    dm1 = numpy.einsum('ypi,qi,i->ypq', mo1, mocc, e)
    b = numpy.einsum('xpq,ypq->xy', s1ao, dm1)
    c = numpy.einsum('xpq,ypq->xy', moe1, moe1)
print('pair body ms', t(pair, 5))
