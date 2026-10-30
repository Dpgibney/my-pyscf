import time, numpy
from functools import reduce
rng = numpy.random.default_rng(0)
def t(f, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter()-t0)
    return best, r
for name, nao, nocc, natm in [('benzene', 114, 21, 12), ('caffeine', 246, 51, 24)]:
    s1ao = rng.standard_normal((3,nao,nao))
    mocc = rng.standard_normal((nao,nocc))
    mo1 = rng.standard_normal((3,nao,nocc))
    h1 = rng.standard_normal((3,nao,nao))
    e = rng.standard_normal(nocc)
    ts, r1 = t(lambda: numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc), 1)
    ts2, r2 = t(lambda: numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc, optimize=True))
    ts3, r3 = t(lambda: numpy.asarray([reduce(numpy.dot, (mocc.T, x, mocc)) for x in s1ao]))
    td1, d1 = t(lambda: numpy.einsum('ypi,qi->ypq', mo1, mocc))
    td1b, d1b = t(lambda: numpy.dot(mo1.reshape(-1,nocc), mocc.T).reshape(3,nao,nao))
    td2, d2 = t(lambda: numpy.einsum('ypi,qi,i->ypq', mo1, mocc, e))
    td2b, d2b = t(lambda: numpy.dot((mo1*e).reshape(-1,nocc), mocc.T).reshape(3,nao,nao))
    tc, c = t(lambda: numpy.einsum('xpq,ypq->xy', h1, d1))
    tcb, cb = t(lambda: numpy.dot(h1.reshape(3,-1), d1.reshape(3,-1).T))
    npair = natm*(natm+1)//2
    print(f'{name}: s1oo 3-op einsum {ts*1e3:.1f} ms (x{natm} = {ts*natm:.2f} s); optimize=True {ts2*1e3:.2f} ms; dot {ts3*1e3:.2f} ms; maxdiff {abs(r1-r3).max():.1e}')
    print(f'  dm1 einsum {td1*1e3:.2f} ms vs dot {td1b*1e3:.2f} ms; dm1e 3-op einsum {td2*1e3:.2f} ms vs dot {td2b*1e3:.2f} ms; xpq,ypq->xy {tc*1e3:.3f} ms vs dot {tcb*1e3:.3f} ms')
    per_pair = td1 + td2 + 3*tc
    print(f'  per pair (2 dm1 + 3 contractions) {per_pair*1e3:.2f} ms x {npair} pairs = {per_pair*npair:.2f} s; per-atom-only alternative {(td1b+td2b)*natm + 3*tcb*npair:.3f} s')
