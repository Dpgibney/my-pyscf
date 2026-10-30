import time, numpy
rng = numpy.random.default_rng(0)
ng, nset, nrho = 67200, 20, 4
fxc = rng.standard_normal((nrho, nrho, 3*ng))[:, :, ng:2*ng]   # a slice of the whole-grid kernel, as fxc[:,:,ip0:ip1]
weight = rng.random(3*ng)[ng:2*ng]
rho1 = rng.standard_normal((nset, nrho, ng))
def tm(f, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r
def cur():
    wv = numpy.empty((nset, nrho, ng))
    for i in range(nset):
        wv[i] = numpy.einsum('yg,xyg,g->xg', rho1[i], fxc, weight)
    return wv
t0, ref = tm(cur)
def alt_a():   # fold the weight into the kernel once per block, one 2-operand einsum for all sets
    wf = fxc * weight
    return numpy.einsum('kyg,xyg->kxg', rho1, wf)
t1, r1 = tm(alt_a)
def alt_b():   # same with explicit broadcasting over the 16 kernel components
    wf = fxc * weight
    wv = numpy.empty((nset, nrho, ng))
    for x in range(nrho):
        numpy.multiply(rho1[:, 0], wf[x, 0], out=wv[:, x])
        for y in range(1, nrho):
            wv[:, x] += rho1[:, y] * wf[x, y]
    return wv
t2, r2 = tm(alt_b)
sc = abs(ref).max()
print(f'GGA, ng={ng}, nset={nset}: current per-set 3-op einsum {t0*1e3:.1f} ms ({t0/nset*1e3:.2f} ms/set)')
print(f'  wf=fxc*w + one 2-op einsum {t1*1e3:.1f} ms (rel diff {abs(r1-ref).max()/sc:.1e});  broadcast loops {t2*1e3:.1f} ms (rel diff {abs(r2-ref).max()/sc:.1e})')
