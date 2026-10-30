import numpy, time
ng = 67200
rng = numpy.random.default_rng(0)
w = rng.random(ng)
f = rng.random((4,4,ng)); r = rng.random((4,ng))
fu = rng.random((2,4,2,4,ng)); ra = rng.random((4,ng)); rb = rng.random((4,ng))
def t(fn, n=20):
    fn(); t0=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n*1e3
print('RKS GGA ms', t(lambda: numpy.einsum('yg,xyg,g->xg', r, f, w)))
def u():
    x = numpy.einsum('xg,xbyg->byg', ra, fu[0]); x += numpy.einsum('xg,xbyg->byg', rb, fu[1]); x *= w
print('UKS GGA ms', t(u))
fw = f*w
def alt():
    out = numpy.empty((4,ng))
    for x in range(4):
        numpy.multiply(r[0], fw[x,0], out=out[x])
        for y in range(1,4): out[x] += r[y]*fw[x,y]
print('RKS GGA numpy loop w/ prescaled fw ms', t(alt))
