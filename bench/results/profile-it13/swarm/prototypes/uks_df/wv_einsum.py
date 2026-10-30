'''Cost of forming the response potentials wv from rho1 and fxc per set with numpy.einsum, as in
numint._FusedXC.rks_fxc / rks_fxc_mo (RKS) and uks_fxc (UKS), for one 67200-point block, against
batched forms. numpy.einsum is single threaded, so these times hold at any OMP_NUM_THREADS.'''
import time
import numpy

ng = 67200
nset = 72
rng = numpy.random.default_rng(0)
weight = rng.random(ng)

def best(f, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r

for xctype, nrho in (('GGA', 4), ('MGGA', 5)):
    # RKS
    fxc = rng.standard_normal((nrho, nrho, ng))
    rho1 = rng.standard_normal((nset, nrho, ng))
    def rks_loop():
        wv = numpy.empty((nset, nrho, ng))
        for i in range(nset):
            wv[i] = numpy.einsum('yg,xyg,g->xg', rho1[i], fxc, weight)
        return wv
    def rks_batched():
        fw = fxc * weight  # once per block
        wv = numpy.empty((nset, nrho, ng))
        for x in range(nrho):
            numpy.multiply(rho1[:,0], fw[x,0], out=wv[:,x])
            for y in range(1, nrho):
                wv[:,x] += rho1[:,y] * fw[x,y]
        return wv
    t_loop, w0 = best(rks_loop)
    t_bat, w1 = best(rks_batched)
    print(f'RKS {xctype}: per-set einsum {t_loop*1e3:.0f} ms per block of {nset} sets '
          f'({t_loop/nset*1e3:.2f} ms/set); batched numpy {t_bat*1e3:.0f} ms; '
          f'max rel diff {abs(w1-w0).max()/abs(w0).max():.1e}')

    # UKS: fxc[2,nrho,2,nrho,ng]; rho1a, rho1b [nset,nrho,ng] -> wv[2,nset,nrho,ng]
    fxcu = rng.standard_normal((2, nrho, 2, nrho, ng))
    r1a = rng.standard_normal((nset, nrho, ng))
    r1b = rng.standard_normal((nset, nrho, ng))
    def uks_loop():
        wv = numpy.empty((2, nset, nrho, ng))
        for i in range(nset):
            w = numpy.einsum('xg,xbyg->byg', r1a[i], fxcu[0])
            w += numpy.einsum('xg,xbyg->byg', r1b[i], fxcu[1])
            w *= weight
            wv[:,i] = w
        return wv
    def uks_batched():
        fw = fxcu * weight
        wv = numpy.empty((2, nset, nrho, ng))
        for b in range(2):
            for y in range(nrho):
                out = wv[b,:,y]
                numpy.multiply(r1a[:,0], fw[0,0,b,y], out=out)
                for x in range(1, nrho):
                    out += r1a[:,x] * fw[0,x,b,y]
                for x in range(nrho):
                    out += r1b[:,x] * fw[1,x,b,y]
        return wv
    t_loop, w0 = best(uks_loop)
    t_bat, w1 = best(uks_batched)
    print(f'UKS {xctype}: per-set einsum {t_loop*1e3:.0f} ms per block of {nset} sets '
          f'({t_loop/nset*1e3:.2f} ms/set); batched numpy {t_bat*1e3:.0f} ms; '
          f'max rel diff {abs(w1-w0).max()/abs(w0).max():.1e}')

# hess_elec inner loop (RHF/RKS, rhf.py:72-81) for one atom pair, caffeine sizes
nao, nocc, nmo = 246, 51, 246
mo1 = rng.standard_normal((3, nao, nocc))
mocc = rng.standard_normal((nao, nocc))
e_occ = rng.standard_normal(nocc)
h1 = rng.standard_normal((3, nao, nao))
s1ao = rng.standard_normal((3, nao, nao))
s1oo = rng.standard_normal((3, nocc, nocc))
moe1 = rng.standard_normal((3, nocc, nocc))
def pair():
    dm1 = numpy.einsum('ypi,qi->ypq', mo1, mocc)
    a = numpy.einsum('xpq,ypq->xy', h1, dm1) * 4
    dm1 = numpy.einsum('ypi,qi,i->ypq', mo1, mocc, e_occ)
    a -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
    a -= numpy.einsum('xpq,ypq->xy', s1oo, moe1) * 2
    return a
t_pair, a0 = best(pair)
# hoisted/MO form: h1mo[ia] = h1[ia] @ mocc once per atom; per pair a 3x(nao*nocc) GEMM
h1mo = (h1.reshape(-1, nao) @ mocc).reshape(3, nao, nocc)
s1mo = (s1ao.reshape(-1, nao) @ (mocc * e_occ)).reshape(3, nao, nocc)
def pair_fast():
    a = h1mo.reshape(3, -1) @ mo1.reshape(3, -1).T * 4
    a -= s1mo.reshape(3, -1) @ mo1.reshape(3, -1).T * 4
    a -= s1oo.reshape(3, -1) @ moe1.reshape(3, -1).T * 2
    return a
t_fast, a1 = best(pair_fast)
t_pre, _ = best(lambda: ((h1.reshape(-1, nao) @ mocc), (s1ao.reshape(-1, nao) @ (mocc*e_occ))))
print(f'hess_elec pair body (RKS): {t_pair*1e3:.1f} ms per pair -> {t_pair*300:.1f} s for caffeine (300 pairs); '
      f'MO form {t_fast*1e3:.3f} ms per pair + {t_pre*1e3:.1f} ms per atom; rel diff {abs(a1-a0).max()/abs(a0).max():.1e}')
