'''Per-atom wv einsums of hessian/uks.py _get_vxc_deriv1/_get_vxc_deriv2_dm (GGA) vs RKS, one
67200-point block, and an all-atoms-at-once variant.'''
import time, numpy
ng, natm = 67200, 24
rng = numpy.random.default_rng(0)
wf = rng.standard_normal((2, 4, 2, 4, ng))
dRa = rng.standard_normal((3, 4, ng)); dRb = rng.standard_normal((3, 4, ng))
def best(f, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r
def uks_atom():
    w = numpy.einsum('xbyg,sxg->bsyg', wf[0], dRa)
    w += numpy.einsum('xbyg,sxg->bsyg', wf[1], dRb)
    return w
t_u, w0 = best(uks_atom)
wfr = rng.standard_normal((4, 4, ng))
t_r, _ = best(lambda: numpy.einsum('xyg,sxg->syg', wfr, dRa))
# alternative: explicit loops over the small indices with vectorized multiply-add over g
def uks_atom_loop():
    w = numpy.empty((2, 3, 4, ng))
    for b in range(2):
        for y in range(4):
            acc = dRa[:, 0] * wf[0, 0, b, y]
            for x in range(1, 4):
                acc += dRa[:, x] * wf[0, x, b, y]
            for x in range(4):
                acc += dRb[:, x] * wf[1, x, b, y]
            w[b, :, y] = acc
    return w
t_l, w1 = best(uks_atom_loop)
print(f'UKS per atom per block: einsum {t_u*1e3:.1f} ms (x{natm} atoms = {t_u*natm:.2f} s per block); '
      f'loop form {t_l*1e3:.1f} ms; rel diff {abs(w1-w0).max()/abs(w0).max():.1e}')
print(f'RKS per atom per block: einsum {t_r*1e3:.1f} ms (x{natm} = {t_r*natm:.2f} s per block)')
