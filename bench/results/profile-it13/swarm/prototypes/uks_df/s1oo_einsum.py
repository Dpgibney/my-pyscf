'''hess_elec's per-atom s1oo = einsum('xpq,pi,qj->xij', s1ao, mocc, mocc) (hessian/rhf.py:71,
hessian/uhf.py:85-86) at caffeine sizes, single numpy einsum call vs two GEMMs.'''
import time, numpy
nao, nocc = 246, 51
rng = numpy.random.default_rng(0)
s1ao = rng.standard_normal((3, nao, nao)); mocc = rng.standard_normal((nao, nocc))
t0 = time.perf_counter(); a = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc); t1 = time.perf_counter()
b = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc, optimize=True); t2 = time.perf_counter()
c = (mocc.T @ s1ao @ mocc); t3 = time.perf_counter()
print(f's1oo einsum (default, no optimize): {t1-t0:.3f} s per atom; optimize=True {t2-t1:.4f} s; matmul {t3-t2:.4f} s; '
      f'rel diff {abs(c-a).max()/abs(a).max():.1e}')
# the UKS Hessian's _get_vxc_deriv2_dm / deriv1 per-atom einsums are per grid block; skip
