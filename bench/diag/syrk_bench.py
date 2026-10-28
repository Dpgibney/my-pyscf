import time, numpy as np
from pyscf import lib
from pyscf.dft import numint
rng = np.random.default_rng(0)
nao, ng = 246, 67200
ao = rng.standard_normal((3, nao, ng)).transpose(0, 2, 1)     # (3, ng, nao), F-ordered components
wv = rng.standard_normal(ng)                                  # mixed-sign like vtau
def gemm_path():
    out = np.zeros((nao, nao))
    for x in range(3):
        aow = numint._scale_ao(ao[x], wv)
        lib.ddot(aow.T, ao[x], 1, out, 1)
    return out
def syrk_path():
    return numint._dot_ao_wv_ao_hermi(ao, wv, np.zeros((nao, nao)))
for f in (gemm_path, syrk_path): f()
for name, f in (('3x scale+dgemm', gemm_path), ('dsyrk kernel', syrk_path)):
    t0 = time.perf_counter()
    for _ in range(5): r = f()
    print(f'  {name:15s} {(time.perf_counter()-t0)/5*1e3:7.1f} ms')
