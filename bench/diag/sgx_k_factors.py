'''One SGX-K of the CPHF-like densities of an n-alkane (those of the overlap-derivative part of every atomic
perturbation): from the density matrices (sgx_jk.get_k_only, many densities) against from their factors, in the MO basis
(SGX.get_k_mo). Wall and CPU time, SGX's split between F = X D, G = A F and K = X^T w G (debug1), the peak of RSS against
its value before the call (sampled every 10 ms), and the largest difference between the two.

A sgx_max_memory of 1 keeps the blocks of grid points at their minimum of 224, as for a molecule that is at max_memory.

Usage: sgx_k_factors.py N LEVEL SGX_MAX_MEMORY [REPEATS [BASIS]]'''
import os, sys, threading, time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane
from pyscf import gto, dft, lib
from pyscf.lib import logger
from pyscf.sgx import sgx_jk
n = int(sys.argv[1]); level = int(sys.argv[2]); maxmem = float(sys.argv[3])
PAGE = 4096
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * PAGE / 1e9
mol = gto.M(atom=n_alkane(n), basis=(sys.argv[5] if len(sys.argv) > 5 else 'def2-tzvp'), verbose=0)
mf = dft.RKS(mol, xc='b3lyp').RIJCOSX()
mf.conv_tol = 1e-8
t0 = time.perf_counter()
mf.kernel()
nao = mol.nao
print(f'n-C{n}H{2*n+2}: nao {nao}, natm {mol.natm}, {lib.num_threads()} threads, SCF {time.perf_counter()-t0:.1f} s', flush=True)
mo_coeff = mf.mo_coeff
mocc = mo_coeff[:, mf.mo_occ > 0]
nocc = mocc.shape[1]
s1a = -mol.intor('int1e_ipovlp', comp=3)
a = []
for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
    for x in range(3):
        s1 = numpy.zeros((nao, nao))
        s1[p0:p1] += s1a[x, p0:p1]
        s1[:, p0:p1] += s1a[x, p0:p1].T
        a.append(s1.dot(mocc))
a = numpy.array(a)
dms = numpy.matmul(a, mocc.T)
dms = dms + dms.transpose(0, 2, 1)
sgx = mf.with_df
sgx.build(level=level)
sgx._full_dm = None
sgx.max_memory = maxmem
sgx.verbose = logger.DEBUG1
sgx.stdout = sys.stdout
mol.verbose = 0
unit = dms.nbytes / 1e9
print(f'{len(a)} densities, nocc {nocc}, grid level {level}: {sgx.grids.weights.size} points, sgx.max_memory {maxmem:.0f}; a stack of AO matrices is {unit:.3f} GB', flush=True)
def profile(name, fn):
    samples = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append(rss())
            time.sleep(.01)
    base = rss()
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t = time.perf_counter(); c = time.process_time()
    out = fn()
    t = time.perf_counter() - t; c = time.process_time() - c
    stop.set(); th.join()
    peak = max(samples)
    print(f'{name:28s} {t:7.2f} s (cpu {c:7.1f} s)  RSS before {base:.2f} GB, peak +{peak-base:.2f} GB = +{(peak-base)/unit:.2f} stacks', flush=True)
    return out
tol = mf.direct_scf_tol
def k_ao():
    return numpy.matmul(mo_coeff.T, numpy.matmul(sgx_jk.get_k_only(sgx, dms, hermi=1, direct_scf_tol=tol), mocc))
def k_mo():
    return sgx.get_k_mo(a, mocc, mo_coeff, tol)
for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
    v2 = profile('K from the factors, MO block', k_mo)
    v1 = profile('K from the dms, then MO', k_ao)
    print(f'  max|dK| {abs(v1 - v2).max():.2e} of max|K| {abs(v1).max():.3f}', flush=True)
