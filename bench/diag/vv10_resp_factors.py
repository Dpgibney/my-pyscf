'''One VV10 response of the CPHF-like densities of an n-alkane (the overlap-derivative part of every atomic perturbation):
from the AO matrices (get_vnlc_resp by chunks of dms, each made for its chunk and the result taken to the MO block, as
vind_mo did) against from the factors of the densities in the MO basis (get_vnlc_resp_mo). Wall and CPU time, the
split between the densities, the pair kernel (f_t) and the potentials, the peak of RSS against its value before the call
(sampled every 10 ms) and the largest difference between the two. The SCF is density-fitted: only its orbitals are used.

Usage: vv10_resp_factors.py N [CHUNK [MAX_MEMORY [REPEATS [BASIS]]]]
CHUNK is the number of dms per chunk of the AO path, by default all of them; MAX_MEMORY (MB) that of the two calls, the
floor of the CPHF's, 2000, by default.'''
import os, sys, threading, time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess
n = int(sys.argv[1])
PAGE = 4096
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * PAGE / 1e9
mol = gto.M(atom=n_alkane(n), basis=(sys.argv[5] if len(sys.argv) > 5 else 'def2-tzvp'), verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.nlcgrids.level = 1
mf.conv_tol = 1e-7
t0 = time.perf_counter()
mf.kernel()
nao = mol.nao
print(f'n-C{n}H{2*n+2}/{mol.basis} wb97m-v: nao {nao}, {lib.num_threads()} threads, density-fitted SCF {time.perf_counter()-t0:.1f} s, '
      f'VV10 grid {mf.nlcgrids.weights.size} points', flush=True)
mo = mf.mo_coeff
occ = mf.mo_occ
mocc = mo[:, occ > 0]
nocc = mocc.shape[1]
nmo = mo.shape[1]
s1a = -mol.intor('int1e_ipovlp', comp=3)
a = []
for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
    for x in range(3):
        s1 = numpy.zeros((nao, nao))
        s1[p0:p1] += s1a[x, p0:p1]
        s1[:, p0:p1] += s1a[x, p0:p1].T
        a.append(s1.dot(mocc))
a = numpy.array(a)
s1a = None
nset = len(a)
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else nset
max_memory = float(sys.argv[3]) if len(sys.argv) > 3 else 2000
print(f'{nset} densities, nocc {nocc}, chunks of {chunk} dms in the AO path, max_memory {max_memory:.0f} MB', flush=True)

times = {}
def timed(owner, name, key):
    f = getattr(owner, name)
    def w(*args, **kw):
        t = time.perf_counter()
        try:
            return f(*args, **kw)
        finally:
            times[key] = times.get(key, 0.) + time.perf_counter() - t
    setattr(owner, name, w)
timed(rks_hess._VV10HessData, 'f_t', 'f_t')
timed(rks_hess, '_vv10_rho', 'rho')
timed(rks_hess, '_vv10_vmat_', 'pot')
timed(numint._FusedXC, 'rho_lowrank', 'rho')
timed(numint._FusedXC, 'dot_ao_mo', 'pot')

def profile(name, fn):
    samples = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append(rss())
            time.sleep(.01)
    base = rss()
    times.clear()
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t = time.perf_counter(); c = time.process_time()
    out = fn()
    t = time.perf_counter() - t; c = time.process_time() - c
    stop.set(); th.join()
    peak = max(samples)
    print(f'{name:18s} {t:7.2f} s (cpu {c:7.1f} s): densities {times.get("rho", 0):6.2f}, f_t {times.get("f_t", 0):6.2f}, '
          f'potentials {times.get("pot", 0):6.2f}; RSS before {base:.2f} GB, peak +{peak-base:.2f} GB', flush=True)
    return out

def from_ao():
    out = numpy.zeros((nset, nmo, nocc))
    for p0, p1 in lib.prange(0, nset, chunk):
        d = numpy.matmul(a[p0:p1], mocc.T)
        d = d + d.transpose(0, 2, 1)
        v = rks_hess.get_vnlc_resp(mf, mol, mo, occ, d, max_memory)
        d = None
        out[p0:p1] += numpy.matmul(mo.T, numpy.matmul(v, mocc))
        v = None
    return out

def from_factors():
    return rks_hess.get_vnlc_resp_mo(mf, mol, mo, occ, a, max_memory, numpy.zeros((nset, nmo, nocc)))

# the data of the ground state are made by whichever comes first and kept, at the same cost for both
rks_hess.get_vnlc_resp(mf, mol, mo, occ, numpy.zeros((1, nao, nao)), max_memory)
for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
    v2 = profile('from the factors', from_factors)
    v1 = profile('from the AO matrices', from_ao)
    print(f'  max|dv| {abs(v1 - v2).max():.2e} of max|v| {abs(v1).max():.3e}', flush=True)
