# The batched kernels (int2e_sph_batch, int2e_ip1 and the Hessian kinds) of two qcint builds compared bitwise on
# benzene/def2-SVP: batch_compare.py LIB1.so LIB2.so [int2e|int2e_ip1|int2e_ipip1|int2e_ipvip1|int2e_ip1ip2]
import sys, ctypes, itertools
import numpy
from pyscf import gto
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
libs = [ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL) for p in sys.argv[1:3]]
name = sys.argv[3] if len(sys.argv) > 3 else 'int2e'
ncomp = {'int2e': 1, 'int2e_ip1': 3, 'int2e_ipip1': 9, 'int2e_ipvip1': 9, 'int2e_ip1ip2': 9}[name]
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
atm, bas, env = mol._atm, mol._bas, mol._env
nbas = mol.nbas
ls = bas[:,1]; nps = bas[:,2]
lanes = libs[0].int2e_sph_batch_lanes()
fs = [getattr(l, name + '_sph_batch') for l in libs]
oks = [getattr(l, name + '_sph_batch_ok') for l in libs]
csize = (libs[0].int2e_hess_sph_batch_cache_size if ncomp == 9 else libs[0].int2e_sph_batch_cache_size)
csize.restype = ctypes.c_size_t
cache = numpy.empty(csize())
# quartets grouped by class (l, nprim of each shell)
groups = {}
for i in range(nbas):
    for j in range(nbas if ncomp > 1 else i + 1):
        for k in range(nbas if ncomp > 1 else i + 1):
            for l in range(k + 1):
                key = (ls[i], nps[i], ls[j], nps[j], ls[k], nps[k], ls[l], nps[l])
                groups.setdefault(key, []).append((i, j, k, l))
nsame = ndiff = 0
worst_abs = vmax = 0.
for key, qs in groups.items():
    shls = numpy.asarray(qs, dtype=numpy.int32)
    if not oks[0](shls[0].ctypes, bas.ctypes, env.ctypes):
        continue
    nf = ncomp * (2*key[0]+1) * (2*key[2]+1) * (2*key[4]+1) * (2*key[6]+1)
    outs = []
    for f in fs:
        out = numpy.empty((len(qs), nf))
        f(out.ctypes, ctypes.c_int(len(qs)), shls.ctypes, atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
          ctypes.c_int(nbas), env.ctypes, None, cache.ctypes)
        outs.append(out)
    same = numpy.array_equal(outs[0].view(numpy.int64), outs[1].view(numpy.int64))
    nsame += same; ndiff += not same
    worst_abs = max(worst_abs, abs(outs[0] - outs[1]).max())
    vmax = max(vmax, abs(outs[1]).max())
print('%s: %d classes bitwise identical, %d differ; max|d| %.2e (max|v| %.2e)' % (
    name, nsame, ndiff, worst_abs, vmax))
