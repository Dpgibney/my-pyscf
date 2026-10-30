'''Trace the CPHF of the RKS Hessian (benzene/SVP B3LYP): number of fx calls, vectors per call, time per
call, and whether the post-Krylov fvind(mo1) equals the linear combination of the Krylov vectors'
responses. Single thread.'''
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, dft, lib
from pyscf.hessian import rhf as rhf_hess

mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.conv_tol = 1e-10
t0 = time.perf_counter()
mf.kernel()
print(f'SCF {time.perf_counter()-t0:.1f} s, {mf.cycles} cycles', flush=True)
h = mf.Hessian()
mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
t0 = time.perf_counter()
h1ao = h.make_h1(mo_coeff, mo_occ)
print(f'make_h1 {time.perf_counter()-t0:.1f} s', flush=True)

fx0 = rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
calls = []
inputs, outputs = [], []
def fx(mo1):
    c0 = time.perf_counter()
    v = fx0(mo1)
    n = mo1.reshape(-1, v.shape[-2]*v.shape[-1]).shape[0] if v.ndim == 3 else len(mo1)
    calls.append((len(v), time.perf_counter() - c0))
    inputs.append(numpy.array(mo1).reshape(len(v), -1)); outputs.append(v.reshape(len(v), -1).copy())
    return v
t0 = time.perf_counter()
mo1, e1 = h.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, fx)
print(f'solve_mo1 {time.perf_counter()-t0:.1f} s; fx calls (nset, s): '
      + ', '.join(f'({n}, {t:.2f})' for n, t in calls))
# is the last call's input in the span of the earlier inputs (so that its response is the same
# combination of their responses)?
X = numpy.vstack(inputs[:-1]); V = numpy.vstack(outputs[:-1])
xl, vl = inputs[-1], outputs[-1]
c, *_ = numpy.linalg.lstsq(X.T, xl.T, rcond=None)
print(f'last input in span of earlier inputs: residual {abs(X.T @ c - xl.T).max():.1e} '
      f'(|x| {abs(xl).max():.1e}); response from the combination vs computed: '
      f'{abs(V.T @ c - vl.T).max():.1e} (|v| {abs(vl).max():.1e})')
