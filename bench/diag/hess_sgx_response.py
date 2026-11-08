'''The options of the Hessian's CPHF on one SCF of an n-alkane: the stock solver, cphf.solve_residual
(Hessian.cphf_residual_tol) and the approximate J/K response (Hessian.sgx_response, direct J + SGX-K on a grid level).
For each: the CPHF's time, the time in the response, its calls and vectors, and the errors of the Hessian and of the
harmonic frequencies against a tightly converged CPHF with the exact response. The partial Hessian and h1ao are
computed once.

Usage: hess_sgx_response.py [--n 5] [--basis def2-svp] [--xc wb97m-v] [--tol 1e-4] [--levels 0,1] [--ref-tol 1e-8]'''
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.lib import logger  # noqa: E402
from pyscf.hessian import rhf as rhf_hess, thermo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=5)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--tol', type=float, default=1e-4, help='cphf_residual_tol of the rows that use it')
ap.add_argument('--levels', default='0,1', help='grid levels of the sgx_response rows')
ap.add_argument('--ref-tol', type=float, default=1e-8, help='cphf_residual_tol of the reference')
args = ap.parse_args()

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
if mf.do_nlc():
    mf.nlcgrids.level = 1
mf.conv_tol = 1e-11
t = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t
mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
nocc = (mo_occ > 0).sum()
nmo = mo_coeff.shape[1]
atmlst = list(range(mol.natm))

t = time.perf_counter()
hobj0 = mf.Hessian()
log = logger.new_logger(hobj0, 0)
de2_partial = hobj0.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, 4000, log)
h1ao = hobj0.make_h1(mo_coeff, mo_occ, None, atmlst, log)
h_nuc = hobj0.hess_nuc(mol, atmlst=atmlst)
print(f'n-C{args.n}H{2*args.n+2}/{args.basis} {args.xc}: nao {mol.nao}, {3 * mol.natm} perturbations, '
      f'{lib.num_threads()} threads; SCF {t_scf:.1f} s, partial Hessian + make_h1 {time.perf_counter() - t:.1f} s',
      flush=True)

count = {}
gen_vind = rhf_hess.gen_vind


def counted_vind(mf_, mo_coeff_, mo_occ_):
    fx = gen_vind(mf_, mo_coeff_, mo_occ_)

    def f(mo1):
        t = time.perf_counter()
        v = fx(mo1)
        count['t'] += time.perf_counter() - t
        count['calls'] += 1
        count['vectors'] += len(mo1.reshape(-1, nmo * nocc))
        return v
    return f


rhf_hess.gen_vind = counted_vind


def run(**attrs):
    hobj = mf.Hessian()
    for k, v in attrs.items():
        setattr(hobj, k, v)
    hobj.partial_hess_elec = lambda *a, **k: de2_partial.copy()
    count.update(t=0., calls=0, vectors=0)
    t = time.perf_counter()
    h = rhf_hess.hess_elec(hobj, mo_energy, mo_coeff, mo_occ, h1ao=h1ao, atmlst=atmlst, max_memory=4000, verbose=0)
    return h + h_nuc, time.perf_counter() - t, dict(count)


def freqs(h):
    return thermo.harmonic_analysis(mol, h)['freq_wavenumber']


h_ref, dt, c = run(cphf_residual_tol=args.ref_tol)
f_ref = freqs(h_ref)
print(f'  reference (exact response, cphf_residual_tol {args.ref_tol:.0e}): {dt:.1f} s, {c["calls"]} calls, '
      f'{c["vectors"]} vectors', flush=True)


def report(label, **attrs):
    h, dt, c = run(**attrs)
    df = freqs(h) - f_ref
    print(f'  {label:42s} CPHF {dt:8.1f} s (response {c["t"]:8.1f} s), {c["calls"]:3d} calls, {c["vectors"]:5d} vectors; '
          f'max|dH| {abs(h - h_ref).max():.2e}, max|df| {abs(df).max():.3f} cm-1, mean|df| {abs(df).mean():.3f}',
          flush=True)


report('stock cphf.solve, exact response')
report(f'solve_residual {args.tol:.0e}, exact response', cphf_residual_tol=args.tol)
for level in [int(x) for x in args.levels.split(',') if x != '']:
    report(f'solve_residual {args.tol:.0e}, direct J + SGX-K level {level}', sgx_response=True, sgx_response_level=level,
           cphf_residual_tol=args.tol)
