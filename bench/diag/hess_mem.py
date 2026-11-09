'''Where an RKS Hessian's memory and time go, by phase: the SCF, then the Hessian's parts (partial_hess_elec
with its two-electron, XC and VV10 terms, make_h1 with its XC and VV10 terms, the CPHF with its J/K, XC and VV10
response, hess_nuc, and the rest of hess_elec). A thread samples the RSS every 0.1 s and credits it to every
phase being run (outer phases include their inner ones); each call's RSS before and after shows what it keeps.
Progress lines are printed as calls end.

Usage: hess_mem.py [--n 20] [--basis def2-tzvp] [--xc wb97m-v] [--nlc-level 1] [--conv-tol 1e-10]
                   [--residual-tol T] [--sgx-level L] [--save FILE]
  (n-alkane C_nH_2n+2; max_memory from PYSCF_MAX_MEMORY, PySCF's default 4000 MB when unset; --residual-tol and
  --sgx-level set Hessian.cphf_residual_tol and Hessian.sgx_response; --save keeps the Hessian in a .npy file)'''
import argparse
import os
import resource
import sys
import threading
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.dft import numint  # noqa: E402
from pyscf.hessian import rhf as rhf_hess, rks as rks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--nlc-level', type=int, default=1)
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--log', default=None, help='PySCF log file (verbose 4)')
ap.add_argument('--residual-tol', type=float, default=None, help='Hessian.cphf_residual_tol (cphf.solve_residual)')
ap.add_argument('--sgx-level', type=int, default=None, help='Hessian.sgx_response on this grid level')
ap.add_argument('--save', default=None, help='save the Hessian to this .npy file')
args = ap.parse_args()

PAGE = os.sysconf('SC_PAGE_SIZE')


def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * PAGE / 1e9


T0 = time.perf_counter()
stack = []
peak = {}
stats = {}  # phase -> [calls, seconds, [(rss before, rss after)]]
order = []
done = threading.Event()


def sampler():
    while not done.wait(0.1):
        r = rss()
        for p in list(stack) + ['total']:
            if r > peak.get(p, 0):
                peak[p] = r


threading.Thread(target=sampler, daemon=True).start()


def wrap(owner, name, phase):
    fn = getattr(owner, name)

    def w(*a, **k):
        if phase in stack:  # a nested call of the same phase (get_vnlc_resp -> get_vnlc_resp1)
            return fn(*a, **k)
        before = rss()
        stack.append(phase)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            stack.pop()
            after = rss()
            if phase not in stats:
                stats[phase] = [0, 0., []]
                order.append(phase)
            s = stats[phase]
            s[0] += 1
            s[1] += dt
            s[2].append((before, after))
            print(f'[{time.perf_counter() - T0:7.0f} s] {"  " * len(stack)}{phase}: {dt:.1f} s, '
                  f'RSS {before:.2f} -> {after:.2f} GB, peak so far {peak.get(phase, after):.2f} GB',
                  flush=True)
    setattr(owner, name, w)


mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=4 if args.log else 0,
            output=args.log)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = args.conv_tol
mf.nlcgrids.level = args.nlc_level
print(f'n-C{args.n}H{2 * args.n + 2}/{args.basis} {args.xc}: {mol.natm} atoms, nao {mol.nao}, '
      f'{lib.num_threads()} threads, max_memory {mf.max_memory} MB, VV10 grid level {args.nlc_level}',
      flush=True)

wrap(mf, 'kernel', 'scf')
mf.kernel()
print(f'SCF: E = {mf.e_tot:.10f}, converged {mf.converged}, {mf.cycles} cycles; '
      f'XC grid {mf.grids.weights.size}, VV10 grid {mf.nlcgrids.weights.size} points', flush=True)

hobj = mf.Hessian()
wrap(rhf_hess, '_partial_hess_ejk', 'partial: 2e J/K terms')
wrap(rhf_hess, '_partial_hess_2e', 'partial: _partial_hess_2e')
wrap(rks_hess, '_get_vxc_diag', 'partial: XC diagonal')
wrap(rks_hess, '_get_vxc_deriv2_dm', 'partial: XC deriv2')
wrap(rks_hess, '_get_enlc_deriv2', 'partial: VV10 deriv2')
wrap(rks_hess, '_get_vxc_deriv1', 'make_h1: XC deriv1')
wrap(rks_hess, '_get_vnlc_deriv1', 'make_h1: VV10 deriv1')
wrap(rks_hess, 'get_vnlc_resp', 'CPHF: VV10 response')
wrap(rks_hess, 'get_vnlc_resp1', 'CPHF: VV10 response')
if hasattr(rks_hess, 'get_vnlc_resp_mo'):
    wrap(rks_hess, 'get_vnlc_resp_mo', 'CPHF: VV10 response')
if args.residual_tol is not None:
    hobj.cphf_residual_tol = args.residual_tol
if args.sgx_level is None:
    wrap(mf, 'get_jk', 'J/K (CPHF and others)')
    wrap(mf, 'get_k', 'J/K (CPHF and others)')
else:
    # The response's J/K are those of a copy of mf that takes mf's __dict__: a wrapper left on the instance mf would
    # be copied and replace the copy's SGX get_jk by the exact one.
    from pyscf.sgx.sgx import _SGXHF, SGX  # noqa: E402
    hobj.sgx_response = True
    hobj.sgx_response_level = args.sgx_level
    wrap(_SGXHF, 'get_jk', 'J/K (SGX response)')
    if hasattr(SGX, 'get_k_mo'):
        wrap(SGX, 'get_k_mo', 'SGX-K of the response, from the factors')
wrap(numint.NumInt, 'nr_rks_fxc', 'XC response (CPHF)')
wrap(numint._FusedXC, 'rks_fxc_mo', 'XC response (CPHF)')
wrap(hobj, 'partial_hess_elec', 'partial_hess_elec')
wrap(hobj, 'make_h1', 'make_h1')
wrap(hobj, 'solve_mo1', 'CPHF (solve_mo1)')
wrap(hobj, 'hess_nuc', 'hess_nuc')
wrap(hobj, 'kernel', 'Hessian')
h = hobj.kernel()
done.set()
if args.save:
    numpy.save(args.save, h)

print(f'\nHessian done, {time.perf_counter() - T0:.0f} s in all; max|H| {abs(h).max():.4f}; '
      f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB (ru_maxrss), '
      f'sampled {peak["total"]:.2f} GB, max_memory {mf.max_memory} MB')
print(f'{"phase":36s} {"calls":>5s} {"time":>9s} {"peak RSS":>9s}  RSS before -> after (first calls)')
for p in order:
    n, t, ba = stats[p]
    calls = ', '.join(f'{b:.2f}->{a:.2f}' for b, a in ba[:3]) + (' ...' if len(ba) > 3 else '')
    print(f'{p:36s} {n:5d} {t:8.1f}s {peak.get(p, 0):8.2f} GB  {calls}')
