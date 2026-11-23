'''What a later reference of a two-level SCF would lose if it did not build the model's own K again.

At the second reference density D2 = D1 + dD the kernel builds K[dD] with the Rys kernels and adds it to
the K[D1] of the first reference. The energy needs K[D2] only through
    tr(D2 K[D2]) = tr(D1 K[D1]) + 2 tr(dD K[D1]) + tr(dD K[dD]),
so with the exact K[D1] kept, a seminumerical K[dD] enters the energy at second order in dD, and the Fock
matrix (the orbital gradient that is tested) at first order. This script runs the two-level SCF as it is
(exact K at every reference), takes the reference densities and their exact K from the kernel, and prints
for each reference after the first, with dD its change from the first:
- the exchange energy of the kernel's K (K[D1] + K[dD], each screened for its own density) and of the same
  matrices in the form above, against one build of K at that density with a tight direct_scf_tol (--tol);
- for a seminumerical K[dD] on several SGX grids: the energy in that form against the tight build and
  against the kernel's K[dD], what a plain replacement of K[dD] in the Fock matrix would do to the energy
  (first order), the error of the orbital gradient, and the time of the build.

    python bench/gpu/two_level_k_identity.py [--mol caffeine] [--basis def2-tzvp] [--xc wb97m-v] [--smd]
        [--grid 99,590] [--nlcgrid 50,194] [--levels 1,2,3] [--tol 1e-16]
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.sgx import sgx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--xc', default='wb97m-v')
p.add_argument('--smd', action='store_true')
p.add_argument('--grid', default=None)
p.add_argument('--nlcgrid', default=None)
p.add_argument('--conv', type=float, default=1e-10)
p.add_argument('--levels', default='1,2,3')
p.add_argument('--tol', type=float, default=1e-16, help='direct_scf_tol of the tight build of K')
args = p.parse_args()

def sync():
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter()

atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
        'water': molecules.WATER}.get(args.mol, args.mol)
mol = pyscf.M(atom=atom, basis=args.basis, charge=args.charge, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
if args.grid:
    mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
if args.nlcgrid:
    mf.nlcgrids.atom_grid = tuple(int(x) for x in args.nlcgrid.split(','))
if args.smd:
    mf = mf.SMD()
    mf.with_solvent.solvent = 'water'
mf.conv_tol = args.conv
mf.two_level = True

refs = []
def keep(envs):
    kref = envs['kref']
    refs.append(dict(dm=cp.asarray(envs['dm']).copy(), vk=kref['vk'], e=envs['e_tot'],
                     g=envs['norm_gorb'], mo_coeff=envs['mo_coeff'], mo_occ=envs['mo_occ']))
mf.callback = keep
t0 = sync()
e = mf.kernel()
t1 = sync()
print('%s %s %s%s: %d AOs; two-level SCF %.1f s, %d references, E = %.12f' % (
    args.mol, args.basis, args.xc, ' SMD water' if args.smd else '', mol.nao, t1 - t0, len(refs), e))
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(args.xc, spin=mol.spin)

tight = dft.RKS(mol, xc=args.xc)
tight.direct_scf_tol = args.tol

def tr(a, b):
    return float(cp.einsum('ij,ji', a, b))

r1 = refs[0]
t0 = sync()
k1_tight = tight.get_k(mol, r1['dm'], 1, omega, alpha, hyb)
t_tight = sync() - t0
print('reference 1: -tr(D1 K[D1])/4 with K[D1] of one build at direct_scf_tol %g against %g (%.1f s): %+.2e' % (
    mf.direct_scf_tol, args.tol, t_tight, -.25 * tr(r1['dm'], r1['vk'] - k1_tight)))
for n in range(1, len(refs)):
    r2 = refs[n]
    ddm = r2['dm'] - r1['dm']
    dk = r2['vk'] - r1['vk']          # the exact K[dD], as the kernel built it
    mo = r2['mo_coeff']
    occ = r2['mo_occ'] > 0
    def gnorm(dv):
        # the orbital gradient of a change dv of the potential (hf.get_grad)
        return float(cp.linalg.norm(mo[:,~occ].T.dot(dv.dot(mo[:,occ])) * 2))
    def second_order_form(dk):
        return -.25 * (tr(r1['dm'], r1['vk']) + 2 * tr(ddm, r1['vk']) + tr(ddm, dk))
    t0 = sync()
    dk_again = mf.get_k(mol, ddm, 1, omega, alpha, hyb)
    t_exact = sync() - t0
    e_tight = -.25 * tr(r2['dm'], tight.get_k(mol, r2['dm'], 1, omega, alpha, hyb))
    print('reference 1 -> %d: max |dD| %.1e, |g| at the reference %.2e; the second-order term '
          '-tr(dD K[dD])/4 = %.3e Eh; the Rys kernels on dD: %.2f s' % (
              n+1, float(abs(ddm).max()), r2['g'], -.25 * tr(ddm, dk), t_exact))
    print('   -tr(D K)/4 at reference %d against the tight build: the kernel\'s K[D1] + K[dD] %+.2e, '
          'the same matrices in the second-order form %+.2e' % (
              n+1, -.25 * tr(r2['dm'], r2['vk']) - e_tight, second_order_form(dk) - e_tight))
    print('   SGX level, precision    points   s      second-order form: against the tight build   '
          'against the kernel\'s K[dD]   plain replacement   error of |g|')
    for level in (int(x) for x in args.levels.split(',')):
        for dtype in ('double', 'float'):
            obj = sgx.SGX(mol)
            obj.grids_level = level
            obj.kernel_dtype = dtype
            obj.get_k(ddm, 1, omega, alpha, hyb)   # the grid, and the kernels compiled
            t0 = sync()
            dk_s = obj.get_k(ddm, 1, omega, alpha, hyb)
            t_s = sync() - t0
            err = dk_s - dk
            print('   %d %-6s            %7d  %5.2f   %+.2e                                    %+.2e                    '
                  '%+.2e           %.2e' % (
                      level, dtype, obj.size, t_s, second_order_form(dk_s) - e_tight,
                      -.25 * tr(ddm, err), -.25 * tr(r2['dm'], err), gnorm(-.5 * err)), flush=True)
            obj = None
