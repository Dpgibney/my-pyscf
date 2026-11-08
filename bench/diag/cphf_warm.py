'''The Hessian's CPHF (all atoms in one block, as solve_mo1 for small molecules) solved with the exact response,
cold and warm-started from the solution of a cheap response: DF-J (J-engine) + SGX-K on a level-0 grid, the XC
kernel on a level-1 grid, no VV10. Prints the exact fvind calls and vectors, the times, and the true residual of
each solution (one more exact fvind call each, here only).

Usage: cphf_warm.py [--mol propane|water|caffeine|alkane:N] [--basis def2-svp] [--xc wb97m-v] [--cheap-tol F]'''
import argparse
import os
import sys
import time
from functools import reduce
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.scf import cphf  # noqa: E402
from pyscf.hessian import rhf as rhf_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='propane')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--cheap-tol', type=float, default=1.)
ap.add_argument('--cheap-xc-level', type=int, default=1)
ap.add_argument('--cheap-sgx-level', type=int, default=0)
args = ap.parse_args()
if args.mol == 'water':
    atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
elif args.mol == 'caffeine':
    atom = CAFFEINE
elif args.mol.startswith('alkane:'):
    atom = n_alkane(int(args.mol.split(':')[1]))
else:
    atom = n_alkane(3)
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.nlcgrids.level = 1
mf.conv_tol = 1e-10
mf.kernel()
mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
nao, nmo = mo_coeff.shape
mocc = mo_coeff[:, mo_occ > 0]
nocc = mocc.shape[1]

t = time.perf_counter()
hobj = mf.Hessian()
h1ao = hobj.make_h1(mo_coeff, mo_occ)
t_h1 = time.perf_counter() - t
s1a = -mol.intor('int1e_ipovlp', comp=3)
aoslices = mol.aoslice_by_atom()
h1vo, s1vo = [], []
for ia in range(mol.natm):
    p0, p1 = aoslices[ia][2:]
    s1ao = numpy.zeros((3, nao, nao))
    s1ao[:, p0:p1] += s1a[:, p0:p1]
    s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    s1vo.append([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in s1ao])
    h1vo.append([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in h1ao[ia]])
h1vo = numpy.asarray(h1vo).reshape(-1, nmo, nocc)
s1vo = numpy.asarray(s1vo).reshape(-1, nmo, nocc)
tol = mf.conv_tol_cpscf * mol.natm

def counted(fx, log):
    def f(mo1):
        t = time.perf_counter()
        v = fx(mo1)
        log.append((len(mo1.reshape(-1, nmo * nocc)), time.perf_counter() - t))
        return v
    return f

fx = rhf_hess.gen_vind(mf, mo_coeff, mo_occ)

# The cheap response
mf_c = mf.RIJCOSX()
mf_c.with_df.grids_level_i = mf_c.with_df.grids_level_f = args.cheap_sgx_level
mf_c.with_df.build(level=args.cheap_sgx_level)
mf_c._nsteps_direct = 0  # the SCF state of an SGX object used outside an SCF
mf_c._in_scf = False
mf_c.grids = dft.gen_grid.Grids(mol)
mf_c.grids.level = args.cheap_xc_level
mf_c.grids.build(with_non0tab=True)
vresp_c = mf_c.gen_response(mo_coeff, mo_occ, hermi=1, with_nlc=False)
def fx_c(mo1):
    mo1 = mo1.reshape(-1, nmo, nocc)
    dm1 = numpy.empty((len(mo1), nao, nao))
    for i, x in enumerate(mo1):
        dm = reduce(numpy.dot, (mo_coeff, x * 2, mocc.T))
        dm1[i] = dm + dm.T
    v1 = vresp_c(dm1)
    return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in v1])

def residual(mo1):
    occ, vir = mo_occ > 0, mo_occ == 0
    e_i, e_a = mo_energy[occ], mo_energy[vir]
    v = fx(mo1).reshape(mo1.shape)
    hs = h1vo - s1vo * e_i
    r = (e_a[:, None] - e_i) * mo1[:, vir] + v[:, vir] + hs[:, vir]
    return abs(r).max()

print(f'{args.mol}/{args.basis} {args.xc}: nao {nao}, natm {mol.natm}, {3*mol.natm} perturbations, tol {tol:.1e}; '
      f'make_h1 {t_h1:.1f} s', flush=True)
runs = {}
for name in ('cold', 'warm'):
    log_x, log_c = [], []
    t = time.perf_counter()
    x0 = None
    if name == 'warm':
        x0 = cphf.solve(counted(fx_c, log_c), mo_energy, mo_occ, h1vo, s1vo, tol=tol * args.cheap_tol)[0]
    t_c = time.perf_counter() - t
    mo1, e1 = cphf.solve(counted(fx, log_x), mo_energy, mo_occ, h1vo, s1vo, tol=tol, x0=x0)
    total = time.perf_counter() - t
    runs[name] = (mo1, e1)
    msg = (f'  {name}: exact fvind {len(log_x)} calls, {sum(n for n, _ in log_x)} vectors, '
           f'{sum(s for _, s in log_x):.1f} s')
    if log_c:
        msg += (f'; cheap fvind {len(log_c)} calls, {sum(n for n, _ in log_c)} vectors, '
                f'{sum(s for _, s in log_c):.1f} s')
    print(msg + f'; total {total:.1f} s; true residual {residual(mo1):.2e}', flush=True)
d1 = abs(runs['cold'][0] - runs['warm'][0]).max()
de = abs(runs['cold'][1] - runs['warm'][1]).max()
print(f'  cold vs warm: max|d mo1| {d1:.2e}, max|d e1| {de:.2e}')
