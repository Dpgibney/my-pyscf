"""The derivative J/K calls of the gradient and the Hessian, with their scripts: grad.rhf.get_jk
(int2e_ip1 s2kl), make_h1 (int2e_ip1 s2kl, RKS and UKS scripts), and _partial_hess_2e's int2e_ip1ip2
(s1) and int2e_ipvip1 (s2kl). Saves the matrices to compare trees bit for bit (run at 1 thread), and
prints the time of each call (min of 3).
Usage: deriv_jk_kernels.py SYSTEM OUT.npz [ATOM ...]"""
import os
import sys
import time
import numpy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import WATER, BENZENE, CAFFEINE  # noqa: E402
from pyscf import gto, scf  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402
from pyscf.hessian.rhf import _get_jk  # noqa: E402

system, out = sys.argv[1:3]
atom = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
atoms = [int(a) for a in sys.argv[3:]] or [0, mol.natm // 2]
nao = mol.nao
nbas = mol.nbas
aoslices = mol.aoslice_by_atom()
dm = scf.RHF(mol).get_init_guess(key='minao')
dm = (dm + dm.T) * .5
rng = numpy.random.default_rng(1)
x = rng.standard_normal((nao, nao)) * 1e-2
dma = dm * .55 + (x + x.T)
dmb = dm * .45 - (x + x.T) * .5

calls = {}
calls['grad_rhf'] = lambda: rhf_grad.get_jk(mol, dm)
calls['grad_uhf'] = lambda: rhf_grad.get_jk(mol, numpy.asarray([dma, dmb]))
for ia in atoms:
    shl0, shl1, p0, p1 = aoslices[ia]
    ksh1, nk = aoslices[ia][1], aoslices[ia][3]
    calls[f'h1_rks_{ia}'] = (lambda shl0=shl0, shl1=shl1, p0=p0, p1=p1: _get_jk(
        mol, 'int2e_ip1', 3, 's2kl',
        ['ji->s2kl', -dm[:, p0:p1], 'lk->s1ij', -dm, 'li->s1kj', -dm[:, p0:p1], 'jk->s1il', -dm],
        shls_slice=(shl0, shl1) + (0, nbas) * 3))
    calls[f'h1_uks_{ia}'] = (lambda shl0=shl0, shl1=shl1, p0=p0, p1=p1: _get_jk(
        mol, 'int2e_ip1', 3, 's2kl',
        ['ji->s2kl', -(dma + dmb)[:, p0:p1], 'lk->s1ij', -(dma + dmb),
         'li->s1kj', -dma[:, p0:p1], 'jk->s1il', -dma, 'li->s1kj', -dmb[:, p0:p1], 'jk->s1il', -dmb],
        shls_slice=(shl0, shl1) + (0, nbas) * 3))
    calls[f'ip1ip2_{ia}'] = (lambda shl0=shl0, shl1=shl1, p0=p0, p1=p1, ksh1=ksh1: _get_jk(
        mol, 'int2e_ip1ip2', 9, 's1',
        ['ji->s1kl', dm[:, p0:p1], 'li->s1kj', dm[:, p0:p1], 'lj->s1ki', dm],
        shls_slice=(shl0, shl1, 0, nbas, 0, ksh1, 0, nbas)))
    calls[f'ipvip1_{ia}'] = (lambda shl0=shl0, shl1=shl1, p0=p0, p1=p1, ksh1=ksh1: _get_jk(
        mol, 'int2e_ipvip1', 9, 's2kl', ['lk->s1ij', dm, 'li->s1kj', dm[:, p0:p1]],
        shls_slice=(shl0, shl1, 0, ksh1, 0, nbas, 0, nbas)))

res = {}
for name, f in calls.items():
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        v = f()
        ts.append(time.perf_counter() - t0)
    for n, a in enumerate(v):
        res[f'{name}_{n}'] = numpy.asarray(a)
    print(f'{name:14s} {min(ts):.3f} s', flush=True)
numpy.savez(out, **res)
