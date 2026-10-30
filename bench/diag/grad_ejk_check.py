"""The two-electron gradient from grad.rhf.jk_energy_per_atom against the J/K matrices of get_jk
contracted as grad_elec does, and times of both.
Usage: grad_ejk_check.py SYSTEM [BASIS]  (SYSTEM: water, benzene, caffeine, c20h42)"""
import sys
import os
import time
import numpy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import WATER, BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, scf, lib  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402

system = sys.argv[1]
basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-svp'
atom = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE,
        'c20h42': n_alkane(20)}[system]
mol = gto.M(atom=atom, basis=basis, verbose=0)
nao = mol.nao
aoslices = mol.aoslice_by_atom()

# A density with the sparsity of a real one: the converged-looking minao guess, made symmetric;
# spin densities from it split unevenly
dm = scf.RHF(mol).get_init_guess(key='minao')
dm = (dm + dm.T) * .5
rng = numpy.random.default_rng(1)
x = rng.standard_normal((nao, nao)) * 1e-2
dma = dm * .55 + (x + x.T)
dmb = dm * .45 - (x + x.T) * .5
dms = numpy.asarray([dma, dmb])


def contract(v, d):
    de = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2:]
        if v.ndim == 4:
            de[ia] = numpy.einsum('sxij,sij->x', v[:, :, p0:p1], d[:, p0:p1]) * 2
        else:
            de[ia] = numpy.einsum('xij,ij->x', v[:, p0:p1], d[p0:p1]) * 2
    return de


def old(d, jf, kf, omega=None):
    with mol.with_range_coulomb(omega):
        if kf == 0:
            vj = rhf_grad.get_j(mol, d)
            v = jf * (vj if d.ndim == 2 else vj[0] + vj[1])
        elif jf == 0:
            vk = rhf_grad.get_k(mol, d)
            v = -kf * (vk * .5 if d.ndim == 2 else vk)
        else:
            vj, vk = rhf_grad.get_jk(mol, d)
            if d.ndim == 2:
                v = jf * vj - kf * .5 * vk
            else:
                v = jf * (vj[0] + vj[1]) - kf * vk
    if d.ndim == 3 and v.ndim == 3:
        v = numpy.asarray([v, v])
    return contract(v, d)


cases = [('RHF J+K', dm, 1., 1., None), ('RHF J', dm, 1., 0., None), ('RHF K', dm, 0., 1., None),
         ('RKS B3LYP', dm, 1., .2, None), ('RKS LR-K w=0.33', dm, 0., .46, .33),
         ('UHF J+K', dms, 1., 1., None), ('UHF J', dms, 1., 0., None),
         ('UKS LR-K w=0.3', dms, 0., .8, .3)]
if system in ('caffeine', 'c20h42'):
    cases = cases[:1] + cases[5:6]
for name, d, jf, kf, omega in cases:
    t0 = time.perf_counter()
    ref = old(d, jf, kf, omega)
    t1 = time.perf_counter()
    new = rhf_grad.jk_energy_per_atom(mol, d, jf, kf, omega=omega)
    t2 = time.perf_counter()
    new2 = rhf_grad.jk_energy_per_atom(mol, d, jf, kf, omega=omega)
    diff = abs(new - ref).max()
    print(f'{name:18s} max|d| {diff:.2e}  max|de| {abs(ref).max():.2e}  sum(new) {abs(new.sum(axis=0)).max():.1e}'
          f'  repeat {abs(new2 - new).max():.0e}  old {t1 - t0:.2f} s  new {t2 - t1:.2f} s', flush=True)
