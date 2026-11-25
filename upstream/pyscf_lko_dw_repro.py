#!/usr/bin/env python
'''Reproducer for the error in PySCF's derivative of the LKO grid weights (VXCgen_grid_lko_deriv in
pyscf/lib/dft/grid_basis.c), with an unmodified PySCF on the CPU (no GPU needed). See pyscf-lko-weight-derivative.md.

The quadrature of a fixed smooth function, Q(R) = sum_g w_g phi(r_g), on the SGX grid (LKO partition), each point moving
with its atom, differentiated analytically with pyscf.grad.rks.get_dw_partition_sorted (the routine of the SGX
gradients) and by finite differences. With the bug the two differ by about 3.5e-6 here; with the corrected factor by
about 2.6e-10 (the finite-difference error).

    python pyscf_lko_dw_repro.py
'''
import numpy
import pyscf
from pyscf.sgx import sgx_jk
from pyscf.grad import rks as rks_grad

# a water and a CH2 fragment: atom pairs from 1.8 to 4.6 Bohr apart (Bohr)
atom0 = [['O', (0., 0., 0.)], ['H', (0., -1.43, 1.11)], ['H', (0.19, 1.43, 1.11)],
         ['C', (0.2, 3.6, 1.0)], ['H', (1.9, 4.2, 0.4)]]
cen = numpy.array([[0.3, 0.5, 0.4], [0.5, 2.6, 1.2], [1.0, 3.9, 0.6]])
ex = numpy.array([0.7, 0.9, 1.3])

def phi(r):
    d = r[:,None,:] - cen[None]
    return numpy.exp(-ex * (d**2).sum(axis=2)).sum(axis=1)

def dphi(r):
    d = r[:,None,:] - cen[None]
    e = numpy.exp(-ex * (d**2).sum(axis=2))
    return (-2 * ex[None,:,None] * d * e[:,:,None]).sum(axis=1)

def grids_of(atom):
    mol = pyscf.M(atom=atom, basis='def2-svp', unit='B', charge=1, verbose=0)
    # the grids of PySCF's SGX (level 1, its ORCA-like grids with the LKO partition), no screening of points
    return sgx_jk.get_gridss(mol, 1, 0., True), mol

g, mol = grids_of(atom0)
keep = g.weights != 0
r, w, atm = g.coords[keep], g.weights[keep], g.atm_idx[keep]
dw = rks_grad.get_dw_partition_sorted(mol, atm.astype(numpy.int32), numpy.ascontiguousarray(r),
                                      numpy.ascontiguousarray(w), g.radii_adjust, g.atomic_radii)
# the weights' derivatives, and the points' motion with their atoms
ana = numpy.einsum('xag,g->ax', dw, phi(r))
gp = w[:,None] * dphi(r)
for x in range(3):
    ana[:,x] += numpy.bincount(atm, gp[:,x], minlength=mol.natm)

fd = numpy.zeros_like(ana)
h = 1e-4
for ia in range(mol.natm):
    for x in range(3):
        e = []
        for s in (1, -1):
            atom = [[a, list(rr)] for a, rr in atom0]
            atom[ia][1][x] += s * h
            gd, _ = grids_of(atom)
            e.append(float((gd.weights * phi(gd.coords)).sum()))
        fd[ia,x] = (e[0] - e[1]) / (2 * h)

print('PySCF', pyscf.__version__)
print('max |analytic - finite differences| = %.2e  (about 3.5e-6 with the bug, 2.6e-10 without)' % abs(ana - fd).max())
print('analytic - finite differences by atom:')
print(numpy.array2string(ana - fd, precision=2))
