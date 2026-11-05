"""Prototype for PCM's surface integrals through PySCF's SGX driver, against pcm.PCM's int3c2e path.
The new path is int1e_grids with Gaussian charges: env[PTR_RINV_ZETA] is set to the exponent of each group of
surface points with equal charge exponents, and the integrals are contracted on the fly with s2 symmetry
(SGXnrs2_ijg_ji_g for the potential on the surface, SGXnrs2_ijg_g_ij for the Fock term). The reference is
_get_v/_get_vmat (int3c2e with fakemol, s1, numpy.einsum).
Prints the number of exponent groups, the largest differences and the times, and the time of _get_vind's two dense
solves against one Cholesky factorization (C-PCM: K = S). Run with PYTHONPATH set to the tree.
Usage: pcm_int1e_proto.py [--n 26] [--basis def2-tzvp] [--tol 1e-15]"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy  # noqa: E402
import scipy.linalg  # noqa: E402
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=26)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--tol', type=float, default=1e-15, help='shell-pair overlap cutoff of the SGX prescreen')
args = ap.parse_args()

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
nao = mol.nao
p = pcm.PCM(mol)
p.build()
coords = p.surface['grid_coords']
charge_exp = p.surface['charge_exp']
ngrids = charge_exp.size
uniq, inv = numpy.unique(charge_exp, return_inverse=True)
groups = [numpy.where(inv == k)[0] for k in range(uniq.size)]
print(f'C{args.n}H{2*args.n+2}/{args.basis}: {nao} AOs, {ngrids} surface points, {uniq.size} distinct charge '
      f'exponents; {lib.num_threads()} threads', flush=True)
dm = dft.RKS(mol).get_init_guess()
jk = sgx_jk._gen_jk_direct(mol, 's2', True, False, args.tol)


def v_on_surface(dm):
    '''sum_ij D_ij (ij|g), g the normalized Gaussian charge of exponent charge_exp[g]**2'''
    v = numpy.empty(ngrids)
    try:
        for k, idx in enumerate(groups):
            mol._env[gto.PTR_RINV_ZETA] = uniq[k]**2
            v[idx] = jk(mol, coords[idx], [dm], None, None)[0][0]
    finally:
        mol._env[gto.PTR_RINV_ZETA] = 0.
    return v


def vmat_of_charges(q):
    '''-sum_g q_g (ij|g), as pcm.PCM._get_vmat'''
    vmat = numpy.zeros((nao, nao))
    try:
        for k, idx in enumerate(groups):
            mol._env[gto.PTR_RINV_ZETA] = uniq[k]**2
            vmat += jk(mol, coords[idx], [numpy.ascontiguousarray(q[idx])], None, None)[0][0]
    finally:
        mol._env[gto.PTR_RINV_ZETA] = 0.
    # the s2 driver fills the lower triangle, as in sgx_jk.get_jk_favorj
    lib.hermi_triu(vmat, inplace=True)
    return -vmat


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


v_ref, t_v_ref = timed(lambda: p._get_v(dm[None])[0])
v_new, t_v_new = timed(v_on_surface, dm)
print(f'potential on the surface: int3c2e {t_v_ref:.1f} s, int1e_grids {t_v_new:.1f} s ({t_v_ref / t_v_new:.1f}x); '
      f'max |dv| {abs(v_new - v_ref).max():.2e} (max |v| {abs(v_ref).max():.2e})', flush=True)

K = p._intermediates['K']
R = p._intermediates['R']
v_grids = p.v_grids_n - v_ref
(q, qt), t_solve = timed(lambda: (numpy.linalg.solve(K, R.dot(v_grids)),
                                  R.T.dot(numpy.linalg.solve(K.T, v_grids))))
q_sym = (q + qt) / 2
(cho, q_cho), t_cho = timed(lambda: (lambda c: (c, scipy.linalg.cho_solve(c, R.dot(v_grids))))(
    scipy.linalg.cho_factor(K)))
_, t_cho_solve = timed(scipy.linalg.cho_solve, cho, R.dot(v_grids))
print(f'solves: two numpy.linalg.solve {t_solve:.1f} s; one Cholesky factorization {t_cho:.1f} s, then '
      f'{t_cho_solve:.2f} s a solve; max |dq| {abs(q_cho - q_sym).max():.2e} (max |q| {abs(q_sym).max():.2e})',
      flush=True)

vmat_ref, t_m_ref = timed(lambda: p._get_vmat(q_sym[None])[0])
vmat_new, t_m_new = timed(vmat_of_charges, q_sym)
print(f'Fock term: int3c2e {t_m_ref:.1f} s, int1e_grids {t_m_new:.1f} s ({t_m_ref / t_m_new:.1f}x); '
      f'max |dV| {abs(vmat_new - vmat_ref).max():.2e} (max |V| {abs(vmat_ref).max():.2e}), '
      f'asymmetry {abs(vmat_new - vmat_new.T).max():.2e}', flush=True)
e_ref = 0.5 * q_sym.dot(v_grids)
e_new = 0.5 * q_sym.dot(p.v_grids_n - v_new)
print(f'E_diel at this density: {e_ref:.12f} (int3c2e), {e_new:.12f} (int1e_grids), diff {e_new - e_ref:.1e}; '
      f'tr(D V) diff {numpy.einsum("ij,ij", dm, vmat_new - vmat_ref):.1e}', flush=True)
