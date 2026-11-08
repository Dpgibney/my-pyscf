'''How far the Hessian moves when only the CPHF response's J and K are approximated, by DF-J (the J-engine) and
SGX-K on grid level L (an RIJCOSX copy of the SCF object), with the XC and VV10 responses, the SCF, and every
derivative integral exact: max |dH| against the exact Hessian, and the harmonic frequencies' differences.

Usage: rijcosx_resp_accuracy.py [--mol water|propane|alkane:N] [--basis def2-svp] [--xc wb97m-v] [--levels 0,1,2]'''
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.hessian import rhf as rhf_hess, thermo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='propane')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--levels', default='0,1,2')
ap.add_argument('--auxbases', default='',
                help='comma-separated fitting bases to try for DF-J with exact K (e.g. def2-universal-jkfit,autoaux)')
ap.add_argument('--converge', action='store_true',
                help="converge every CPHF: lib.krylov with normalized vectors (cphf_residual.py's), conv_tol_cpscf 1e-12; "
                     "PySCF's stops short for DFT, by amounts that move the Hessian as much as the approximation")
ap.add_argument('--split', action='store_true',
                help='also DF-J with exact K (density_fit only_dfj), and exact J with SGX-K on each level')
args = ap.parse_args()
if args.mol == 'water':
    atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
elif args.mol.startswith('alkane:'):
    atom = n_alkane(int(args.mol.split(':')[1]))
else:
    atom = n_alkane(3)
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.nlcgrids.level = 1
mf.conv_tol = 1e-11
mf.kernel()


def krylov_norm(aop, b, x0=None, tol=1e-10, max_cycle=60, dot=numpy.dot, lindep=1e-14,
                callback=None, hermi=False, max_memory=4000, verbose=0):
    '''lib.krylov with normalized Krylov vectors (as bench/diag/cphf_residual.py): the norms against which the
    solver drops vectors and stops are relative to their parents rather than absolute'''
    from pyscf.lib.linalg_helper import _qr
    if x0 is not None:
        b = b - (x0 + aop(x0))
    x1 = b.reshape(-1, b.shape[-1])
    ndim = x1.shape[1]
    x1 = _qr(x1, dot, lindep)[0]
    if len(x1) == 0:
        return numpy.zeros_like(b) if x0 is None else x0
    xs, ax = [], []
    for cycle in range(max_cycle):
        axt = aop(x1).reshape(-1, ndim)
        xs.extend(x1)
        ax.extend(axt)
        if callable(callback):
            callback(cycle, xs, ax)
        x1 = axt.copy()
        for rep in range(2):  # two passes of Gram-Schmidt against the space
            for xsi in xs:
                x1 -= xsi * dot(x1, xsi.conj())[:, None]
        x1, rmat = _qr(x1, dot, lindep)
        norms = abs(rmat.diagonal())
        if len(x1) == 0 or norms.max() < tol:
            break
        x1 = x1[norms > tol]
    else:
        raise RuntimeError('Krylov solver failed to converge.')
    nd = len(xs)
    h = numpy.array([[dot(xs[i].conj(), ax[j]) for j in range(nd)] for i in range(nd)]) + numpy.eye(nd)
    g = numpy.array([dot(b.reshape(-1, ndim), xs[i].conj()) for i in range(nd)])
    x = numpy.dot(numpy.linalg.solve(h, g).T, numpy.asarray(xs)).reshape(b.shape)
    if x0 is not None:
        x += x0
    return x


if args.converge:
    lib.krylov = krylov_norm
    mf.conv_tol_cpscf = 1e-12


def freqs(h):
    return thermo.harmonic_analysis(mol, h)['freq_wavenumber']


t = time.perf_counter()
h_x = mf.Hessian().kernel()
t_x = time.perf_counter() - t
f_x = freqs(h_x)
print(f'{args.mol}/{args.basis} {args.xc}: nao {mol.nao}, {3 * mol.natm} perturbations; exact Hessian {t_x:.1f} s, '
      f'max|H| {abs(h_x).max():.4f}, frequencies {f_x.min():.1f}..{f_x.max():.1f} cm-1', flush=True)

gen_vind = rhf_hess.gen_vind


def hess_with(mf_c, label):
    def approx_vind(mf_, mo_coeff, mo_occ):
        return gen_vind(mf_c, mo_coeff, mo_occ)
    rhf_hess.gen_vind = approx_vind
    try:
        t = time.perf_counter()
        h = mf.Hessian().kernel()
        dt = time.perf_counter() - t
    finally:
        rhf_hess.gen_vind = gen_vind
    df = freqs(h) - f_x
    print(f'  {label}: {dt:.1f} s; max|dH| {abs(h - h_x).max():.2e} ({abs(h - h_x).max() / abs(h_x).max():.1e} of '
          f'max|H|); frequencies: max|df| {abs(df).max():.2f} cm-1, mean|df| {abs(df).mean():.2f} cm-1', flush=True)


def sgx_copy(level, dfj=True):
    mf_c = mf.RIJCOSX()
    sgx = mf_c.with_df
    if not dfj:
        sgx.dfj = False
    sgx.grids_level_i = sgx.grids_level_f = level
    sgx.build(level=level)
    mf_c._nsteps_direct = 0  # the SCF state of an SGX object used outside an SCF
    mf_c._in_scf = False
    return mf_c


if args.split:
    hess_with(mf.density_fit(auxbasis='def2-universal-jfit', only_dfj=True), 'DF-J + exact K')
for aux in [a for a in args.auxbases.split(',') if a]:
    if aux == 'autoaux':
        from pyscf.df import autoaux
        aux_b = autoaux(mol)
    elif aux == 'etb':
        from pyscf.df import aug_etb
        aux_b = aug_etb(mol)
    else:
        aux_b = aux
    mf_d = mf.density_fit(auxbasis=aux_b, only_dfj=True)
    mf_d.with_df.build()
    hess_with(mf_d, f'DF-J ({aux}, {mf_d.with_df.auxmol.nao} functions) + exact K')
for level in [int(x) for x in args.levels.split(',')]:
    mf_c = sgx_copy(level)
    hess_with(mf_c, f'DF-J + SGX-K level {level} ({mf_c.with_df.grids.weights.size} points)')
    if args.split:
        hess_with(sgx_copy(level, dfj=False), f'exact J + SGX-K level {level}')
