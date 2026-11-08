'''The true residual of the CPHF solution that cphf.solve returns for a Hessian (RKS/RHF): with one more fvind
call made here, max |b - (1 + a) mo1| on the virtual rows, (e_a - e_i) mo1 + fvind(mo1) + h1 - s1 e_i = 0.

Usage: cphf_residual.py [--mol water|propane] [--basis def2-svp] [--xc b3lyp] (CPSCF_TOL env: conv_tol_cpscf)'''
import argparse
import os
import sys
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, scf  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='water')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
args = ap.parse_args()
atom = {'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', 'propane': n_alkane(3)}[args.mol]
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = scf.RHF(mol) if args.xc == 'hf' else dft.RKS(mol, xc=args.xc)
if args.xc != 'hf':
    mf.nlcgrids.level = 1
mf.conv_tol = 1e-11
if os.environ.get('CPSCF_TOL'):
    mf.conv_tol_cpscf = float(os.environ['CPSCF_TOL'])
mf.kernel()

def krylov_norm(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=numpy.dot, lindep=1e-14,
                callback=None, hermi=False, max_memory=4000, verbose=0):
    '''lib.krylov with normalized Krylov vectors: the norms of the new vectors, against which the solver drops
    vectors and stops, are relative to their parents rather than absolute'''
    from pyscf.lib.linalg_helper import _qr
    if x0 is not None:
        b = b - (x0 + aop(x0))
    x1 = b.reshape(-1, b.shape[-1])
    nroots, ndim = x1.shape
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
                x1 -= xsi * dot(x1, xsi.conj())[:,None]
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
    print(f'krylov_norm: {cycle+1} cycles, {nd} vectors', flush=True)
    return x

if os.environ.get('KRYLOV_NORM'):
    from pyscf import lib
    lib.krylov = krylov_norm

solve0 = cphf.solve
def solve(fvind, mo_energy, mo_occ, h1, s1=None, *a, **k):
    mo1, e1 = solve0(fvind, mo_energy, mo_occ, h1, s1, *a, **k)
    occ, vir = mo_occ > 0, mo_occ == 0
    e_i, e_a = mo_energy[occ], mo_energy[vir]
    nset = h1.shape[0]
    v = fvind(mo1).reshape(mo1.shape)
    hs = h1.reshape(mo1.shape) - s1.reshape(mo1.shape) * e_i
    r = (e_a[:,None] - e_i) * mo1[:,vir] + v[:,vir] + hs[:,vir]
    print(f'residual of the returned mo1: max {abs(r).max():.3e}, max over sets of the norm '
          f'{numpy.linalg.norm(r.reshape(nset,-1), axis=1).max():.3e}; max|mo1| {abs(mo1[:,vir]).max():.3e}', flush=True)
    return mo1, e1
cphf.solve = solve
h = mf.Hessian().kernel()
print(f'{args.mol}/{args.basis} {args.xc}: max|H| {abs(h).max():.8f}')
