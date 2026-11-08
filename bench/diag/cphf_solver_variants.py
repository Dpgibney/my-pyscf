'''The prototype of cphf.solve_residual, with the choices it did not take. A CPHF solver that stops on the true residual: subspace shared by the perturbations, orthonormal
trial vectors (2 Gram-Schmidt passes), trial vectors = preconditioned residuals of the unconverged perturbations.

Tries, on one SCF (partial Hessian and h1ao cached): start from the uncoupled x0 (one call, as cphf.solve) or from the
occupied block only (no offset, one call more), Galerkin in the plain dot on M = D + K (symmetric) or on the
preconditioned operator, with or without the final update of the virtual block. Prints calls, vectors, residuals,
and the errors of the Hessian and the frequencies against a tight solve with the same response.
'''
import argparse
import os
import sys
import time
import numpy
import scipy.linalg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.lib import logger  # noqa: E402
from pyscf.hessian import rhf as rhf_hess, thermo  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='propane')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--resp', default='exact', help='exact, sgx0, sgx1')
ap.add_argument('--tols', default='1e-2,1e-3,1e-4,1e-5,1e-6,1e-8')
ap.add_argument('--variants', default='x0:M:0,x0:M:1,x0:D:0,x0:D:1,oo:M:0')
ap.add_argument('--stock', type=int, default=1)
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
t0 = time.perf_counter()
mf.kernel()
mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
natm = mol.natm


def orth(t, Vs, drop):
    for _ in range(2):
        for V in Vs:
            t -= lib.dot(lib.dot(t, V.T), V)
    q, r, piv = scipy.linalg.qr(t.T, mode='economic', pivoting=True)
    n = int((abs(r.diagonal()) > drop).sum())
    q = q[:, :n].T
    for V in Vs:
        q -= lib.dot(lib.dot(q, V.T), V)
    return q


CFG = {}
STAT = {}


def solve_r(fvind, mo_energy, mo_occ, h1, s1=None, max_cycle=50, tol=1e-9, hermi=False, verbose=0, level_shift=0,
            x0=None):
    rtol, start, metric, last = CFG['tol'], CFG['start'], CFG['metric'], CFG['last']
    occ, vir = mo_occ > 0, mo_occ == 0
    e_i, e_a = mo_energy[occ], mo_energy[vir]
    D = e_a[:, None] - e_i
    nvir, nocc = D.shape
    nmo = nvir + nocc
    ndim = nvir * nocc
    Df = D.ravel()
    s1 = s1.reshape(-1, nmo, nocc)
    hs = h1.reshape(-1, nmo, nocc) - s1 * e_i
    nset = hs.shape[0]

    calls, vecs, rhist = [0], [0], []

    def fv(m):
        calls[0] += 1
        vecs[0] += len(m)
        return fvind(m).reshape(-1, nmo, nocc)

    x = numpy.zeros_like(hs)
    x[:, occ] = -s1[:, occ] * .5
    if start == 'x0':
        x[:, vir] = -hs[:, vir] / D
    v0 = fv(x)
    if start == 'x0':
        R0 = -v0[:, vir].reshape(nset, ndim)
    else:
        R0 = -(hs[:, vir] + v0[:, vir]).reshape(nset, ndim)

    Vs, MVs, Woo = [], [], []
    S = numpy.zeros((0, 0))
    P = numpy.zeros((0, nset))
    R = R0
    tn = numpy.linalg.norm(R / Df, axis=1)
    rhist.append(tn.max())
    active = tn > rtol
    drop = max(rtol * 1e-2, 1e-13)
    c = None
    for cycle in range(max_cycle):
        if not active.any():
            break
        t = (R[active] / Df).copy()
        q = orth(t, Vs, drop)
        if len(q) == 0:
            print('   no new vectors', flush=True)
            break
        nq = len(q)
        m = numpy.zeros((nq, nmo, nocc))
        m[:, vir] = q.reshape(nq, nvir, nocc)
        w = fv(m)
        mq = w[:, vir].reshape(nq, ndim) + q * Df
        Woo.append(w[:, occ])
        w = None
        nd = S.shape[0]
        Snew = numpy.empty((nd + nq, nd + nq))
        Snew[:nd, :nd] = S
        Vs.append(q)
        MVs.append(mq)
        if metric == 'M':
            tv = q
            rows = [lib.dot(V, mq.T) for V in Vs[:-1]]
            cols = [lib.dot(q, mq0.T) for mq0 in MVs]
        else:
            tv = q / Df
            rows = [lib.dot(V / Df, mq.T) for V in Vs[:-1]]
            cols = [lib.dot(tv, mq0.T) for mq0 in MVs]
        p0 = 0
        for row in rows:
            Snew[p0:p0 + len(row), nd:] = row
            p0 += len(row)
        p0 = 0
        for col in cols:
            Snew[nd:, p0:p0 + col.shape[1]] = col
            p0 += col.shape[1]
        S = Snew
        P = numpy.vstack((P, lib.dot(tv, R0.T)))
        c = numpy.linalg.solve(S, P)
        R = R0.copy()
        p0 = 0
        for mv in MVs:
            n = len(mv)
            R -= lib.dot(c[p0:p0 + n].T, mv)
            p0 += n
        tn = numpy.linalg.norm(R / Df, axis=1)
        rhist.append(tn.max())
        active = tn > rtol
    else:
        print('   not converged', flush=True)

    x = x.copy()
    v1 = v0.copy()
    if Vs:
        p0 = 0
        for k, V in enumerate(Vs):
            n = len(V)
            ck = c[p0:p0 + n].T
            xv = lib.dot(ck, V)
            x[:, vir] += xv.reshape(nset, nvir, nocc)
            v1[:, vir] += (lib.dot(ck, MVs[k]) - xv * Df).reshape(nset, nvir, nocc)
            v1[:, occ] += numpy.einsum('rk,kij->rij', ck, Woo[k])
            p0 += n
    STAT.update(calls=calls[0], vecs=vecs[0], r=rhist, nd=sum(len(V) for V in Vs))
    h = hs + v1
    if last:
        x[:, vir] = h[:, vir] / (e_i - e_a[:, None])
    mo_e1 = h[:, occ]
    mo_e1 += x[:, occ] * (e_i[:, None] - e_i)
    return x, mo_e1


stock_solve = cphf.solve
gen_vind0 = rhf_hess.gen_vind

if args.resp != 'exact':
    level = int(args.resp[3:])
    mf_c = mf.RIJCOSX()
    sgx = mf_c.with_df
    sgx.dfj = False
    sgx.direct_j = True
    sgx.grids_level_i = sgx.grids_level_f = level
    sgx.build(level=level)
    mf_c._nsteps_direct = 0
    mf_c._in_scf = False
    rhf_hess.gen_vind = lambda mf_, c_, o_: gen_vind0(mf_c, c_, o_)

hobj = mf.Hessian()
log = logger.new_logger(hobj, 0)
atmlst = list(range(natm))
t1 = time.perf_counter()
de2p = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, 4000, log)
h1ao = hobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
print(f'{args.mol}/{args.basis} {args.xc} {args.resp}: nao {mol.nao}, {3*natm} perturbations, '
      f'nvir*nocc {(mo_occ==0).sum()*(mo_occ>0).sum()}; SCF {t1-t0:.1f} s, partial + h1 {time.perf_counter()-t1:.1f} s',
      flush=True)
hobj.partial_hess_elec = lambda *a, **k: de2p.copy()
nuc = hobj.hess_nuc(mol, atmlst=atmlst)


def hess():
    return rhf_hess.hess_elec(hobj, mo_energy, mo_coeff, mo_occ, h1ao=h1ao, atmlst=atmlst, max_memory=4000,
                              verbose=0) + nuc


def freqs(h):
    return thermo.harmonic_analysis(mol, h)['freq_wavenumber']


def show(label, h, dt, stat):
    d = abs(h - h_ref).max()
    df = freqs(h) - f_ref
    r = ' '.join(f'{x:.1e}' for x in stat.get('r', []))
    print(f'  {label:18s} {dt:6.1f} s calls {stat.get("calls", 0):3d} vecs {stat.get("vecs", 0):5d} | max|dH| {d:.2e} '
          f'max|df| {abs(df).max():7.4f} cm-1 | r: {r}', flush=True)


cphf.solve = solve_r
CFG.update(tol=1e-10, start='oo', metric='M', last=0)
t = time.perf_counter()
h_ref = hess()
f_ref = freqs(h_ref)
print(f'  reference (oo:M:0 tol 1e-10): {time.perf_counter()-t:.1f} s calls {STAT["calls"]} vecs {STAT["vecs"]}, '
      f'freq {f_ref.min():.1f}..{f_ref.max():.1f}', flush=True)

if args.stock:
    cphf.solve = stock_solve
    n = [0, 0]
    gv = rhf_hess.gen_vind

    def counted(mf_, c_, o_):
        f = gv(mf_, c_, o_)

        def g(m):
            n[0] += 1
            n[1] += len(m.reshape(-1, c_.shape[1] * (o_ > 0).sum()))
            return f(m)
        return g
    rhf_hess.gen_vind = counted
    t = time.perf_counter()
    h = hess()
    show('stock', h, time.perf_counter() - t, dict(calls=n[0], vecs=n[1]))
    rhf_hess.gen_vind = gv
    cphf.solve = solve_r

for v in args.variants.split(','):
    start, metric, last = v.split(':')
    for tol in [float(x) for x in args.tols.split(',')]:
        CFG.update(tol=tol, start=start, metric=metric, last=int(last))
        t = time.perf_counter()
        h = hess()
        show(f'{v} {tol:.0e}', h, time.perf_counter() - t, STAT)
