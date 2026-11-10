'''Prototype: the XC part of the nuclear gradient contracted with the density on the grid, instead of three
nao x nao matrices per grid block that are contracted with the density afterwards (grad.rks.get_vxc).
  de[A,x] = -2 sum_{i in A} sum_g { d_x ao_i sum_c wv_c M_c[g,i] + (wv_0 d_x ao_i + sum_y wv_{1+y} d_x d_y ao_i) M_0[g,i]
                                    + wv_4 sum_k d_x d_k ao_i M_{1+k}[g,i] },   M_c = ao_c D = 2 (ao_c C_occ) C_occ^T
(wv = weight * vxc, wv_0 and wv_4 halved). numpy/BLAS only; checked against get_vxc contracted with D.
Usage: xc_grad_proto.py MOL NPZ [--grid 99,590]'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('npz')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
z = numpy.load(args.npz)
mf = dft.RKS(mol, xc=args.xc)
if ',' in args.grid:
    mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
else:
    mf.grids.level = int(args.grid)
mf.mo_coeff, mf.mo_occ = z['mo_coeff'], z['mo_occ']
mocc = numpy.asarray(mf.mo_coeff[:, mf.mo_occ > 0], order='C')
dm = mf.make_rdm1()
ni = mf._numint
grids = mf.grids
grids.build(with_non0tab=True)
nao = mol.nao
aoslices = mol.aoslice_by_atom()
print(f'{args.mol}/{args.basis}: nao {nao}, nocc {mocc.shape[1]}, {grids.weights.size} grid points', flush=True)

# the current code
for rep in range(2):
    t = time.perf_counter()
    exc, vxc = rks_grad.get_vxc(ni, mol, grids, args.xc, dm, max_memory=8000)
    t_now = time.perf_counter() - t
de_now = numpy.array([numpy.einsum('xij,ij->x', vxc[:, p0:p1], dm[p0:p1]) * 2 for p0, p1 in aoslices[:, 2:]])
t = time.perf_counter()
n, e, v = ni.nr_rks(mol, grids, args.xc, dm, max_memory=8000)
t_scf = time.perf_counter() - t
print(f'grad.rks.get_vxc: {t_now:.2f} s (an SCF nr_rks call: {t_scf:.2f} s)', flush=True)

# d_x d_y component of ao (deriv 2): 4..9 = xx xy xz yy yz zz
D2 = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
for rep in range(2):
    t0 = time.perf_counter()
    gao = numpy.zeros((3, nao))
    t_ao = t_mo = t_xc = t_sum = 0.
    t = time.perf_counter()
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 2, max_memory=500):
        t1 = time.perf_counter()
        t_ao += t1 - t
        psi = [lib.dot(ao[c], mocc) for c in range(4)]
        rho = numpy.empty((5, weight.size))
        rho[0] = 2 * numpy.einsum('gi,gi->g', psi[0], psi[0])
        for k in range(3):
            rho[1 + k] = 4 * numpy.einsum('gi,gi->g', psi[0], psi[1 + k])
        rho[4] = sum(numpy.einsum('gi,gi->g', psi[1 + k], psi[1 + k]) for k in range(3))
        t2 = time.perf_counter()
        t_mo += t2 - t1
        wv = weight * ni.eval_xc_eff(args.xc, rho, 1, xctype='MGGA')[1]
        wv[0] *= .5
        wv[4] *= .5
        t3 = time.perf_counter()
        t_xc += t3 - t2
        # P = sum_c wv_c M_c + wv_0 M_0 and Q_y = wv_{1+y} M_0 + wv_4 M_{1+y}, M_c = 2 psi_c C_occ^T: the weights go
        # onto the MO values first, so each is one product with C_occ^T
        a = 2 * wv[0, :, None] * psi[0]
        for c in range(1, 4):
            a += wv[c, :, None] * psi[c]
        P = lib.dot(a, mocc.T, 2)
        Q = [lib.dot(wv[1 + y, :, None] * psi[0] + wv[4, :, None] * psi[1 + y], mocc.T, 2) for y in range(3)]
        t4 = time.perf_counter()
        t_mo += t4 - t3
        for x in range(3):
            g = numpy.einsum('gi,gi->i', ao[1 + x], P)
            for y in range(3):
                g += numpy.einsum('gi,gi->i', ao[D2[x][y]], Q[y])
            gao[x] += g
        t = time.perf_counter()
        t_sum += t - t4
    de_new = numpy.array([gao[:, p0:p1].sum(axis=1) * 2 for p0, p1 in aoslices[:, 2:]])
    t_new = time.perf_counter() - t0
# get_vxc returns -vmat; de_now carries that sign
de_new = -de_new
nth = lib.num_threads()
print(f'density-contracted prototype (numpy): {t_new:.2f} s  (AO values to 2nd derivatives {t_ao:.2f}, MO products {t_mo:.2f}, '
      f'libxc {t_xc:.2f}, 12 column dot products per block in one numpy thread {t_sum:.2f})')
print(f'   with the column dot products on {nth} threads (in a C kernel): about {t_new - t_sum + t_sum / nth * 1.3:.2f} s')
print(f'max |de_new - de_now| {abs(de_new - de_now).max():.2e} (max |de| {abs(de_now).max():.3f})')
print('done')
