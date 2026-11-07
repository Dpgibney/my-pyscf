"""How much of the seminumerical exchange's work a given accuracy needs: the exact contribution of every shell pair
and grid batch to the exchange energy, against the pairs SGX's screening evaluates.

For the converged density D (an RKS SCF with density-fitted J and K, --mol/--basis/--xc) and SGX's grid of --level,
in batches of 224 points as SGX's K driver takes them: F = D X (X the AO values times the grid weights' square roots
are not used; SGX's F_ug = sum_v D_uv X_vg with X the AO values and w the weights), A_ij(g) the potential of the AO
pair ij at g. The energy of a shell pair (I,J) on a batch b is
    e_IJb = -1/4 * sum_{g in b} w_g sum_{i in I, j in J} A_ij(g) F_ig F_jg   (times 2 for I != J),
so the exchange energy is sum e_IJb. With --incr N, F of the density difference D - D_N (D_N the density of SCF cycle
N) on one side: the change of the exchange energy to first order, as an incremental K build would give it.
Dropping the pair-batches of smallest |e| first, prints how many must be kept for the sum of the dropped |e| to stay
under each tolerance, against all of them. Run with PYTHONPATH set to the tree.
Then, for screening by an estimate of each pair-batch's |e| against a threshold, the pair-batches kept and the
error of those dropped, with the estimates
  amax:  max over the batch of |A_ij(g)| (from the integrals: the best a per-batch bound on A can do) times the
         weighted norms of F over the batch, ||F_I||_b ||F2_J||_b (+ the transpose for --incr), as SGX forms them;
  sgx:   SGX's bound of A on the batch (sample_pos: rbar_IJ m_bI m_bJ) times the same norms.
Usage: cosx_oracle.py [--mol caffeine|c20h42] [--basis def2-tzvp] [--level 1] [--incr N]"""
import argparse
import os
import sys
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--level', type=int, default=1)
ap.add_argument('--incr', type=int, default=None)
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}.get(args.mol)
if atom is None:
    atom = n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc).density_fit()
dms = []
mf.callback = lambda envs: dms.append(envs['dm'])
mf.kernel()
dm = mf.make_rdm1()
grids = sgx_jk.get_gridss(mol, args.level, 1e-10, True)
coords, weights = grids.coords, grids.weights
ngrids = weights.size
ao_loc = mol.ao_loc_nr()
nbas = mol.nbas
blk = sgx_jk.SGX_BLKSIZE
nblk = (ngrids + blk - 1) // blk
print(f'{args.mol}/{args.basis} nao {mol.nao} nbas {nbas}, level {args.level}: {ngrids} points, {nblk} batches; '
      f'E_SCF {mf.e_tot:.10f}', flush=True)
dm2 = dm if args.incr is None else dm - dms[args.incr]
if args.incr is not None:
    print(f'incremental: D - D_{args.incr}, |ddm| {numpy.linalg.norm(dm2):.2e}')

# shell index of each AO
sh_of = numpy.repeat(numpy.arange(nbas), numpy.diff(ao_loc))
e_pb = numpy.zeros((nblk, nbas, nbas))
amax = numpy.zeros((nblk, nbas, nbas))
fb1 = numpy.zeros((nblk, nbas))
fb2 = numpy.zeros((nblk, nbas))
for b in range(nblk):
    g0, g1 = b * blk, min(ngrids, (b + 1) * blk)
    c, w = coords[g0:g1], weights[g0:g1]
    ao = mol.eval_gto('GTOval', c)                          # (g, nao)
    f = ao @ dm                                             # F_gi = sum_v X_gv D_vi
    f2 = f if args.incr is None else ao @ dm2
    a = mol.intor('int1e_grids', grids=c)                   # (g, nao, nao)
    # sum_g w_g A_ij(g) F_gi F2_gj, symmetrized for an incremental build
    e = numpy.einsum('gij,gi,gj->ij', a, f * w[:, None], f2)
    if args.incr is not None:
        e = e + e.T
    e *= -.25
    # reduce to shell pairs
    es = numpy.zeros((nbas, nbas))
    numpy.add.at(es, (sh_of[:, None], sh_of[None, :]), e)
    e_pb[b] = es
    aa = numpy.abs(a).max(axis=0)
    for I in range(nbas):
        amax[b, I] = numpy.maximum.reduceat(aa[ao_loc[I]:ao_loc[I + 1]].max(axis=0), ao_loc[:-1])
    fw1 = (f**2 * numpy.abs(w)[:, None]).sum(axis=0)
    fw2 = (f2**2 * numpy.abs(w)[:, None]).sum(axis=0)
    fb1[b] = numpy.sqrt(numpy.add.reduceat(fw1, ao_loc[:-1]))
    fb2[b] = numpy.sqrt(numpy.add.reduceat(fw2, ao_loc[:-1]))
    if b % 20 == 0:
        print(f'  batch {b}/{nblk}', flush=True)

il = numpy.tril_indices(nbas)
# a shell pair (I,J), I > J, carries both (I,J) and (J,I)
e_low = e_pb[:, il[0], il[1]] + numpy.where(il[0] != il[1], e_pb[:, il[1], il[0]], 0)
vals = numpy.abs(e_low).ravel()
order = numpy.sort(vals)
cum = numpy.cumsum(order)
total = e_low.sum()
print(f'exchange energy (or its change) {total:.10f}; {vals.size} pair-batches, {(vals > 0).sum()} nonzero')
for tol in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    ndrop = numpy.searchsorted(cum, tol)
    keep = vals.size - ndrop
    print(f'  sum of dropped |e| <= {tol:.0e}: keep {keep} of {vals.size} pair-batches ({keep / vals.size:.1%})')
# the same per pair-batch threshold, as a fixed cutoff would screen
for thr in [1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
    keep = (vals > thr).sum()
    print(f'  |e| > {thr:.0e}: keep {keep} ({keep / vals.size:.1%}); dropped sum {vals[vals <= thr].sum():.2e}, '
          f'dropped signed sum {e_low.ravel()[vals <= thr].sum():+.2e}')

# SGX's position-dependent bound of the ESP integrals on each batch
sd = sgx_jk.SGXData(mol, grids, etol='auto', vtol='auto', hermi=1, bound_algo='sample_pos', direct_scf_tol=1e-13)
from pyscf.sgx.sgx import _make_opt
sd._opt = _make_opt(mol, False, 1e-13)
sd._build_integral_bounds()
asgx = sd._rbar_ij[None, :, :] * sd._mbar_bi[:, :, None] * sd._mbar_bi[:, None, :]
for name, abar in [('amax', amax), ('sgx', asgx)]:
    est = abar * fb1[:, :, None] * fb2[:, None, :]
    if args.incr is not None:
        est = est + est.transpose(0, 2, 1)
    est = .25 * est
    est_low = (est[:, il[0], il[1]] * numpy.where(il[0] != il[1], 2, 1)).ravel() if args.incr is None else \
        est[:, il[0], il[1]].ravel()
    ratio = est_low[vals > 1e-14] / vals[vals > 1e-14]
    print(f'estimate {name}: est/|e| median {numpy.median(ratio):.2e}, 5% {numpy.percentile(ratio, 5):.2e}, '
          f'min {ratio.min():.2e}')
    for thr in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
        keep = est_low > thr
        drop = ~keep
        print(f'  est > {thr:.0e}: keep {keep.sum()} ({keep.mean():.1%}); dropped |e| sum {vals[drop].sum():.2e}, '
              f'signed {e_low.ravel()[drop].sum():+.2e}')
