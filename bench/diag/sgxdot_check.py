"""The two dense-AO steps of PySCF's pjs SGX K build on their own, as get_k_only calls them, over the whole SGX
grid: fg = D chi (_sgxdot_ao_dm_sparse) and K += chi^T gv (_sgxdot_ao_gv_sparse, with gv = fg), plus eval_gto.
The density is the minao guess; the screening masks are the K build's own. Times each step (best of --reps), and
with --save writes the results, or with --compare checks them against a saved file (bitwise fraction, max |diff|).
Run with PYTHONPATH set to the tree.
Usage: sgxdot_check.py [--mol caffeine|c20h42] [--basis def2-svp] [--reps 3] [--save F.npz | --compare F.npz]"""
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib, sgx  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'c20h42'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--save')
ap.add_argument('--compare')
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'c20h42': n_alkane(20)}[args.mol]
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = sgx.sgx_fit(dft.RKS(mol, xc='B3LYP'), pjs=True)
dm = mf.get_init_guess()
sgxobj = mf.with_df
sgxobj.build(level=sgxobj.grids_level_f)
sgxobj._build_pjs(1e-13)
dms = dm.reshape(-1, mol.nao, mol.nao)
nbins, screen_index, pair_mask, ao_loc, blksize = sgxobj._pjs_data.get_loop_data(nset=1, with_pair_mask=True)
proj = sgxobj._pjs_data._overlap_correction_matrix
proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
dm_mask = sgxobj._pjs_data.get_dm_threshold_matrix(dms) if sgxobj._pjs_data.use_dm_screening else None
grids = sgxobj.grids
ngrids = grids.weights.size
blocks = list(lib.prange(0, ngrids, blksize))
BLKSIZE = 56
aos = []
t_ao = 1e9
for rep in range(args.reps):
    t0 = time.perf_counter()
    aos = [mol.eval_gto('GTOval', grids.coords[i0:i1], non0tab=screen_index[i0 // BLKSIZE:]) for i0, i1 in blocks]
    t_ao = min(t_ao, time.perf_counter() - t0)
t_dm = t_gv = 1e9
for rep in range(args.reps):
    fgs = []
    t0 = time.perf_counter()
    for (i0, i1), ao in zip(blocks, aos):
        fgs.append(sgx_jk._sgxdot_ao_dm_sparse(ao, proj_dm, screen_index[i0 // BLKSIZE:], dm_mask, ao_loc).copy())
    t_dm = min(t_dm, time.perf_counter() - t0)
    vk = numpy.zeros_like(dms)
    t0 = time.perf_counter()
    for (i0, i1), ao, fg in zip(blocks, aos, fgs):
        mask = screen_index[i0 // BLKSIZE:]
        sgx_jk._sgxdot_ao_gv_sparse(ao, fg, grids.weights[i0:i1], mask, mask, ao_loc, out=vk)
    t_gv = min(t_gv, time.perf_counter() - t0)
fg_all = numpy.concatenate([f[0] for f in fgs], axis=1)
print(f'{args.mol} {args.basis}: nao {mol.nao}, {ngrids} points in {len(blocks)} blocks, {lib.num_threads()} threads; '
      f'eval_gto {t_ao:.3f} s, _sgxdot_ao_dm_sparse {t_dm:.3f} s, _sgxdot_ao_gv_sparse {t_gv:.3f} s (best of {args.reps})')
if args.save:
    numpy.savez(args.save, fg=fg_all, vk=vk)
if args.compare:
    ref = numpy.load(args.compare)
    for name, new in (('fg', fg_all), ('vk', vk)):
        old = ref[name]
        print(f'  {name}: {100 * numpy.mean(old == new):.4f}% bitwise equal, max |diff| {abs(old - new).max():.2e} '
              f'(max |value| {abs(old).max():.2e})')
