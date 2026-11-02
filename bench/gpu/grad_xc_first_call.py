"""Where the gradient XC term's first call spends its extra time (node1): runs the SCF, then grad.rks.get_exc twice
under cProfile (the largest cumulative entries of each call), and counts the AOs of the grid blocks that the SCF's
loop (AO derivative order 1) and the gradient's (order 2) work on, by 64-AO tile.
Usage: grad_xc_first_call.py [--xc pbe] [--basis def2-svp] [--mol caffeine|benzene] [--top 22]"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='pbe')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--top', type=int, default=22)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
dm = mf.make_rdm1()   # with mo_coeff and mo_occ attached, as the gradient passes it
for call in (1, 2):
    pr = cProfile.Profile()
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    pr.enable()
    rks_grad.get_exc(mf._numint, mol, mf.grids, mf.xc, dm)
    cupy.cuda.Device().synchronize()
    pr.disable()
    print(f'== {args.mol} {args.basis} {args.xc}: get_exc call {call}, {time.perf_counter() - t0:.2f} s')
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(args.top)
    print('\n'.join(x for x in s.getvalue().split('\n') if x.strip()))
ni = mf._numint
for deriv in (1, 2):
    nao_sub = []
    for ao_mask, idx, weight, _ in ni.block_loop(ni.gdftopt._sorted_mol, mf.grids, mol.nao, deriv, None):
        nao_sub.append(len(idx))
    nao_sub.sort()
    tiles = {}
    for n in nao_sub:
        k = -(-n // 64) * 64
        tiles[k] = tiles.get(k, 0) + 1
    print(f'AO derivative order {deriv}: {len(nao_sub)} blocks, {mf.grids.coords.shape[0]} points; AOs per block '
          f'{nao_sub[0]}-{nao_sub[-1]} (median {nao_sub[len(nao_sub) // 2]}) of {mol.nao}; blocks by 64-AO tile: ' +
          ', '.join(f'{k - 63}-{k}: {c}' for k, c in sorted(tiles.items())))
