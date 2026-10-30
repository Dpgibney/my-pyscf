"""Per-group AO counts (ns) of the fused XC kernels on the default grids, and the flop model of
the dm path vs the factored path for TD response dms (r = nq = nocc).
Usage: group_stats.py benzene caffeine c20"""
import sys
import time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft
from pyscf.dft import numint

SYS = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20': n_alkane(20)}
for name in sys.argv[1:]:
    mol = gto.M(atom=SYS[name], basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='B3LYP')
    t0 = time.perf_counter()
    mf.grids.build(with_non0tab=True)
    t1 = time.perf_counter()
    g = mf.grids
    data = numint._GridAOData(mol, g)
    ns = data.group_naos(mol)
    nao = mol.nao
    nocc = mol.nelectron // 2
    nvir = nao - nocc
    ngrids = g.weights.size
    s1, s2 = ns.sum(), (ns**2).sum()
    print(f'{name}: nao {nao} nocc {nocc} nvir {nvir} ngrids {ngrids} groups {ns.size} '
          f'grid build {t1-t0:.1f}s')
    print(f'  ns mean {ns.mean():.1f} min {ns.min():.0f} max {ns.max():.0f}  '
          f'sum ns^2/sum ns {s2/s1:.1f}  nonzero frac {np.count_nonzero(g.non0tab)/g.non0tab.size:.3f}')
    # dm path flops per set over the grid (GEMM part): c_dm*ns^2 per point
    for xctype, (c_dm, c_rho, c_v, c_pass) in {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                                'MGGA': (13, 8, 10, 8)}.items():
        ng = 224
        dm = c_dm * s2 * ng
        lr = (c_rho * nocc + c_v * nocc) * s1 * ng
        lr_pass = c_pass * (2 * nocc) * s1 * ng
        for nset in (5, 10, 16, 20):
            ratio = (nset * lr + lr_pass) / (nset * dm)
            print(f'  {xctype:4s} nset {nset:2d}: dm path {dm/1e9:6.2f} GF/set, factored {lr/1e9:6.2f} '
                  f'GF/set + {lr_pass/1e9:5.2f} GF/pass -> ratio {ratio:.3f}')
