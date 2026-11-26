'''GPU (examples/wb97mv_ts_gpu.py's grids): the points of the XC and VV10 grids after an SCF, with and without
small_rho_cutoff = 1e-7, plain and two-level (padding and zero weights apart).
Usage: count_points_gpu.py GEOM.xyz SPIN'''
import sys
import cupy
from pyscf import gto
from gpu4pyscf import dft
xyz, spin = sys.argv[1], int(sys.argv[2])
mol = gto.M(atom=xyz, basis='def2-tzvp', spin=spin, verbose=0)
for two_level in (False, True):
    for cut in (0, 1e-7):
        mf = (dft.UKS if spin else dft.RKS)(mol, xc='wb97m-v')
        mf.grids.atom_grid = (99, 590)
        mf.nlcgrids.atom_grid = (50, 194)
        mf.small_rho_cutoff = cut
        mf.two_level = two_level
        mf.kernel()
        g, n = mf.grids, mf.nlcgrids
        print('%s two_level %-5s cutoff %-6g E %.10f  XC grid %d (%d nonzero)  VV10 grid %d (%d nonzero)' % (
            xyz.split('/')[-2], two_level, cut, mf.e_tot, g.weights.size, int((g.weights != 0).sum()),
            n.weights.size, int((n.weights != 0).sum())), flush=True)
