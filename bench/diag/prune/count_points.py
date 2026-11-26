'''The points of the TS examples' XC and VV10 grids with and without small_rho_cutoff = 1e-7, from the converged
density at a geometry (an .xyz file; the second argument the spin). Usage: count_points.py GEOM.xyz SPIN'''
import sys
from pyscf import gto, dft
xyz, spin = sys.argv[1], int(sys.argv[2])
mol = gto.M(atom=xyz, basis='def2-tzvp', spin=spin, verbose=0)
mf = (dft.UKS if spin else dft.RKS)(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mf.kernel()
dm = mf.make_rdm1()
for cut in (0, 1e-7):
    m = mf.copy()
    m.grids = dft.gen_grid.Grids(mol); m.grids.atom_grid = (99, 590)
    m.nlcgrids = dft.gen_grid.Grids(mol); m.nlcgrids.atom_grid = (50, 194)
    m.small_rho_cutoff = cut
    m.initialize_grids(mol, dm)   # UKS: the two spins (its initialize_grids sums them)
    print('%s cutoff %g: XC grid %d, VV10 grid %d points' % (xyz, cut, m.grids.weights.size, m.nlcgrids.weights.size))
