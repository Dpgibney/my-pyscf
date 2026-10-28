"""Screen-index statistics for medium molecules: nonzero fraction (upstream's switch uses < 0.5) and the
flops of the gathered products (4-block groups) relative to the dense products."""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft

systems = [('caffeine/SVP', CAFFEINE, 'def2-svp'), ('caffeine/TZVP', CAFFEINE, 'def2-tzvp'),
           ('C8H18/SVP', n_alkane(8), 'def2-svp'), ('C12H26/SVP', n_alkane(12), 'def2-svp'),
           ('C20H42/SVP', n_alkane(20), 'def2-svp')]
for label, atom, basis in systems:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.build(with_non0tab=True)
    si = grids.non0tab
    nz = si != 0
    shl = np.diff(mol.ao_loc_nr())
    m = 4
    nblk = si.shape[0]
    ng = (nblk + m - 1) // m
    pad = np.zeros((ng * m, mol.nbas), dtype=bool)
    pad[:nblk] = nz
    nact = pad.reshape(ng, m, mol.nbas).any(axis=1).astype(float) @ shl
    ratio = (nact**2).sum() * 56 * m / (mol.nao**2 * grids.weights.size)
    print(f'{label:14s} natm={mol.natm:3d} nao={mol.nao:4d} nonzero={nz.mean():.2f} '
          f'gathered/dense flops={ratio:.2f} mean active AOs {nact.mean():.0f}')
