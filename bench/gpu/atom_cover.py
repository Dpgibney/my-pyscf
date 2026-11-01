"""Atoms and screened AOs per DFT grid block (ni.block_loop, the masks the Hessian's XC terms get), for caffeine
and n-C20H42 in def2-SVP on node1: how much of a per-atom loop over all atoms meets atoms without AOs in the block.
Source scripts/node1-gpu-env.sh perf first.
Usage: atom_cover.py"""
import sys, numpy, cupy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto
from gpu4pyscf import dft
for name, atom, basis in (('caffeine', CAFFEINE, 'def2-svp'), ('C20H42', n_alkane(20), 'def2-svp')):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.grids.build()
    ni = mf._numint
    ni.build(mol, mf.grids.coords)
    opt = ni.gdftopt
    smol = opt._sorted_mol
    ao_atom = numpy.repeat(smol._bas[:, 0], smol.ao_loc_nr()[1:] - smol.ao_loc_nr()[:-1])
    natm, nao = mol.natm, smol.nao
    present, nsub, nblk = [], [], 0
    for ao, mask, weight, _ in ni.block_loop(smol, mf.grids, nao, 1, None):
        idx = cupy.asnumpy(mask)
        present.append(len(numpy.unique(ao_atom[idx])))
        nsub.append(len(idx))
        nblk += 1
    present, nsub = numpy.array(present), numpy.array(nsub)
    print(f'{name} {basis}: natm {natm}, nao {nao}, {nblk} blocks; atoms with AOs in a block: mean {present.mean():.1f} '
          f'({100*present.mean()/natm:.0f}%), min {present.min()}, max {present.max()}; nao_sub mean {nsub.mean():.0f} '
          f'({100*nsub.mean()/nao:.0f}%)')
