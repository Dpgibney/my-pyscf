"""1-thread breakdown of Grids.build(with_non0tab=True), and an exact replacement of
arg_group_grids (scalar box key + one stable argsort) checked for an identical permutation."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto, lib
from pyscf.dft import gen_grid, radi


def arg_group_grids_key(mol, coords, box_size=gen_grid.GROUP_BOX_SIZE):
    atom_coords = mol.atom_coords()
    boundary = [atom_coords.min(axis=0) - gen_grid.GROUP_BOUNDARY_PENALTY,
                atom_coords.max(axis=0) + gen_grid.GROUP_BOUNDARY_PENALTY]
    boxes = ((boundary[1] - boundary[0]) * (1./box_size)).round().astype(int)
    box_size = (boundary[1] - boundary[0]) / boxes
    frac_coords = (coords - boundary[0]) * (1./box_size)
    box_ids = numpy.floor(frac_coords).astype(int)
    box_ids[box_ids<-1] = -1
    box_ids[box_ids[:,0] > boxes[0], 0] = boxes[0]
    box_ids[box_ids[:,1] > boxes[1], 1] = boxes[1]
    box_ids[box_ids[:,2] > boxes[2], 2] = boxes[2]
    # (bx, by, bz) in [-1, boxes]: lexicographic order == order of this key
    n1, n2 = boxes[1] + 2, boxes[2] + 2
    key = ((box_ids[:,0] + 1) * n1 + (box_ids[:,1] + 1)) * n2 + (box_ids[:,2] + 1)
    return key.argsort(kind='stable')


def tm(f, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
    return best, r


for name, atom, basis in [('caffeine/SVP', CAFFEINE, 'def2-svp'), ('C20H42/SVP', n_alkane(20), 'def2-svp')]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    g = gen_grid.Grids(mol)
    t_total, _ = tm(lambda: gen_grid.Grids(mol).build(with_non0tab=True), 2)
    t_atomic, tab = tm(lambda: g.gen_atomic_grids(mol, g.atom_grid, g.radi_method, g.level, g.prune))
    t_part, (coords, weights) = tm(lambda: g.get_partition(mol, tab, g.radii_adjust, g.atomic_radii, g.becke_scheme), 2)
    t_group, idx = tm(lambda: gen_grid.arg_group_grids(mol, coords))
    t_group2, idx2 = tm(lambda: arg_group_grids_key(mol, coords))
    same = numpy.array_equal(idx, idx2)
    g.build(with_non0tab=False)
    t_mask, _ = tm(lambda: g.make_mask(mol, g.coords))
    print(f'{name}: nao {mol.nao} natm {mol.natm} ngrids {weights.size}')
    print(f'  build(with_non0tab) total {t_total:.3f} s: gen_atomic_grids {t_atomic:.3f}, get_partition {t_part:.3f}, '
          f'arg_group_grids {t_group:.3f} (key version {t_group2:.3f}, identical={same}), make_mask {t_mask:.3f}')
    sys.stdout.flush()
