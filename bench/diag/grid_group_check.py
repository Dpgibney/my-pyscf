"""It15 check: gen_grid.arg_group_grids (an integer box key and one stable argsort) against the
numpy.unique(axis=0) version it replaces: identical permutations, and the time of each.

Run in a configuration's environment, e.g.
    bench/diag/run_in.py G=pyscf 1 bench/diag/grid_group_check.py
"""
import os
import sys
import time

import numpy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cases  # noqa: E402
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
from pyscf import dft, gto  # noqa: E402
from pyscf.dft import gen_grid  # noqa: E402


def old_arg_group_grids(mol, coords, box_size=gen_grid.GROUP_BOX_SIZE):
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
    rev_idx, counts = numpy.unique(box_ids, axis=0, return_inverse=True,
                                   return_counts=True)[1:3]
    return rev_idx.ravel().argsort(kind='stable')


def timed(f, *args):
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = f(*args)
        t.append(time.perf_counter() - t0)
    return out, min(t)


def main():
    rng = numpy.random.default_rng(3)
    systems = [('water', WATER, 3), ('benzene', BENZENE, 3), ('caffeine', CAFFEINE, 3),
               ('caffeine level 5', CAFFEINE, 5), ('n-C20H42', cases.C20H42, 3)]
    for name, atom, level in systems:
        mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
        grids = dft.gen_grid.Grids(mol)
        grids.level = level
        coords = grids.get_partition(mol)[0]   # atomic grids before grouping
        tests = [('grid', coords),
                 ('displaced by N(0, 3 bohr)', coords + rng.normal(scale=3., size=coords.shape))]
        for label, xyz in tests:
            ref, t0 = timed(old_arg_group_grids, mol, xyz)
            new, t1 = timed(gen_grid.arg_group_grids, mol, xyz)
            print(f'{name:17s} {label:26s} {len(xyz):8d} points: '
                  f'{"identical" if numpy.array_equal(ref, new) else "DIFFERENT"}, '
                  f'{t0*1e3:7.1f} -> {t1*1e3:6.1f} ms', flush=True)


if __name__ == '__main__':
    main()
