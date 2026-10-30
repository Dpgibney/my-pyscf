import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft
from pyscf.dft import gen_grid
def fast(mol, coords, box_size=gen_grid.GROUP_BOX_SIZE):
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
    # lexicographic (x,y,z) order of boxes = order of this linear index; ids are in [-1, boxes]
    nb = boxes + 2
    key = ((box_ids[:,0] + 1) * nb[1] + (box_ids[:,1] + 1)) * nb[2] + (box_ids[:,2] + 1)
    return key.argsort(kind='stable')
for name, atom in [('caffeine', molecules.CAFFEINE), ('c20h42', molecules.n_alkane(20))]:
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    g = gen_grid.Grids(mol)
    coords, weights = g.get_partition(mol, g.gen_atomic_grids(mol))[:2] if hasattr(g, 'get_partition') else (None, None)
    t0 = time.perf_counter(); ref = gen_grid.arg_group_grids(mol, coords); t1 = time.perf_counter()
    new = fast(mol, coords); t2 = time.perf_counter()
    print(f'{name}: ngrids {len(coords)}; arg_group_grids {1e3*(t1-t0):.0f} ms, integer key {1e3*(t2-t1):.1f} ms; identical permutation: {numpy.array_equal(ref, new)}')
