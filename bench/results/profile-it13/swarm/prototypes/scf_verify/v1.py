import sys, time, numpy, scipy.linalg
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto, lib, scf
from pyscf.dft import gen_grid

def key_group(mol, coords, box_size=gen_grid.GROUP_BOX_SIZE):
    atom_coords = mol.atom_coords()
    b0 = atom_coords.min(axis=0) - gen_grid.GROUP_BOUNDARY_PENALTY
    b1 = atom_coords.max(axis=0) + gen_grid.GROUP_BOUNDARY_PENALTY
    boxes = ((b1 - b0) * (1./box_size)).round().astype(int)
    bs = (b1 - b0) / boxes
    box_ids = numpy.floor((coords - b0) * (1./bs)).astype(int)
    box_ids[box_ids<-1] = -1
    for k in range(3):
        box_ids[box_ids[:,k] > boxes[k], k] = boxes[k]
    key = ((box_ids[:,0]+1)*(boxes[1]+2) + (box_ids[:,1]+1))*(boxes[2]+2) + (box_ids[:,2]+1)
    return key.argsort(kind='stable')

def best(f, n=3):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t = min(t, time.perf_counter()-t0)
    return t, r

for name, atom, basis, lvl in [('caffeine/SVP', CAFFEINE, 'def2-svp', None),
                          ('caffeine lvl5', CAFFEINE, 'def2-svp', 5),
                          ('C20H42/SVP', n_alkane(20), 'def2-svp', None)]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    g = gen_grid.Grids(mol)
    if lvl: g.level = lvl
    tab = g.gen_atomic_grids(mol, g.atom_grid, g.radi_method, g.level, g.prune)
    coords, w = g.get_partition(mol, tab, g.radii_adjust, g.atomic_radii, g.becke_scheme)
    t1, i1 = best(lambda: gen_grid.arg_group_grids(mol, coords))
    t2, i2 = best(lambda: key_group(mol, coords))
    # random perturbed coords test
    rng = numpy.random.default_rng(1)
    c3 = coords + rng.normal(scale=3.0, size=coords.shape)
    same3 = numpy.array_equal(gen_grid.arg_group_grids(mol, c3), key_group(mol, c3))
    print(f'{name}: ngrids {w.size} old {t1:.3f}s new {t2:.3f}s same={numpy.array_equal(i1,i2)} rand_same={same3}', flush=True)
    t, cond = best(lambda: mol.get_overlap_cond(), 5)
    dm = numpy.random.rand(mol.nao, mol.nao)
    tc, _ = best(lambda: mol.condense_to_shell(dm, 'absmax'), 5)
    print(f'  get_overlap_cond {t*1e3:.3f} ms, condense {tc*1e3:.3f} ms (nbas {mol.nbas})', flush=True)
    if name.startswith('caffeine lvl'): continue
    mf = scf.RHF(mol)
    h = mf.get_hcore(); s = mf.get_ovlp()
    x = scf.addons.canonical_orth_(s)
    f = x.T @ h @ x
    te, _ = best(lambda: scipy.linalg.eigh(f.copy(), overwrite_a=True), 5)
    td, _ = best(lambda: scipy.linalg.eigh(f.copy(), overwrite_a=True, driver='evd'), 5)
    print(f'  eigh n={f.shape[0]} evr {te*1e3:.1f} ms evd {td*1e3:.1f} ms', flush=True)
