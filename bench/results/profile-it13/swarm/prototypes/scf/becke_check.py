"""1 thread: libdft VXCgen_grid vs the same source built -msse3 / -mavx2 / -mavx512 (no FMA), and a fused
variant returning the weights; bitwise comparison and timings over all atoms' grids (as get_partition)."""
import sys, time, ctypes
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto, lib
from pyscf.dft import gen_grid, radi

D = '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/scf/'
libs = {v: ctypes.CDLL(D + f'becke_{v}.so') for v in ('sse3', 'avx2', 'avx512')}
libdft = lib.load_library('libdft')
c = lambda a: a.ctypes.data_as(ctypes.c_void_p)

for name, atom, basis in [('caffeine', CAFFEINE, 'def2-svp'), ('C20H42', n_alkane(20), 'def2-svp')]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    g = gen_grid.Grids(mol)
    tab = g.gen_atomic_grids(mol, g.atom_grid, g.radi_method, g.level, g.prune)
    f_radii_adjust = radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
    radii = numpy.asarray([f_radii_adjust(i, j, 0) for i in range(mol.natm) for j in range(mol.natm)])
    atm_coords = numpy.asarray(mol.atom_coords(), order='C')
    natm = mol.natm
    inputs = []
    for ia in range(natm):
        coords, vol = tab[mol.atom_symbol(ia)]
        coords = numpy.asarray(coords + atm_coords[ia], order='F')
        inputs.append((ia, coords, vol))

    def run_ref():
        out = []
        for ia, coords, vol in inputs:
            p = numpy.empty((natm, coords.shape[0]))
            libdft.VXCgen_grid(c(p), c(coords), c(atm_coords), c(radii), ctypes.c_int(natm), ctypes.c_int(coords.shape[0]))
            out.append(vol * p[ia] * (1./p.sum(axis=0)))
        return out

    def run_kernel(L, keep=False):
        out = []
        for ia, coords, vol in inputs:
            p = numpy.empty((natm, coords.shape[0]))
            L.BECKEgen_grid(c(p), c(coords), c(atm_coords), c(radii), ctypes.c_int(natm), ctypes.c_int(coords.shape[0]))
            out.append(vol * p[ia] * (1./p.sum(axis=0)))
        return out

    def run_fused(L):
        out = []
        for ia, coords, vol in inputs:
            w = numpy.empty(coords.shape[0])
            vol = numpy.ascontiguousarray(vol)
            L.BECKEweights(c(w), c(vol), ctypes.c_int(ia), c(coords), c(atm_coords), c(radii),
                           ctypes.c_int(natm), ctypes.c_int(coords.shape[0]))
            out.append(w)
        return out

    def tm(f, n=3):
        best = 1e9
        for _ in range(n):
            t0 = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t0)
        return best, r

    t_ref, ref = tm(run_ref)
    ref = numpy.hstack(ref)
    # also the full get_partition, for the share of the kernel in it
    t_gp, (_, w_gp) = tm(lambda: gen_grid.get_partition(mol, tab, radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII))
    print(f'{name}: natm {natm} ngrids {ref.size}; get_partition {t_gp:.3f} s; libdft kernel + numpy normalization {t_ref:.3f} s; '
          f'identical to get_partition: {numpy.array_equal(w_gp, ref)}')
    for v, L in libs.items():
        t, w = tm(lambda: run_kernel(L))
        w = numpy.hstack(w)
        tf, wf = tm(lambda: run_fused(L))
        wf = numpy.hstack(wf)
        print(f'   {v:7s} kernel+numpy {t:.3f} s (bitwise {numpy.array_equal(w, ref)}), fused weights {tf:.3f} s '
              f'(bitwise {numpy.array_equal(wf, ref)}, max rel diff {numpy.max(abs(wf-ref)/numpy.maximum(abs(ref),1e-300)):.1e})')
    sys.stdout.flush()
