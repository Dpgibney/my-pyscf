"""Memory and time of keeping AO values on the grid between passes (SCF cycles, response
iterations). Time per pass over the grid of three ways to get the AO values:
  eval    compute them (stock eval_ao, GTOval_sph_deriv0/1)
  full    copy finished AO values back from a cache
  radial  rebuild them from a cache of each shell's contracted radial factors
          (sum_p c exp(-a r^2), plus the -2a-weighted sum for deriv 1) with PySCF's polynomial
          and cart->sph code (ao_radial.c)
each made two ways:
  block_loop  dense blocks of 67200 points in RAM, as NumInt.block_loop makes them; "zeros" is
              eval_ao with every shell screened, i.e. only writing the block
  fused       every 56-point block into a per-thread buffer that stays in cache (L2, or L3 for
              large nao), as a numint that consumed each block right away would
Both caches keep only the 56-point blocks x shells that the grid's screen index keeps, and the
rebuilt blocks are checked to equal eval_ao bit for bit.
Usage: ao_radial_cache.py [case ...]     cases: caffeine-svp caffeine-tzvp c20h42-svp c20h42-tzvp
Build ao_radial.so first (see ao_radial.c).
"""
import ctypes
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.gto.mole import ANG_OF, NCTR_OF

_lib = np.ctypeslib.load_library('ao_radial', os.path.dirname(os.path.abspath(__file__)))
RAD_FILL, RAD_EVAL, FULL_FILL, FULL_EVAL, STOCK_EVAL = range(5)
FULL_LIMIT = 12e9   # full caches above this many bytes are sized but not allocated or timed
REPS = 5
CASES = {'caffeine-svp': (CAFFEINE, 'def2-svp'), 'caffeine-tzvp': (CAFFEINE, 'def2-tzvp'),
         'c20h42-svp': (n_alkane(20), 'def2-svp'), 'c20h42-tzvp': (n_alkane(20), 'def2-tzvp')}


def run(name):
    atom, basis = CASES[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    grids = dft.gen_grid.Grids(mol).build(with_non0tab=True)
    ni = numint.NumInt()
    nao, nbas, ngrids = mol.nao, mol.nbas, grids.weights.size
    atm, bas, env = mol._atm, mol._bas, mol._env
    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int32)
    mask = grids.non0tab
    keep = mask != 0
    ls, nctr = bas[:, ANG_OF], bas[:, NCTR_OF]
    blocks = [(ip0, ip1, np.asarray(grids.coords[ip0:ip1], order='F'), mask[ip0 // BLKSIZE:])
              for ip0, ip1 in lib.prange(0, ngrids, 1200 * BLKSIZE)]   # block_loop's block size
    zero_blocks = [(ip0, ip1, c, np.zeros_like(m)) for ip0, ip1, c, m in blocks]  # writes zeros only
    coords_all = np.asarray(grids.coords, order='F')
    print(f'{name}: nao={nao} ngrids={ngrids} screen index keeps {keep.mean():.1%} of '
          f'(56-point block, shell) pairs, {len(blocks)} block_loop blocks', flush=True)

    rows = []
    for deriv in (0, 1):
        ncomp = 1 + 3 * deriv
        buf = numint._empty_aligned(ncomp * 1200 * BLKSIZE * nao)

        def layout(per_shell):
            sz = (keep * per_shell).ravel().astype(np.int64)
            return (np.cumsum(sz) - sz).reshape(mask.shape), int(sz.sum())

        def drv(mode, cache, offs, ip0, ip1, coords, m):
            _lib.AOcache_drv(
                ctypes.c_int(mode), ctypes.c_int(deriv), ctypes.c_int(ip1 - ip0),
                buf.ctypes.data_as(ctypes.c_void_p), cache.ctypes.data_as(ctypes.c_void_p),
                offs[ip0 // BLKSIZE:].ctypes.data_as(ctypes.c_void_p),
                coords.ctypes.data_as(ctypes.c_void_p), m.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
                env.ctypes.data_as(ctypes.c_void_p))

        def fused(mode, cache, offs):
            _lib.AOcache_fused(
                ctypes.c_int(mode), ctypes.c_int(deriv), ctypes.c_int(ngrids),
                cache.ctypes.data_as(ctypes.c_void_p), offs.ctypes.data_as(ctypes.c_void_p),
                coords_all.ctypes.data_as(ctypes.c_void_p), mask.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
                env.ctypes.data_as(ctypes.c_void_p))

        def eval_pass(blks):
            for ip0, ip1, coords, m in blks:
                ni.eval_ao(mol, coords, deriv=deriv, non0tab=m, cutoff=grids.cutoff, out=buf)

        rad_offs, rad_n = layout((deriv + 1) * nctr * BLKSIZE)
        full_offs, full_n = layout(ncomp * nctr * (2 * ls + 1) * BLKSIZE)
        rad = np.empty(rad_n)
        t0 = time.perf_counter()
        for ip0, ip1, coords, m in blocks:
            drv(RAD_FILL, rad, rad_offs, ip0, ip1, coords, m)
        t_fill = time.perf_counter() - t0
        full = np.empty(full_n) if full_n * 8 <= FULL_LIMIT else None

        # fill the full cache from eval_ao and check that the caches (and the C copy of
        # eval_ao's loop) give eval_ao's blocks
        diffs = {'radial': [], 'full': [], 'stock': []}
        for ip0, ip1, coords, m in blocks:
            n = ncomp * nao * (ip1 - ip0)
            ni.eval_ao(mol, coords, deriv=deriv, non0tab=m, cutoff=grids.cutoff, out=buf)
            ref = buf[:n].copy()
            for k, mode, cache, offs in (('full', FULL_EVAL, full, full_offs),
                                         ('radial', RAD_EVAL, rad, rad_offs),
                                         ('stock', STOCK_EVAL, rad, rad_offs)):
                if cache is None:
                    continue
                if mode == FULL_EVAL:
                    drv(FULL_FILL, full, full_offs, ip0, ip1, coords, m)
                buf[:n] = np.nan
                drv(mode, cache, offs, ip0, ip1, coords, m)
                diffs[k].append(np.abs(buf[:n] - ref).max())
            del ref
        maxdiff = {k: float(np.max(v)) if v else np.nan for k, v in diffs.items()}   # NaN = unwritten

        # block_loop: dense blocks of 67200 points in RAM; fused: 56-point blocks in cache
        passes = {('block_loop', 'zeros'): lambda: eval_pass(zero_blocks),
                  ('block_loop', 'eval'): lambda: eval_pass(blocks),
                  ('block_loop', 'radial'): lambda: [drv(RAD_EVAL, rad, rad_offs, *b) for b in blocks],
                  ('fused', 'eval'): lambda: fused(STOCK_EVAL, rad, rad_offs),
                  ('fused', 'radial'): lambda: fused(RAD_EVAL, rad, rad_offs)}
        if full is not None:
            passes['block_loop', 'full'] = lambda: [drv(FULL_EVAL, full, full_offs, *b) for b in blocks]
            passes['fused', 'full'] = lambda: fused(FULL_EVAL, full, full_offs)
        times = {k: [] for k in passes}
        for k, f in passes.items():   # warm-up
            f()
        for _ in range(REPS):          # interleaved against clock drift
            for k, f in passes.items():
                t0 = time.perf_counter()
                f()
                times[k].append(time.perf_counter() - t0)
        t = {k: float(np.median(v)) for k, v in times.items()}
        row = dict(case=name, deriv=deriv, nao=nao, ngrids=ngrids,
                   dense_gb=ncomp * nao * ngrids * 8 / 1e9, full_gb=full_n * 8 / 1e9,
                   radial_gb=rad_n * 8 / 1e9, t_fill=t_fill, maxdiff=maxdiff,
                   times={'/'.join(k): v for k, v in t.items()})
        rows.append(row)
        print(f'  deriv={deriv}: AO values dense {row["dense_gb"]:.2f} GB, screened (full cache) '
              f'{row["full_gb"]:.2f} GB, radial cache {row["radial_gb"]:.2f} GB '
              f'({row["full_gb"] / row["radial_gb"]:.1f}x less); radial fill {t_fill:.3f} s; '
              f'max|diff| vs eval_ao: ' + ', '.join(f'{k} {v:.0e}' for k, v in maxdiff.items()))
        for path in ('block_loop', 'fused'):
            print(f'    {path:10s} s/pass: ' + ', '.join(
                f'{k[1]} {v:.3f}' for k, v in t.items() if k[0] == path), flush=True)
        del rad, full, buf
    return rows


if __name__ == '__main__':
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[1:] or CASES:
        run(name)
