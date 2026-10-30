"""It15 check: xc_deriv's transforms (transform_xc, ud2ts, ts2ud, the fg to direct-derivative
kernel) against the code before It15, bit for bit, and their times.

The reference is the old Python code of _unfold_gga/transform_xc driving the old libdft, loaded
from the pyscf-it13 tree. Run in a configuration's environment, e.g.
    bench/diag/run_in.py G=pyscf 10 bench/diag/xc_deriv_check.py
"""
import ctypes
import itertools
import time

import numpy as np
from pyscf import lib
from pyscf.dft import xc_deriv

OLD_LIB = '/home/danny/Chemistry/my-pyscf/pyscf-it13/pyscf/lib/libdft.so'
libdft_old = ctypes.CDLL(OLD_LIB)


def old_unfold_gga(rho, xc_val, spin, order, nvar, xlen, reserve=0):
    assert nvar >= 4
    ngrids = rho.shape[-1]
    n_transform = order - reserve
    if spin == 0:
        nvar2 = nvar
        drv = libdft_old.VXCunfold_sigma_spin0
    else:
        nvar2 = nvar * 2
        drv = libdft_old.VXCunfold_sigma_spin1
    idx = xc_deriv._product_uniq_indices(xlen, order)
    xc_tensor = np.empty((xlen**reserve * nvar2**n_transform, ngrids))
    xc_tensor[:idx.size] = xc_val[idx.ravel()]
    buf = np.empty_like(xc_tensor)
    for i in range(n_transform):
        xc_tensor, buf = buf, xc_tensor
        ncounts = xlen**(order-1-i) * nvar2**i
        drv(xc_tensor.ctypes, buf.ctypes, rho.ctypes,
            ctypes.c_int(ncounts), ctypes.c_int(nvar), ctypes.c_int(ngrids))
    if spin == 0:
        xc_tensor = xc_tensor.reshape([xlen]*reserve + [nvar]*n_transform + [ngrids])
    else:
        xc_tensor = xc_tensor.reshape([xlen]*reserve + [2,nvar]*n_transform + [ngrids])
    return xc_tensor


def old_transform_xc(rho, xc_val, xctype, spin, order):
    xc_val = np.asarray(xc_val, order='C')
    nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
    offsets = [0] + [xc_deriv.count_combinations(xlen, o) for o in range(order+1)]
    p0, p1 = offsets[order:order+2]
    rho = np.asarray(rho, order='C')
    xc_tensor = old_unfold_gga(rho, xc_val[p0:p1], spin, order, nvar, xlen)
    nabla_idx = np.arange(1, 4)
    if spin == 0:
        for n_pairs in range(1, order//2+1):
            p0, p1 = offsets[order-n_pairs:order-n_pairs+2]
            xc_sub = old_unfold_gga(rho, xc_val[p0:p1], spin, order-n_pairs, nvar, xlen, n_pairs)
            xc_sub = xc_sub[(1,)*n_pairs] * 2**n_pairs
            low_sigmas = itertools.combinations(range(order), n_pairs*2)
            pair_combs = [list(itertools.chain(*p[::-1]))
                          for p in xc_deriv._pair_combinations(list(range(n_pairs*2)))]
            diag_idx = xc_deriv._diagonal_indices(nabla_idx, n_pairs)
            for dim_lst in low_sigmas:
                rest_dims = [i for i in range(xc_tensor.ndim) if i not in dim_lst]
                for pair_comb in pair_combs:
                    xc_tensor_1 = xc_tensor.transpose([dim_lst[i] for i in pair_comb] + rest_dims)
                    xc_tensor_1[diag_idx] += xc_sub
    else:
        i3to2x2 = xc_deriv._product_uniq_indices(2, 2)
        for n_pairs in range(1, order//2+1):
            p0, p1 = offsets[order-n_pairs:order-n_pairs+2]
            xc_sub = old_unfold_gga(rho, xc_val[p0:p1], spin, order-n_pairs, nvar, xlen, n_pairs)
            xc_sub = xc_sub[(slice(2,5),) * n_pairs]
            for i in range(n_pairs):
                xc_sub[(slice(None),)*i+(0,)] *= 2
                xc_sub[(slice(None),)*i+(2,)] *= 2
            sigma_idx = (i3to2x2[(slice(None),)*2 + (np.newaxis,)*(i*2)]
                         for i in reversed(range(n_pairs)))
            xc_sub = xc_sub[tuple(sigma_idx)]
            low_sigmas = itertools.combinations(range(order), n_pairs*2)
            pair_combs = [list(itertools.chain(*p[::-1]))
                          for p in xc_deriv._pair_combinations(list(range(n_pairs*2)))]
            diag_idx = xc_deriv._diagonal_indices(nabla_idx, n_pairs)
            for dim_lst in low_sigmas:
                dim_lst2 = np.array(dim_lst)*2 + np.array([1, 0])[:,None]
                rest_dims = [i for i in range(xc_tensor.ndim) if i not in dim_lst2]
                for pair_comb in pair_combs:
                    leading_dims = list(dim_lst2[:,pair_comb].ravel())
                    xc_tensor_1 = xc_tensor.transpose(leading_dims + rest_dims)
                    xc_tensor_1[diag_idx] += xc_sub
    return xc_tensor


# Every update recorded by transform_xc must hit distinct rows of xc_tensor (numpy's fancy-index
# += adds once per distinct index; _add_rows adds every pair)
_add_rows = xc_deriv._add_rows
def checked_add_rows(xc_tensor, xc_sub, tgt, src):
    for t in tgt:
        assert np.unique(t).size == t.size, 'repeated target rows in one update'
    return _add_rows(xc_tensor, xc_sub, tgt, src)


def inputs(rng, xctype, spin, order, ngrids):
    nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
    rho = rng.standard_normal((spin+1, nvar, ngrids))
    rho[:,0] = abs(rho[:,0]) + .1
    if spin == 0:
        rho = rho[0]
    xc_val = rng.standard_normal((xc_deriv.count_combinations(xlen, order), ngrids))
    return rho, xc_val


def timed(f, *args, reps=3):
    f(*args)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*args)
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    rng = np.random.default_rng(7)
    print(f'{lib.num_threads()} threads')
    xc_deriv._add_rows = checked_add_rows
    nbad = 0
    for xctype, spin, order in itertools.product(('GGA', 'MGGA'), (0, 1), (1, 2, 3, 4)):
        for ngrids in (1, 3001):
            rho, xc_val = inputs(rng, xctype, spin, order, ngrids)
            ref = old_transform_xc(rho, xc_val, xctype, spin, order)
            new = xc_deriv.transform_xc(rho, xc_val, xctype, spin, order)
            same = ref.shape == new.shape and np.array_equal(ref, new)
            nbad += not same
            if not same:
                print(f'  DIFFERENT {xctype} spin={spin} order={order} ngrids={ngrids}: '
                      f'max|d| {abs(ref - new).max():.1e}')
    xc_deriv._add_rows = _add_rows
    print(f'transform_xc, GGA/MGGA x spin 0/1 x order 1-4 x 1 and 3001 points: '
          f'{"all identical" if nbad == 0 else f"{nbad} differ"}')

    # ud2ts, ts2ud and the fg kernel through their Python wrappers, old and new library
    # ud2ts/ts2ud take [2,nvar]*order + [ngrids] arrays
    vs = [rng.standard_normal((2, nvar) * order + (3001,)) for nvar, order in ((5, 1), (5, 2), (4, 3))]
    fg = rng.standard_normal((3, 3, 4, 3001))
    rho2 = rng.standard_normal((2, 4, 3001))
    outs = {}
    for label, libh in (('old', libdft_old), ('new', xc_deriv.libdft)):
        saved = xc_deriv.libdft
        xc_deriv.libdft = libh
        try:
            # (for order >= 2 both overwrite their input with intermediates, so pass copies)
            outs[label] = (np.hstack([xc_deriv.ud2ts(v.copy()).ravel() for v in vs]),
                           np.hstack([xc_deriv.ts2ud(v.copy()).ravel() for v in vs]))
            qg = np.empty((3, 2, 3, 4, 3001))
            libh.VXCfg_to_direct_deriv(qg.ctypes, fg.ctypes, rho2.ctypes, ctypes.c_int(3),
                                       ctypes.c_int(4), ctypes.c_int(4), ctypes.c_int(3001))
            outs[label] += (qg,)
        finally:
            xc_deriv.libdft = saved
    for name, a, b in zip(('ud2ts', 'ts2ud', 'fg_to_direct'), outs['old'], outs['new']):
        print(f'{name}: {"identical" if np.array_equal(a, b) else "DIFFERENT"}')

    print('times at 1M points, old -> new transform_xc (s); eval_xc_eff passes deriv = order')
    for xctype, spin, order in itertools.product(('GGA', 'MGGA'), (0, 1), (1, 2, 3)):
        rho, xc_val = inputs(rng, xctype, spin, order, 1_000_000)
        t0 = timed(old_transform_xc, rho, xc_val, xctype, spin, order)
        t1 = timed(xc_deriv.transform_xc, rho, xc_val, xctype, spin, order)
        print(f'  {xctype:4s} spin={spin} order={order}: {t0:6.3f} -> {t1:6.3f}  ({t0/t1:4.1f}x)', flush=True)


if __name__ == '__main__':
    main()
