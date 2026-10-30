"""1 thread: (a) AO count ns per 224-point group of the fused kernels; (b) cost of LIBXC_eval_xc on each
of the 10 contiguous ranges that its static split hands to 10 threads, on real block_loop blocks."""
import sys, time
import numpy as np
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, BENZENE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint, libxc

def groups_ns(mol, grids):
    non0 = grids.non0tab  # [nblk, nbas]
    ao_loc = mol.ao_loc_nr()
    nao_sh = np.diff(ao_loc)
    nblk = non0.shape[0]
    ns = []
    for g0 in range(0, nblk, 4):
        m = (non0[g0:g0+4] != 0).any(axis=0)
        ns.append(nao_sh[m].sum())
    return np.array(ns)

which = sys.argv[1] if len(sys.argv) > 1 else 'all'
if which in ('all', 'ns'):
    for name, atom, basis in [('benzene/SVP', BENZENE, 'def2-svp'), ('caffeine/SVP', CAFFEINE, 'def2-svp'),
                              ('caffeine/TZVP', CAFFEINE, 'def2-tzvp'), ('C20H42/SVP', n_alkane(20), 'def2-svp')]:
        t0 = time.perf_counter()
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        grids = dft.gen_grid.Grids(mol)
        grids.build(with_non0tab=True)
        ns = groups_ns(mol, grids)
        q = np.percentile(ns, [10, 25, 50, 75, 90])
        # flop-weighted mean of ns (work ~ ns^2 per group)
        w = (ns.astype(float)**2)
        print(f'{name:14s} nao {mol.nao:4d} ngrids {grids.weights.size:7d} groups {ns.size:5d} '
              f'ns mean {ns.mean():6.1f} p10/25/50/75/90 {q.astype(int)} max {ns.max()} '
              f'ns^2-weighted mean {np.sqrt((w*ns**2).sum()/w.sum()) if False else (w*ns).sum()/w.sum():6.1f}  '
              f'({time.perf_counter()-t0:.1f}s)', flush=True)

if which in ('all', 'xc'):
    mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.build(with_non0tab=True)
    dm = dft.RKS(mol).get_init_guess(key='minao')
    ni = numint.NumInt()
    ngrids = grids.weights.size
    blk = 1200 * 56
    nthr = 10
    for xc, xctype in [('PBE', 'MGGA'), ('B3LYP', 'GGA'), ('r2SCAN', 'MGGA')]:
        pass
    rho_all = np.empty((5, ngrids))
    for i0, i1 in lib.prange(0, ngrids, blk):
        ao = numint.eval_ao(mol, grids.coords[i0:i1], deriv=1)
        rho_all[:, i0:i1] = numint.eval_rho(mol, ao, dm, xctype='MGGA', with_lapl=False)
        ao = None
    print('rho built; points with rho<1e-10: %.3f' % (rho_all[0] < 1e-10).mean(), flush=True)
    for xc, nvar in [('PBE', 4), ('B3LYP', 4), ('r2SCAN', 5)]:
        for deriv in (1, 2):
            tot_static = tot_mean = tot_serial = 0.
            per_block = []
            for i0, i1 in lib.prange(0, ngrids, blk):
                np_ = i1 - i0
                # LIBXC_eval_xc split: blksize = np/nthr, the first np%nthr get one more
                base, rem = divmod(np_, nthr)
                sizes = [base + (1 if t < rem else 0) for t in range(nthr)]
                offs = np.cumsum([0] + sizes)
                ts = []
                for t in range(nthr):
                    r = rho_all[:nvar, i0+offs[t]:i0+offs[t+1]].copy()
                    best = 1e9
                    for rep in range(3):
                        t0 = time.perf_counter()
                        libxc.eval_xc_eff(xc, r, deriv=deriv)
                        best = min(best, time.perf_counter() - t0)
                    ts.append(best)
                ts = np.array(ts)
                r = rho_all[:nvar, i0:i1].copy()
                best = 1e9
                for rep in range(3):
                    t0 = time.perf_counter()
                    libxc.eval_xc_eff(xc, r, deriv=deriv)
                    best = min(best, time.perf_counter() - t0)
                tot_serial += best
                tot_static += ts.max()
                tot_mean += ts.mean()
                per_block.append(ts.max() / ts.mean())
            print(f'{xc:7s} deriv {deriv}: 1-thread whole-grid {tot_serial*1e3:7.1f} ms '
                  f'({tot_serial/ngrids*1e9:5.0f} ns/pt); sum over blocks of max(chunk)={tot_static*1e3:6.1f} ms, '
                  f'mean(chunk)={tot_mean*1e3:6.1f} ms -> static/ideal = {tot_static/tot_mean:4.2f}; '
                  f'per block max/mean {np.round(per_block, 2)}', flush=True)
