'''wB97M-V/def2-TZVPD SCF and gradient by part, exact J/K and RIJCOSX, one process.
Usage: prof_scf_grad.py MOL [--basis def2-tzvpd] [--grid 99,590] [--nlc 50,194] [--skip-rijcosx] [--save PREFIX]
Each timed function is wrapped; times are wall seconds.'''
import argparse
import sys
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rhf as rhf_grad, rks as rks_grad
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlc', default='50,194')
ap.add_argument('--skip-rijcosx', action='store_true')
ap.add_argument('--skip-grad', action='store_true')
ap.add_argument('--save', default=None)
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)

GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)

T = {}
LOG = []


def wrap(owner, name, key, split=None):
    fn = getattr(owner, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            kk = key if split is None else key + split(a, k)
            T[kk] = T.get(kk, 0.) + dt
            LOG.append((kk, dt))
    setattr(owner, name, w)
    return fn


def setgrid(g, spec):
    if ',' in spec:
        g.atom_grid = tuple(int(x) for x in spec.split(','))
    else:
        g.level = int(spec)


def make(rijcosx=False):
    mf = dft.RKS(mol, xc=args.xc)
    setgrid(mf.grids, args.grid)
    setgrid(mf.nlcgrids, args.nlc)
    if rijcosx:
        mf = mf.RIJCOSX()
    return mf


def report(title, total, keys=None):
    print(f'--- {title}: {total:.1f} s')
    acc = 0.
    for k in sorted(T):
        print(f'    {k:34s} {T[k]:8.1f} s  {100 * T[k] / total:5.1f}%')
    sys.stdout.flush()


# ---------------- exact SCF
mf = make()
print(f'{args.mol}/{args.basis} {args.xc}: natm {mol.natm} nao {mol.nao} nocc {mol.nelectron // 2}, '
      f'max_memory {mf.max_memory} MB', flush=True)
jksplit = lambda a, k: ' LR-K' if k.get('omega') else (' J+K' if k.get('with_k', True) and k.get('with_j', True) else ' other')
wrap(mf, 'get_jk', 'jk', jksplit)
ni = mf._numint
wrap(ni, 'nr_rks', 'xc nr_rks')
wrap(ni, 'nr_nlc_vxc', 'vv10 nr_nlc_vxc')
cyc = []
t_last = [time.perf_counter()]


def cb(envs):
    now = time.perf_counter()
    cyc.append((envs['cycle'], now - t_last[0], envs['e_tot']))
    t_last[0] = now
mf.callback = cb
t0 = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t0
print(f'exact SCF: E = {mf.e_tot:.10f} conv {mf.converged} cycles {mf.cycles}; XC grid {mf.grids.weights.size} '
      f'pts, VV10 grid {mf.nlcgrids.weights.size} pts')
print('  per cycle (s):', ' '.join(f'{c[1]:.1f}' for c in cyc))
jk_calls = [(k, round(dt, 1)) for k, dt in LOG if k.startswith('jk')]
print('  J/K calls:', jk_calls)
print('  XC calls:', [round(dt, 1) for k, dt in LOG if k.startswith('xc')])
print('  VV10 calls:', [round(dt, 1) for k, dt in LOG if k.startswith('vv10')])
report('exact SCF', t_scf)
e_exact = mf.e_tot
dm_exact = mf.make_rdm1()
if args.save:
    numpy.savez(args.save + '_exact_scf.npz', mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
                e_tot=mf.e_tot)

# ---------------- exact gradient
if not args.skip_grad:
    T.clear(); LOG.clear()
    g = mf.Gradients()
    wrap(rhf_grad, 'jk_energy_per_atom', 'jk grad', lambda a, k: ' LR-K' if k.get('omega') else ' J+K')
    wrap(rks_grad, 'get_vxc', 'xc grad')
    wrap(rks_grad, 'get_nlc_vxc', 'vv10 grad')
    t0 = time.perf_counter()
    de_exact = g.kernel()
    t_grad = time.perf_counter() - t0
    report('exact gradient', t_grad)
    if args.save:
        numpy.save(args.save + '_exact_grad.npy', de_exact)

# ---------------- RIJCOSX SCF
if not args.skip_rijcosx:
    from pyscf.df import df_jk
    from pyscf.sgx import sgx_jk
    T.clear(); LOG.clear()
    mfx = make(rijcosx=True)
    wrap(df_jk, 'get_j', 'rij J')
    wrap(sgx_jk, 'get_k_only', 'sgx K')
    nix = mfx._numint
    wrap(nix, 'nr_rks', 'xc nr_rks')
    wrap(nix, 'nr_nlc_vxc', 'vv10 nr_nlc_vxc')
    cyc = []
    t_last[0] = time.perf_counter()
    mfx.callback = cb
    t0 = time.perf_counter()
    mfx.kernel()
    t_x = time.perf_counter() - t0
    print(f'RIJCOSX SCF: E = {mfx.e_tot:.10f} conv {mfx.converged} cycles {mfx.cycles}; error vs exact '
          f'{mfx.e_tot - e_exact:+.3e} Eh')
    print('  per cycle (s):', ' '.join(f'{c[1]:.1f}' for c in cyc))
    print('  K calls:', [round(dt, 1) for k, dt in LOG if k.startswith('sgx')])
    print('  J calls:', [round(dt, 2) for k, dt in LOG if k.startswith('rij')])
    report('RIJCOSX SCF (incl. final level-2 step)', t_x)
    if not args.skip_grad:
        T.clear(); LOG.clear()
        gx = mfx.Gradients()
        from pyscf.sgx.grad import rhf as sgx_grad
        from pyscf.df.grad import rhf as df_grad
        wrap(sgx_grad, '_get_jk', 'sgx K grad')
        wrap(df_grad, 'get_j', 'rij J grad')
        t0 = time.perf_counter()
        de_x = gx.kernel()
        t_gx = time.perf_counter() - t0
        report('RIJCOSX gradient', t_gx)
        d = de_x - de_exact
        print(f'  gradient error vs exact: max {abs(d).max():.3e}, rms {numpy.sqrt((d ** 2).mean()):.3e} Eh/bohr '
              f'(max |g| {abs(de_exact).max():.3e})')
print('done', flush=True)
