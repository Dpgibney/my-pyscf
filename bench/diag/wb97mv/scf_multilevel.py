'''Exact wB97M-V SCF started from a cheap pre-SCF, against the same SCF from the minao guess.
Stage 1: RIJCOSX (no final level-2 step) with XC/VV10 grids of its own, converged to --pre-tol.
Stage 2: the exact SCF (production grids) from stage 1's density.
Usage: scf_multilevel.py MOL [--mode two-stage|pre_scf|minao] [--pre-grid 2] [--pre-nlc 1] [--pre-tol 1e-6]'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlc', default='50,194')
ap.add_argument('--mode', default='two-stage')
ap.add_argument('--pre-grid', default='2')
ap.add_argument('--pre-nlc', default='1')
ap.add_argument('--pre-tol', type=float, default=1e-6)
ap.add_argument('--eref', type=float, default=None)
ap.add_argument('--aux', default='def2-universal-jfit', help="stage-1 J-fitting basis; 'aug_etb', 'aug_etb:BETA' or 'autoaux' generate one")
ap.add_argument('--direct-j', action='store_true', help='stage 1 with direct (exact) J instead of RI-J')
ap.add_argument('--final-level', type=int, default=2, help='SGX grid level of the one-shot energy of mode nsc')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)


def setgrid(g, spec):
    if ',' in spec:
        g.atom_grid = tuple(int(x) for x in spec.split(','))
    else:
        g.level = int(spec)


def run(mf, label, dm0=None):
    cyc = []
    t_last = [time.perf_counter()]

    def cb(envs):
        now = time.perf_counter()
        cyc.append(now - t_last[0])
        t_last[0] = now
    mf.callback = cb
    t = time.perf_counter()
    mf.kernel(dm0=dm0)
    dt = time.perf_counter() - t
    err = '' if args.eref is None else f' (E - Eref {mf.e_tot - args.eref:+.2e})'
    print(f'{label}: {dt:.1f} s, {mf.cycles} cycles, conv {mf.converged}, E {mf.e_tot:.10f}{err}')
    print('   per cycle:', ' '.join(f'{c:.1f}' for c in cyc), flush=True)
    return dt


def exact():
    mf = dft.RKS(mol, xc=args.xc)
    setgrid(mf.grids, args.grid)
    setgrid(mf.nlcgrids, args.nlc)
    return mf


print(f'{args.mol}/{args.basis} {args.xc} nao {mol.nao}; mode {args.mode}', flush=True)
if args.mode == 'minao':
    run(exact(), 'exact from minao')
elif args.mode == 'pre_scf':
    mf = exact()
    mf.init_guess = 'pre_scf'
    mf.verbose = 4
    run(mf, "exact with init_guess='pre_scf' (pre-SCF included)")
else:
    from pyscf.df import addons

    def auxbasis():
        if args.aux.startswith('aug_etb'):
            beta = float(args.aux.split(':')[1]) if ':' in args.aux else 2.0
            return addons.aug_etb(mol, beta=beta)
        if args.aux == 'autoaux':
            return addons.autoaux(mol)
        return args.aux

    def rijcosx(grid, nlc):
        mf = dft.RKS(mol, xc=args.xc)
        setgrid(mf.grids, grid)
        setgrid(mf.nlcgrids, nlc)
        mf = mf.RIJCOSX(auxbasis=auxbasis())
        if args.direct_j:
            mf.with_df.dfj = False
            mf.with_df.direct_j = True
        return mf

    mf1 = rijcosx(args.pre_grid, args.pre_nlc)
    mf1.with_df.grids_level_energy = None
    mf1.conv_tol = args.pre_tol
    jlabel = 'direct J' if args.direct_j else 'RI-J ' + args.aux
    t1 = run(mf1, f'stage 1: {jlabel} + SGX-K, XC grid {args.pre_grid}, VV10 grid {args.pre_nlc}, conv_tol {args.pre_tol:g}')
    dm = mf1.make_rdm1()
    mf1 = None
    if args.mode == 'two-stage':
        t2 = run(exact(), 'stage 2: exact from the stage-1 density', dm0=dm)
    else:
        # mode nsc: the energy of the stage-1 density once, on the production XC and VV10 grids and SGX level --final-level
        t = time.perf_counter()
        mf2 = rijcosx(args.grid, args.nlc)
        mf2.with_df.grids_level_i = mf2.with_df.grids_level_f = args.final_level
        mf2.with_df.grids_level_energy = None
        mf2.build()
        vhf = mf2.get_veff(mol, dm)
        e = mf2.energy_tot(dm, vhf=vhf)
        t2 = time.perf_counter() - t
        err = '' if args.eref is None else f' (E - Eref {e - args.eref:+.2e})'
        print(f'stage 2: one-shot energy on the production grids, SGX level {args.final_level}: {t2:.1f} s, E {e:.10f}{err}')
    print(f'total {t1 + t2:.1f} s')
print('done')
