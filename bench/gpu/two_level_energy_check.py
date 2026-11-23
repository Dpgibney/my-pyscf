'''The energy an SCF reports against the energy of its final density built from scratch.

Every SCF here takes its energy from potentials that were put together from changes of the density: J and K
of the cycles (and of the references of a two-level SCF), each screened for its own change. This script runs
the SCF in several ways, then evaluates the model's potential of each final density in one build with no
history, once with the model's direct_scf_tol and once with a tight one (--tol), and prints what the SCF's
energy differs from them by, in all and by terms (J, as tr(D J)/2 of the SCF's J matrix; the rest of the
two-electron energy: K together with the XC terms, which are evaluated the same way in both, and what the
second-order form of a two-level reference adds for its K, and for its J where the cheap model's is fitted;
the solvent), and the orbital gradient of the tight build. For the plain SCF it also
follows the two-electron energy through the cycles in the form that takes each change of the potential at
second order, E2 += tr(dD V_last) + tr(dD dV)/2, in place of tr(D V)/2 with the potential summed up.

    python bench/gpu/two_level_energy_check.py [--mol caffeine] [--basis def2-tzvp] [--xc wb97m-v] [--smd]
        [--grid 99,590] [--nlcgrid 50,194] [--runs plain,two_level,ref2,ref2f,ref3] [--tol 1e-16]

Runs: plain; two_level (the model's own K at every reference); refN / refNf (the change of K at the later
references from the seminumerical K on SGX level N, in double / single precision: two_level_sgx_ref_level);
"+etb" for the fitted J of the cheap model (two_level_auxbasis = 'aug_etb').
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.lib.cupy_helper import tag_array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--xc', default='wb97m-v')
p.add_argument('--smd', action='store_true')
p.add_argument('--grid', default=None)
p.add_argument('--nlcgrid', default=None)
p.add_argument('--conv', type=float, default=1e-10)
p.add_argument('--runs', default='plain,two_level,ref2,ref2f,ref3')
p.add_argument('--tol', type=float, default=1e-16, help='direct_scf_tol of the tight build')
p.add_argument('--no-loose', action='store_true', help='skip the build with the direct_scf_tol of the model')
args = p.parse_args()

def sync():
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter()

atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
        'water': molecules.WATER}.get(args.mol, args.mol)
mol = pyscf.M(atom=atom, basis=args.basis, charge=args.charge, verbose=0)

def model():
    mf = dft.RKS(mol, xc=args.xc)
    if args.grid:
        mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
    if args.nlcgrid:
        mf.nlcgrids.atom_grid = tuple(int(x) for x in args.nlcgrid.split(','))
    if args.smd:
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    mf.conv_tol = args.conv
    return mf

def terms(dm, vhf):
    '''(J, the other two-electron terms, solvent) of the energy in the tags of a potential'''
    e2 = float(vhf.ecoul) + float(vhf.exc)
    vk = getattr(vhf, 'vk', None)
    if vk is None:
        e_j = float(vhf.ecoul)
    else:
        # the tags of a hybrid's potential: vj holds J - K/2, vk the K/2
        e_j = .5 * float(cp.einsum('ij,ji', cp.asarray(dm), cp.asarray(vhf.vj) + cp.asarray(vk)).real)
    e_s = getattr(vhf, 'e_solvent', None)
    return np.array([e_j, e2 - e_j, 0. if e_s is None else float(e_s)])

def tr(a, b):
    return float(cp.einsum('ij,ji', a, b).real)

def one_build(mf, dm, mo_coeff, mo_occ):
    '''The energy, its terms, tr(D (J - K/2))/2 and the orbital gradient of dm from one build of the potential'''
    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    # the density on the grids from the orbitals, as in the SCF
    vhf = mf.get_veff(mol, tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ))
    e = float(mf.energy_tot(dm, h1e, vhf))
    g = float(cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, mf.get_fock(h1e, s1e, vhf, dm))))
    return e, terms(dm, vhf), .5 * tr(dm, cp.asarray(vhf.vj)), g

print('%s %s %s%s: %d AOs; one build of the potential with direct_scf_tol %g (the model: %g)' % (
    args.mol, args.basis, args.xc, ' SMD water' if args.smd else '', mol.nao, args.tol,
    model().direct_scf_tol), flush=True)
results = []
for run in args.runs.split(','):
    mf = model()
    name, _, extra = run.partition('+')
    if name != 'plain':
        mf.two_level = True
        if name.startswith('ref'):
            single = name.endswith('f')
            mf.two_level_sgx_ref_level = int(name[3:-1] if single else name[3:])
            mf.two_level_sgx_ref_dtype = 'float' if single else 'double'
        elif name != 'two_level':
            raise SystemExit('unknown run ' + run)
    if extra == 'etb':
        mf.two_level_auxbasis = 'aug_etb'
    elif extra:
        raise SystemExit('unknown run ' + run)
    last = {}
    def keep(envs, follow=(name == 'plain')):
        # the last cycle's density and the terms of the energy the kernel took for it
        dm = cp.asarray(envs['dm'])
        if follow:
            # tr(D (J - K/2))/2 with the potential as it is summed up over the cycles, and by
            # the second-order form from the first cycle seen on
            v = cp.asarray(envs['vhf'].vj)
            if 'v' in last:
                ddm = dm - last['dm']
                last['e2_form'] += tr(ddm, last['v']) + .5 * tr(ddm, v - last['v'])
            else:
                last['e2_form'] = .5 * tr(dm, v)
            last.update(v=v.copy(), e2=.5 * tr(dm, v))
        last.update(dm=dm.copy(), e=float(envs['e_tot']), terms=terms(dm, envs['vhf']),
                    mo_coeff=cp.asarray(envs['mo_coeff']).copy(), mo_occ=cp.asarray(envs['mo_occ']).copy())
    mf.callback = keep
    t0 = sync()
    e = mf.kernel()
    t = sync() - t0
    results.append((run, t, mf.cycles, last))
    print('   %-14s E = %.12f  %d cycles  %.1f s%s' % (
        run, e, mf.cycles, t,
        '' if abs(e - last['e']) < 1e-13 else '  (the last cycle seen: %.12f)' % last['e']), flush=True)
    mf = None
    cp.get_default_memory_pool().free_all_blocks()

loose = model()
tight = model()
tight.direct_scf_tol = args.tol
print('run             SCF s    E of the SCF        - one build   - tight build:  in all        J     K and XC      solvent'
      '   |g| of the tight build   tight build - that of the first run')
e_first = None
for run, t, cycles, last in results:
    dm, mo_coeff, mo_occ = last['dm'], last['mo_coeff'], last['mo_occ']
    e_loose = np.nan if args.no_loose else one_build(loose, dm, mo_coeff, mo_occ)[0]
    e_tight, terms_tight, e2_tight, g = one_build(tight, dm, mo_coeff, mo_occ)
    if e_first is None:
        e_first = e_tight
    d = last['terms'] - terms_tight
    print('%-14s %6.1f   %.12f   %+.2e                    %+.2e  %+.2e  %+.2e  %+.2e   %.2e                 %+.2e' % (
        run, t, last['e'], last['e'] - e_loose, last['e'] - e_tight, d[0], d[1], d[2], g, e_tight - e_first),
        flush=True)
    if 'e2_form' in last:
        print('   tr(D (J - K/2))/2 of this SCF against the tight build: with the potential summed up over the cycles '
              '%+.2e, in the second-order form through the cycles %+.2e' % (
                  last['e2'] - e2_tight, last['e2_form'] - e2_tight), flush=True)
