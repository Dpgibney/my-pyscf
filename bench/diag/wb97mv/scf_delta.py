'''Prototype: the exact wB97M-V SCF solution from a few exact Fock builds.

  stage 1   RIJCOSX with coarse XC/VV10 grids, converged (cheap operator F_c)
  macro k   one exact Fock build on the production grids at the current density D_ref: V_ex[D_ref];
            the exact energy and orbital gradient at D_ref (the convergence test of the exact SCF)
  micro     DIIS cycles with F[D] = h + V_ex[D_ref] + (V_c[D] - V_c[D_ref]): only cheap builds

J and K are linear in D and the grid error of XC is smooth in D, so the error of F is that of the cheap operator
on D - D_ref: second order. Prints the exact energy and gradient norm at each macro step and all timings.
Usage: scf_delta.py MOL [--pre-grid 2] [--pre-nlc 1] [--pre-tol 1e-7] [--macro 3] [--eref E]'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, scf, lib
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlc', default='50,194')
ap.add_argument('--pre-grid', default='2')
ap.add_argument('--pre-nlc', default='1')
ap.add_argument('--pre-tol', type=float, default=1e-7)
ap.add_argument('--micro-tol', type=float, default=1e-6, help='orbital gradient norm at which the micro cycles stop')
ap.add_argument('--aux', default='def2-universal-jfit')
ap.add_argument('--macro', type=int, default=3)
ap.add_argument('--eref', type=float, default=None)
ap.add_argument('--save', default=None)
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


T0 = time.perf_counter()
# ---- stage 1
mfx = dft.RKS(mol, xc=args.xc)
setgrid(mfx.grids, args.pre_grid)
setgrid(mfx.nlcgrids, args.pre_nlc)
if args.aux == 'aug_etb':
    from pyscf.df import addons
    mfx = mfx.RIJCOSX(auxbasis=addons.aug_etb(mol))
else:
    mfx = mfx.RIJCOSX(auxbasis=args.aux)
mfx.with_df.grids_level_energy = None
mfx.conv_tol = args.pre_tol
t = time.perf_counter()
mfx.kernel()
t_stage1 = time.perf_counter() - t
print(f'stage 1 (RIJCOSX, XC grid {args.pre_grid}, VV10 grid {args.pre_nlc}, conv_tol {args.pre_tol:g}): {t_stage1:.1f} s, '
      f'{mfx.cycles} cycles, E {mfx.e_tot:.10f}', flush=True)
dm = mfx.make_rdm1()
mo_coeff, mo_occ = mfx.mo_coeff, mfx.mo_occ

# ---- the exact object
mfe = dft.RKS(mol, xc=args.xc)
setgrid(mfe.grids, args.grid)
setgrid(mfe.nlcgrids, args.nlc)
s1e = mfe.get_ovlp()
# as SCF.kernel: the eigenvectors of the overlap below 1e-6 are left out of the orbital space
x_orth = mfe.check_linear_dependency(s1e)
print(f'orbital space: {x_orth.shape[1]} of {x_orth.shape[0]} functions', flush=True)
h1e = mfe.get_hcore()


def cheap(dm):
    t = time.perf_counter()
    mfx._nsteps_direct = 0   # a full build
    v = mfx.get_veff(mol, dm)
    return numpy.asarray(v), time.perf_counter() - t


def gnorm(fock, mo_coeff, mo_occ):
    return numpy.linalg.norm(mfe.get_grad(mo_coeff, mo_occ, fock))


t_exact = t_cheap = 0.
n_exact = n_cheap = 0
for macro in range(args.macro):
    t = time.perf_counter()
    vref = mfe.get_veff(mol, dm)
    e_ex = mfe.energy_tot(dm, h1e, vref)
    dt = time.perf_counter() - t
    t_exact += dt
    n_exact += 1
    fock = h1e + numpy.asarray(vref)
    g = gnorm(fock, mo_coeff, mo_occ)
    err = '' if args.eref is None else f'  E - Eref {e_ex - args.eref:+.2e}'
    print(f'macro {macro}: exact build {dt:.1f} s; exact E {e_ex:.10f}{err}; exact |g| {g:.2e} '
          f'(conv_tol_grad of conv_tol 1e-9: 3.2e-05)   [{time.perf_counter() - T0:.0f} s so far]', flush=True)
    if macro == args.macro - 1:
        break
    vref = numpy.asarray(vref)
    vx0, dt = cheap(dm)
    t_cheap += dt
    n_cheap += 1
    diis = scf.diis.CDIIS()
    dm_ref = dm
    for it in range(30):
        if it > 0:
            vx, dt = cheap(dm)
            t_cheap += dt
            n_cheap += 1
            fock = h1e + vref + (vx - vx0)
            g = gnorm(fock, mo_coeff, mo_occ)
            print(f'   micro {it}: |g| {g:.2e}  |D - D_ref| {numpy.linalg.norm(dm - dm_ref):.2e}  cheap build {dt:.1f} s', flush=True)
            if g < args.micro_tol:
                break
        f = diis.update(s1e, dm, fock)
        mo_energy, mo_coeff = mfe.eig(f, s1e, x=x_orth)
        mo_occ = mfe.get_occ(mo_energy, mo_coeff)
        dm = mfe.make_rdm1(mo_coeff, mo_occ)
if args.save:
    numpy.savez(args.save, mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy, e_tot=e_ex)
print(f'totals: stage 1 {t_stage1:.1f} s; {n_exact} exact builds {t_exact:.1f} s; {n_cheap} cheap builds {t_cheap:.1f} s; '
      f'all {time.perf_counter() - T0:.1f} s')
print('done')
