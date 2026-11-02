"""Broken-symmetry singlets (node1): UKS with spin 0 from a guess whose alpha and beta densities differ, so that the SCF
converges to alpha != beta (a diradical's two unpaired electrons on different centres). Neither closed shells
(alpha == beta) nor doublets (different electron counts) test that case. The SCF starts from a saved guess, the same in
every tree; the script reports E, <S^2>, the Mulliken spin populations of the molecule's two halves and the cycles,
then the gradient and the Hessian.
--save PREFIX keeps the results and orbitals; --compare PREFIX compares with another run's, and also takes the gradient
(with --hess-at-ref the Hessian too) at that run's orbitals, which factors out the two SCFs' convergence: a
broken-symmetry state's soft spin-polarization mode lets two codes stop at orbitals that differ at the 1e-6 level.
(PySCF's CPU Hessian takes part of its input from the SCF object, so on the CPU it is not fully at those orbitals.)
--cpu runs PySCF on the CPU instead (the independent reference). --make-guess builds the guess: RKS B3LYP on the CPU,
then its HOMO and LUMO mixed at 45 degrees, + for alpha and - for beta. --guess none takes the default guess instead
(controls: the closed shell, or the triplet with --spin 2), and --guess breaksym the default guess with
init_guess_breaksym = True. The tda task runs TDA (5 states) from the reference.
Usage: bs_singlet.py --mol twisted_ethylene|benzyl_dimer --guess GUESS.npy|none|breaksym [--make-guess] [--spin 0]
                     [--xc b3lyp] [--basis def2-svp] [--df] [--cpscf-tol T] [--cphf-main-grid]
                     [--tasks scf,grad,hess,tda] [--cpu] [--label L] [--save PREFIX] [--compare PREFIX [--hess-at-ref]]
                     [--out FILE]"""
import argparse
import json
import os
import sys
import time
import numpy
from pyscf import gto

# twisted (90 degree) ethylene, the two CH2 halves; C-C 1.48 A as in the twisted diradical
TWISTED_ETHYLENE = '''
C  0.000  0.000  0.000
H -0.569  0.929  0.000
H -0.569 -0.929  0.000
C  1.480  0.000  0.000
H  2.049  0.000  0.929
H  2.049  0.000 -0.929'''


def _benzyl(flip, z):
    # planar benzyl radical: ring C-C 1.40 A, exocyclic C-C 1.40 A, C-H 1.08 A; flip turns it by 180 degrees about z
    s = -1 if flip else 1
    c, h = [], []
    for k in range(6):
        a = numpy.radians(60 * k)
        c.append((1.40 * numpy.cos(a), 1.40 * numpy.sin(a)))
        if k:
            h.append((2.48 * numpy.cos(a), 2.48 * numpy.sin(a)))
    c.append((2.80, 0.))
    h += [(2.80 + 1.08 * 0.5, 1.08 * numpy.sqrt(.75)), (2.80 + 1.08 * 0.5, -1.08 * numpy.sqrt(.75))]
    return [('C', s * x, s * y, z) for x, y in c] + [('H', s * x, s * y, z) for x, y in h]


# two benzyl radicals stacked 3.8 A apart, the second turned by 180 degrees: one radical's spin per molecule
BENZYL_DIMER = '\n'.join(f'{e} {x:.4f} {y:.4f} {z:.4f}' for e, x, y, z in _benzyl(False, 0.) + _benzyl(True, 3.8))
MOLS = {'twisted_ethylene': (TWISTED_ETHYLENE, 3), 'benzyl_dimer': (BENZYL_DIMER, 14)}

ap = argparse.ArgumentParser()
ap.add_argument('--mol', required=True, choices=list(MOLS))
ap.add_argument('--guess', required=True)  # 'none': the default initial guess (a control: closed shell, or --spin 2)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--make-guess', action='store_true')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--df', action='store_true')
ap.add_argument('--cpscf-tol', type=float, default=None)
ap.add_argument('--cphf-main-grid', action='store_true')  # GPU4PySCF: the CPHF on the SCF's grid, as PySCF does
ap.add_argument('--tasks', default='scf,grad,hess')
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--label', default=None)
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--hess-at-ref', action='store_true')
ap.add_argument('--out', default=None)
args = ap.parse_args()
tasks = args.tasks.split(',')
atom, nfirst = MOLS[args.mol]
mol = gto.M(atom=atom, basis=args.basis, spin=args.spin, verbose=0)

if args.make_guess:
    from pyscf import dft
    rks = dft.RKS(mol, xc='b3lyp')
    rks.conv_tol = 1e-9
    rks.max_cycle = 200
    rks.kernel()
    c = rks.mo_coeff
    h = mol.nelectron // 2 - 1
    occ = c[:, :h]
    homo, lumo = c[:, h], c[:, h + 1]
    dms = [numpy.hstack([occ, ((homo + sign * lumo) / numpy.sqrt(2))[:, None]]) for sign in (1, -1)]
    numpy.save(args.guess, numpy.array([x @ x.T for x in dms]))
    print(f'guess for {args.mol}: RKS B3LYP converged {rks.converged}, HOMO-LUMO gap '
          f'{rks.mo_energy[h + 1] - rks.mo_energy[h]:.4f} Eh, saved to {args.guess}')
    sys.exit()

dm0 = None if args.guess in ('none', 'breaksym') else numpy.load(args.guess)
if args.cpu:
    from pyscf import dft
    # these modules give the SCF objects their .Gradients and .Hessian
    import pyscf.grad  # noqa: F401
    import pyscf.hessian  # noqa: F401
    import pyscf.df.grad  # noqa: F401
    import pyscf.df.hessian  # noqa: F401
    xp = numpy
    mf = dft.UKS(mol, xc=args.xc)
    tree = 'pyscf ' + __import__('pyscf').__version__ + ' (CPU)'
else:
    import cupy
    import gpu4pyscf
    from gpu4pyscf.dft import uks
    xp = cupy
    mf = uks.UKS(mol, xc=args.xc)
    tree = os.path.dirname(gpu4pyscf.__file__)
    dm0 = None if dm0 is None else cupy.asarray(dm0)
if args.df:
    mf = mf.density_fit(auxbasis='def2-universal-jkfit')
if args.guess == 'breaksym':  # the code's own symmetry-breaking guess (PySCF's default; GPU4PySCF's is off)
    mf.init_guess_breaksym = True
mf.conv_tol = 1e-10
mf.max_cycle = 200
if args.cpscf_tol is not None:  # the Hessian's CPHF; defaults: GPU4PySCF 1e-6, PySCF 1e-8
    mf.conv_tol_cpscf = args.cpscf_tol
if args.cphf_main_grid:  # GPU4PySCF's default cphf_grids is a coarser (50, 194) SG-1 grid
    mf.cphf_grids = mf.grids


def sync():
    if not args.cpu:
        cupy.cuda.Device().synchronize()


def host(x):
    return cupy.asnumpy(x) if not args.cpu and isinstance(x, cupy.ndarray) else numpy.asarray(x)


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t0, out


times, res = {}, {}
times['scf'], e = timed(lambda: mf.kernel(dm0=dm0))
res['e'] = numpy.array([e])
ss = float(mf.spin_square()[0])
res['s2'] = numpy.array([ss])
# Mulliken spin populations of the two halves
s = mol.intor('int1e_ovlp')
dma, dmb = (host(x) for x in mf.make_rdm1())
pop = numpy.einsum('ij,ji->i', dma - dmb, s)
atoms = mol.ao_labels(fmt=False)
first = numpy.array([a[0] < nfirst for a in atoms])
res['spin'] = numpy.array([pop[first].sum(), pop[~first].sum()])
if 'grad' in tasks:
    times['grad'], g = timed(lambda: mf.Gradients().kernel())
    res['grad'] = host(g)
if 'hess' in tasks:
    try:
        times['hess'], h = timed(lambda: mf.Hessian().kernel())
        res['hess'] = host(h)
    except NotImplementedError as err:
        print(f'no Hessian: {err}')
if 'tda' in tasks:  # TDA (5 states) from the broken-symmetry reference
    td = mf.TDA()
    td.nstates = 5
    if not args.cpu:
        td.max_memory = 1000  # GPU4PySCF sizes its trial space from max_memory (see overview.py)
    times['tda'], out = timed(td.kernel)
    res['tda'] = host(out[0])
record = {'label': args.label, 'tree': tree, 'mol': args.mol, 'spin': args.spin, 'guess': args.guess != 'none',
          'xc': args.xc, 'df': args.df,
          'conv_tol_cpscf': mf.conv_tol_cpscf, 'cphf_main_grid': args.cphf_main_grid, 'converged': bool(mf.converged),
          'cycles': getattr(mf, 'cycles', None), 'E': e, 'S2': ss, 'spin_halves': [round(x, 6) for x in res['spin']],
          'times': {k: round(v, 2) for k, v in times.items()}}
if args.save:
    numpy.savez(args.save, mo_coeff=host(mf.mo_coeff), mo_occ=host(mf.mo_occ), mo_energy=host(mf.mo_energy), **res)
if args.compare:
    ref = numpy.load(args.compare + '.npz')
    record['diff'] = {k: float(abs(v - ref[k]).max()) for k, v in res.items() if k in ref}
    if 'grad' in res and 'grad' in ref:
        # the gradient at the other run's orbitals: the code's difference without the SCFs'
        g = mf.Gradients().kernel(mo_energy=xp.asarray(ref['mo_energy']), mo_coeff=xp.asarray(ref['mo_coeff']),
                                  mo_occ=xp.asarray(ref['mo_occ']))
        record['diff']['grad_at_ref_orbitals'] = float(abs(host(g) - ref['grad']).max())
    if args.hess_at_ref and 'hess' in res and 'hess' in ref:
        h = mf.Hessian().kernel(mo_energy=xp.asarray(ref['mo_energy']), mo_coeff=xp.asarray(ref['mo_coeff']),
                                mo_occ=xp.asarray(ref['mo_occ']))
        record['diff']['hess_at_ref_orbitals'] = float(abs(host(h) - ref['hess']).max())
    record['max_ref'] = {k: float(abs(ref[k]).max()) for k in ('grad', 'hess') if k in ref}
print(json.dumps(record), flush=True)
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
