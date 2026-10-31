"""Initial guess from a cheaper SCF (It21): the same functional in a smaller basis and on a coarser grid,
converged loosely, its occupied orbitals projected onto the target basis and orthonormalized there. Prints, per
molecule, the minao-guess SCF (cycles, time) against the pre-SCF (cycles, time) plus the SCF from its guess.
Usage: scf_proj_guess.py --basis def2-tzvp --small def2-svp [--xc B3LYP] [--level 1] [--tol 1e-6] [--mols ...]"""
import argparse
import glob
import os
import sys
import time
import numpy
import scipy.linalg
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--small', default='minao')
ap.add_argument('--xc', default='B3LYP')
ap.add_argument('--level', type=int, default=1)
ap.add_argument('--tol', type=float, default=1e-6)
ap.add_argument('--df', action='store_true')
ap.add_argument('--pre-xc', default=None)
ap.add_argument('--mols', default='aspirin,caffeine,tryptophan,porphine,retinal,tempo')
args = ap.parse_args()

mols = {}
for f in sorted(glob.glob(os.path.join(here, '..', 'geom', '*.xyz'))):
    lines = open(f).read().splitlines()
    charge, spin = map(int, lines[1].split()[:2])
    mols[os.path.basename(f)[:-4]] = ('\n'.join(lines[2:]), charge, spin)
mols['benzene'] = (BENZENE, 0, 0)
mols['caffeine'] = (CAFFEINE, 0, 0)
mols['caffeine+'] = (CAFFEINE, 1, 1)
mols['c20h42'] = (n_alkane(20), 0, 0)


def project_occ(mol_s, mf_s, mol):
    '''Density of the occupied orbitals of mf_s projected onto mol's basis, orthonormalized there'''
    s = mol.intor_symmetric('int1e_ovlp')
    s12 = gto.intor_cross('int1e_ovlp', mol, mol_s)
    mo_coeff, mo_occ = mf_s.mo_coeff, mf_s.mo_occ
    if numpy.ndim(mo_occ) == 1:
        mo_coeff, mo_occ = [mo_coeff], [mo_occ]
    dms = []
    for c, occ in zip(mo_coeff, mo_occ):
        c = c[:, occ > 0]
        p = scipy.linalg.solve(s, s12 @ c, assume_a='pos')
        w, v = numpy.linalg.eigh(p.T @ s @ p)
        p = p @ (v * w**-.5) @ v.T
        dms.append((p * occ[occ > 0]) @ p.T)
    return dms[0] if len(dms) == 1 else numpy.asarray(dms)


for name in args.mols.split(','):
    atom, charge, spin = mols[name]
    mol = gto.M(atom=atom, basis=args.basis, charge=charge, spin=spin, verbose=0)
    KS = dft.RKS if spin == 0 else dft.UKS
    t0 = time.perf_counter()
    mf = KS(mol, xc=args.xc).run()
    t1 = time.perf_counter()
    ref_cycles, e_ref = mf.cycles, mf.e_tot

    mol_s = gto.M(atom=atom, basis=args.small, charge=charge, spin=spin, verbose=0)
    t2 = time.perf_counter()
    mf_s = KS(mol_s, xc=args.pre_xc or args.xc)
    mf_s.grids.level = args.level
    mf_s.conv_tol = args.tol
    if args.df:
        mf_s = mf_s.density_fit()
    mf_s.kernel()
    dm0 = project_occ(mol_s, mf_s, mol)
    t3 = time.perf_counter()
    mf2 = KS(mol, xc=args.xc)
    mf2.kernel(dm0=dm0)
    t4 = time.perf_counter()
    print(f'{name:14s} {args.basis} pre-xc {args.pre_xc or args.xc}: minao {ref_cycles:3d} cyc {t1 - t0:6.1f} s | pre {args.small} '
          f'{mf_s.cycles:3d} cyc {t3 - t2:5.1f} s + {mf2.cycles:3d} cyc {t4 - t3:6.1f} s = {t4 - t2:6.1f} s'
          f'  dE {mf2.e_tot - e_ref:.1e}', flush=True)
