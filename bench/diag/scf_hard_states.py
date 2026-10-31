"""The SCF solutions of caffeine+ UKS (def2-SVP): the two Cs-symmetric states with the beta hole in an A'' (pi)
or an A' (sigma) orbital, converged with fixed irrep occupations, and the symmetry-broken state the CDIIS run
ends in, converged tightly with SOSCF. Prints energies, the beta orbital energies around the hole, and the
internal stability of each solution (in the non-symmetric UKS, so symmetry-breaking rotations count).
Usage: scf_hard_states.py [XC]"""
import sys, os, time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE
from pyscf import gto, dft, scf, lib

xc = sys.argv[1] if len(sys.argv) > 1 else 'PBE'
log = f'states_{xc}.log'
mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=4, output=log)
mol_s = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, symmetry=True, verbose=4, output=log + '.symm')
assert abs(mol_s.atom_coords() - mol.atom_coords()).max() == 0  # same frame, so the same grids

def report(tag, mf, t):
    e_b, o_b = mf.mo_energy[1], mf.mo_occ[1]
    hb, lb = e_b[o_b > 0].max(), e_b[o_b == 0].min()
    print(f'{tag:10s} conv {mf.converged} cyc {mf.cycles:3d} E {mf.e_tot:.10f} <S2> {mf.spin_square()[0]:.4f} '
          f'beta: highest occ {hb:+.5f} lowest empty {lb:+.5f} ({"aufbau" if lb > hb else "NOT aufbau"}) '
          f't {t:.1f} s', flush=True)

def stability(tag, mf_s):
    # copy the solution into a plain UKS object and test internal stability there
    mf = dft.UKS(mol, xc=xc)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = [numpy.asarray(x) for x in mf_s.mo_coeff], numpy.asarray(mf_s.mo_occ), numpy.asarray(mf_s.mo_energy)
    mf.e_tot, mf.converged = mf_s.e_tot, True
    from pyscf.scf import stability as st
    mo_i, _, stable, _ = st.uhf_stability(mf, internal=True, external=False, return_status=True, verbose=4)
    print(f'{tag:10s} internally stable: {stable}', flush=True)

res = {}
for tag, irrep_nelec in [('sym_pi', {"A'": (41, 41), 'A"': (10, 9)}),
                         ('sym_sigma', {"A'": (41, 40), 'A"': (10, 10)})]:
    mf = dft.UKS(mol_s, xc=xc)
    mf.irrep_nelec = irrep_nelec
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    t0 = time.perf_counter()
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run(mf.mo_coeff, mf.mo_occ)
    report(tag, mf, time.perf_counter() - t0)
    res[tag] = mf.e_tot
    stability(tag, mf)

mf = dft.UKS(mol, xc=xc)
mf.max_cycle = 100                # as in the guess/DIIS study; PySCF's default of 50 stops in the flipping phase
t0 = time.perf_counter()
mf.kernel()                       # ends near the symmetry-broken state
report('cdiis', mf, time.perf_counter() - t0)
t0 = time.perf_counter()
mf2 = dft.UKS(mol, xc=xc).newton()
mf2.conv_tol = 1e-10
mf2.kernel(mf.mo_coeff, mf.mo_occ)
report('bs_soscf', mf2, time.perf_counter() - t0)
res['bs'] = mf2.e_tot
stability('bs_soscf', mf2)
for k, v in res.items():
    print(f'{k:10s} E - E(bs) = {v - res["bs"]:+.6f} Eh = {(v - res["bs"]) * 627.5095:+.3f} kcal/mol')
