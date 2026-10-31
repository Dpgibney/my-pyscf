"""Trace of the caffeine+ UKS SCF: per cycle E, |g|, and per spin the HOMO/LUMO energies, their mirror parity
(z -> -z, the molecular plane), the number of occupied A'/A'' orbitals, and the smallest singular value of the
overlap of this cycle's occupied space with the previous one (0 = an orbital swapped in or out).
Usage: scf_hard_trace.py [XC] [cdiis|hyb|shiftX|dampX|newton] [MAX_CYCLE]"""
import sys, os, time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE
from pyscf import gto, dft, scf, lib

xc = sys.argv[1] if len(sys.argv) > 1 else 'PBE'
variant = sys.argv[2] if len(sys.argv) > 2 else 'cdiis'
max_cycle = int(sys.argv[3]) if len(sys.argv) > 3 else 100
mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=4, output=f'trace_{xc}_{variant}.log')
mf = dft.UKS(mol, xc=xc)
mf.max_cycle = max_cycle
if variant == 'hyb':
    mf.DIIS = scf.diis.ADIIS_CDIIS
elif variant.startswith('shift'):
    mf.level_shift = float(variant[5:])
elif variant.startswith('damp'):
    mf.damp = float(variant[4:]); mf.diis_start_cycle = 1
elif variant == 'newton':
    mf = mf.newton()

# mirror parity from AO values at random points and their images through z = 0
rng = numpy.random.default_rng(1)
xyz = mol.atom_coords()
pts = xyz[rng.integers(0, mol.natm, 4000)] + rng.normal(scale=1.0, size=(4000, 3))
ao = mol.eval_gto('GTOval', pts)
ao_m = mol.eval_gto('GTOval', pts * [1, 1, -1])
s = mol.intor_symmetric('int1e_ovlp')

def parity(c):
    v, vm = ao @ c, ao_m @ c
    return numpy.einsum('pi,pi->i', v, vm) / numpy.einsum('pi,pi->i', v, v)

last = [None, None]
def callback(envs):
    mo_e, mo_occ, mo_c = envs['mo_energy'], envs['mo_occ'], envs['mo_coeff']
    out = [f"cyc {envs['cycle']+1:3d} E {envs['e_tot']:.10f} |g| {envs['norm_gorb']:.2e}"]
    for sp in range(2):
        occ = mo_occ[sp] > 0
        nocc = occ.sum()
        c_occ = mo_c[sp][:, occ]
        p = parity(mo_c[sp][:, max(0, nocc-3):nocc+2])
        p_occ = parity(c_occ)
        n_ap = int((p_occ > 0).sum()); n_app = int((p_occ < 0).sum())
        if last[sp] is not None:
            sv = numpy.linalg.svd(last[sp].T @ s @ c_occ, compute_uv=False).min()
        else:
            sv = float('nan')
        last[sp] = c_occ
        h, l = mo_e[sp][nocc-1], mo_e[sp][nocc]
        out.append(f"{'ab'[sp]}: H {h:+.4f} L {l:+.4f} gap {l-h:.4f} A'/A'' {n_ap}/{n_app} "
                   f"par(H-2..L+1) {' '.join('%+d' % round(x) for x in p)} svmin {sv:.3f}")
    print(' | '.join(out), flush=True)

mf.callback = callback
t0 = time.perf_counter()
mf.kernel()
print(f'converged {mf.converged} cycles {mf.cycles} E {mf.e_tot:.10f} <S2> {mf.spin_square()[0]:.4f} '
      f't {time.perf_counter()-t0:.1f} s')
