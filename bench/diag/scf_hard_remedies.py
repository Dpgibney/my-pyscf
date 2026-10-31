"""Convergence remedies on caffeine+ UKS (def2-SVP, minao guess unless stated): cycles, Fock builds, wall time,
whether it converged (with PySCF's extra cycle), and which solution it ends on: the energy above the lowest
solution E_REF, the beta gap, and whether the occupied beta orbitals keep the mirror symmetry.
Usage: scf_hard_remedies.py XC E_REF OUT variant [variant ...]"""
import sys, os, time, json
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE
from pyscf import gto, dft, scf, lib

xc, e_ref, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
variants = sys.argv[4:]
mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=0)

rng = numpy.random.default_rng(1)
pts = mol.atom_coords()[rng.integers(0, mol.natm, 4000)] + rng.normal(scale=1.0, size=(4000, 3))
ao, ao_m = mol.eval_gto('GTOval', pts), mol.eval_gto('GTOval', pts * [1, 1, -1])
def parity(c):
    v, vm = ao @ c, ao_m @ c
    return numpy.einsum('pi,pi->i', v, vm) / numpy.einsum('pi,pi->i', v, v)

def describe(mf):
    e_b, o_b, c_b = mf.mo_energy[1], mf.mo_occ[1], mf.mo_coeff[1]
    p = parity(c_b[:, o_b > 0])
    return dict(e_tot=float(mf.e_tot), de=float(mf.e_tot - e_ref),
                beta_gap=float(e_b[o_b == 0].min() - e_b[o_b > 0].max()),
                sym_broken=bool(abs(p).min() < 0.99), beta_npi=int((p < -0.99).sum()),
                s2=float(mf.spin_square()[0]))

LOG = [None]
def make(xc_=None):
    mf = dft.UKS(mol, xc=xc_ or xc)
    mf.max_cycle = 100
    mf.verbose, mf.stdout = 4, LOG[0]
    return mf

def builds(path):
    '''Fock builds of the DIIS runs (initial guess, cycles, extra cycle) and the J/K builds SOSCF reports'''
    n = 0
    for line in open(path):
        if line.startswith('init E=') or line.startswith('cycle=') or line.startswith('Extra cycle'):
            n += 1
        elif ' KF ' in line and line.rstrip().endswith('JK') and 'macro' in line and 'total' in line:
            n += int(line.split()[-2])
    return n

def run(variant):
    logpath = f'remedy_{xc}_{variant}.log'
    LOG[0] = open(logpath, 'w')
    t0 = time.perf_counter()
    rec = {'variant': variant, 'xc': xc}
    if variant == 'cdiis':
        mf = make(); mf.kernel()
    elif variant == 'cdiis50':
        mf = make(); mf.max_cycle = 50; mf.kernel()
    elif variant.startswith('tol'):          # tighter convergence: does the extra cycle pass then?
        mf = make(); mf.conv_tol = float(variant[3:]); mf.kernel()
    elif variant == 'hyb':
        mf = make(); mf.DIIS = scf.diis.ADIIS_CDIIS; mf.kernel()
    elif variant.startswith('shift'):
        mf = make(); mf.level_shift = float(variant[5:]); mf.kernel()
    elif variant.startswith('damp'):
        mf = make(); mf.damp = float(variant[4:]); mf.diis_start_cycle = 1; mf.kernel()
    elif variant == 'newton':
        mf = make().newton(); mf.kernel()
    elif variant == 'cdiis50+newton':        # the usual rescue after a failed DIIS
        mf = make(); mf.max_cycle = 50; mf.kernel()
        rec['cycles_diis'] = mf.cycles
        mf2 = make().newton(); mf2.kernel(mf.mo_coeff, mf.mo_occ); mf = mf2
    elif variant.startswith('smear'):        # Fermi smearing, then the integer-occupation SCF from its density
        sigma = float(variant[5:])
        mf0 = scf.addons.smearing(make(), sigma=sigma, method='fermi'); mf0.kernel()
        rec['cycles_smear'] = mf0.cycles
        rec['smear_conv'] = bool(mf0.converged)
        mf = make(); mf.kernel(mf0.make_rdm1())
    elif variant == 'b3lyp_guess':
        mf0 = make('B3LYP'); mf0.kernel()
        rec['cycles_guess'] = mf0.cycles
        mf = make(); mf.kernel(mf0.make_rdm1())
    elif variant == 'pre_scf':
        mf = make(); mf.init_guess = 'pre_scf'; mf.kernel()
    else:
        raise KeyError(variant)
    t = time.perf_counter() - t0
    LOG[0].flush()
    rec.update(cycles=mf.cycles, converged=bool(mf.converged), fock_builds=builds(logpath), time=t)
    rec.update(describe(mf))
    macro = [l for l in open(logpath) if l.startswith('macro X = ')]
    if macro:
        rec['soscf_macro'] = int(macro[-1].split()[3])
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(' '.join(f'{k} {v:.3e}' if isinstance(v, float) and k in ('de', 'beta_gap') else
                   f'{k} {v:.10f}' if k == 'e_tot' else f'{k} {v:.1f}' if k == 'time' else f'{k} {v}'
                   for k, v in rec.items()), flush=True)

for v in variants:
    run(v)
