"""Symmetry-broken starts for caffeine+ UKS (def2-SVP). Prototype of 'flip-pair mixing': run CDIIS until the
aufbau occupation of one spin oscillates (an orbital leaves the occupied space and the same orbital comes back),
then replace the orbital that came in by the normalized sum of it and the one that left, and restart the SCF
from that density. Also: the same mixing applied to the converged Cs-symmetric pi-hole state.
Usage: scf_flipmix.py XC E_REF OUT variant [variant ...]"""
import sys, os, time, json
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE
from pyscf import gto, dft, scf, lib

xc, e_ref, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=0)
s1e = mol.intor_symmetric('int1e_ovlp')

rng = numpy.random.default_rng(1)
pts = mol.atom_coords()[rng.integers(0, mol.natm, 4000)] + rng.normal(scale=1.0, size=(4000, 3))
ao, ao_m = mol.eval_gto('GTOval', pts), mol.eval_gto('GTOval', pts * [1, 1, -1])
def parity(c):
    v, vm = ao @ c, ao_m @ c
    return numpy.einsum('pi,pi->i', v, vm) / numpy.einsum('pi,pi->i', v, v)

FIRST = int(os.environ.get('FLIP_FIRST', 4))
SWAP = float(os.environ.get('FLIP_SWAP', 0.1))

class Flip(Exception):
    pass

def detect_and_mix(mf, max_cycle=100):
    '''CDIIS with a callback that stops at the first occupation oscillation and returns the mixed density'''
    hist = {0: [], 1: []}   # per spin: (cycle, orbital that left, orbital that came in)
    last = [None, None]
    found = {}
    def callback(envs):
        c, occ = envs['mo_coeff'], envs['mo_occ']
        for sp in range(2):
            cur = c[sp][:, occ[sp] > 0]
            # a clean swap (one orbital fully out, another fully in) after the first Roothaan steps
            if last[sp] is not None and envs['cycle'] + 1 >= FIRST:
                u, sv, vt = numpy.linalg.svd(last[sp].T @ s1e @ cur)
                if sv[-1] < SWAP:
                    out_ = last[sp] @ u[:, -1]
                    in_ = cur @ vt[-1]
                    for cyc, o1, i1 in hist[sp]:
                        # the orbital coming in now is one that left before: an oscillation
                        if abs(in_ @ s1e @ o1) > 0.9:
                            others = cur @ vt[:-1].T
                            mixed = in_ + out_ * numpy.sign(in_ @ s1e @ out_ or 1.)
                            mixed /= numpy.sqrt(mixed @ s1e @ mixed)
                            found.update(spin=sp, cycle=envs['cycle'] + 1, others=others, mixed=mixed,
                                         dm=[None, None], c=c, occ=occ)
                            raise Flip
                    hist[sp].append((envs['cycle'] + 1, out_, in_))
            last[sp] = cur
    mf.callback = callback
    mf.max_cycle = max_cycle
    try:
        mf.kernel()
        return None
    except Flip:
        sp = found['spin']
        dms = []
        for s in range(2):
            if s == sp:
                cocc = numpy.hstack([found['others'], found['mixed'][:, None]])
            else:
                cocc = found['c'][s][:, found['occ'][s] > 0]
            dms.append(cocc @ cocc.T)
        p = parity(found['mixed'][:, None])[0]
        return numpy.asarray(dms), found['cycle'], sp, p

def describe(mf):
    e_b, o_b, c_b = mf.mo_energy[1], mf.mo_occ[1], mf.mo_coeff[1]
    p = parity(c_b[:, o_b > 0])
    return dict(e_tot=float(mf.e_tot), de=float(mf.e_tot - e_ref),
                beta_gap=float(e_b[o_b == 0].min() - e_b[o_b > 0].max()),
                sym_broken=bool(abs(p).min() < 0.99), s2=float(mf.spin_square()[0]))

def make(log):
    mf = dft.UKS(mol, xc=xc)
    mf.max_cycle = 100
    mf.verbose, mf.stdout = 4, log
    return mf

for variant in sys.argv[4:]:
    log = open(f'flipmix_{xc}_{variant}.log', 'w')
    t0 = time.perf_counter()
    rec = {'variant': variant, 'xc': xc}
    if variant in ('flipmix', 'flipmix+newton'):
        res = detect_and_mix(make(log))
        if res is None:
            rec['note'] = 'no oscillation'
            mf = None
        else:
            dm, cyc, sp, p = res
            rec.update(detect_cycle=cyc, spin=sp, mixed_parity=float(p))
            mf = make(log)
            if variant.endswith('newton'):
                mf = mf.newton()
            mf.kernel(dm)
    elif variant.startswith('sympi_rot'):       # sym pi state, then rotate the hole pair by the angle given
        theta = numpy.radians(float(variant[9:]))
        mol_s = mol.copy(); mol_s.symmetry = True; mol_s.build(False, False)
        mfs = dft.UKS(mol_s, xc=xc); mfs.irrep_nelec = {"A'": (41, 41), 'A"': (10, 9)}
        mfs.verbose, mfs.stdout = 4, log
        mfs.kernel()
        rec['cycles_sym'] = mfs.cycles
        cb, ob, eb = mfs.mo_coeff[1], mfs.mo_occ[1], mfs.mo_energy[1]
        orbsym = numpy.asarray(cb.orbsym)
        ir_ap = mol_s.irrep_id[mol_s.irrep_name.index("A'")]
        i = numpy.where((ob > 0) & (orbsym == ir_ap))[0][-1]
        a = numpy.where((ob == 0) & (orbsym != ir_ap))[0][0]
        c = numpy.array(cb)
        ci, ca = c[:, i].copy(), c[:, a].copy()
        c[:, i] = numpy.cos(theta) * ci + numpy.sin(theta) * ca
        c[:, a] = -numpy.sin(theta) * ci + numpy.cos(theta) * ca
        dm = numpy.asarray([mfs.make_rdm1()[0], (c[:, ob > 0]) @ c[:, ob > 0].T])
        mf = make(log)
        mf.kernel(dm)
    else:
        raise KeyError(variant)
    if mf is not None:
        rec.update(cycles=getattr(mf, 'cycles', None), converged=bool(mf.converged))
        rec.update(describe(mf))
    rec['time'] = time.perf_counter() - t0
    log.flush()
    nb = 0
    for line in open(log.name):
        if line.startswith(('init E=', 'cycle=', 'Extra cycle')):
            nb += 1
        elif 'macro X' in line:
            nb += int(line.split()[-2]); rec['soscf_macro'] = int(line.split()[3])
    rec['fock_builds'] = nb
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(' '.join(f'{k} {v:.3e}' if isinstance(v, float) and k in ('de', 'beta_gap') else
                   f'{k} {v:.10f}' if k == 'e_tot' else f'{k} {v:.1f}' if k == 'time' else f'{k} {v}'
                   for k, v in rec.items()), flush=True)
