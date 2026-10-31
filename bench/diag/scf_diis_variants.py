"""SCF cycles of DIIS variants (It21): subspace size, start cycle, and ADIIS+CDIIS blends, minao guess, default
thresholds. Usage: scf_diis_variants.py OUT --mols a,b,... [--xc B3LYP] [--basis def2-svp] [--variants v1,...]"""
import argparse
import glob
import json
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, scf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--mols', required=True)
ap.add_argument('--xc', default='B3LYP')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--guess', default='minao')
ap.add_argument('--variants', default=None)
args = ap.parse_args()

mols = {}
for f in sorted(glob.glob(os.path.join(here, '..', 'geom', '*.xyz')) +
                glob.glob(os.path.join(here, '..', 'geom', 'hard', '*.xyz'))):
    lines = open(f).read().splitlines()
    charge, spin = map(int, lines[1].split()[:2])
    mols[os.path.basename(f)[:-4]] = ('\n'.join(lines[2:]), charge, spin)
mols['benzene'] = (BENZENE, 0, 0)
mols['caffeine'] = (CAFFEINE, 0, 0)
mols['caffeine+'] = (CAFFEINE, 1, 1)
mols['c20h42'] = (n_alkane(20), 0, 0)
# water with both O-H bonds stretched to twice their length
mols['water_2x'] = ('O 0 0 0.2346; H 0 1.5144 -0.9384; H 0 -1.5144 -0.9384', 0, 0)


def hybrid(**kw):
    return type('Hybrid', (scf.diis.ADIIS_CDIIS,), kw)


# name: (DIIS class, diis_space, diis_start_cycle[, init_guess])
VARIANTS = {
    'pre_cdiis': (scf.diis.CDIIS, 8, 1, 'pre_scf'),
    'pre_hyb': (scf.diis.ADIIS_CDIIS, 8, 1, 'pre_scf'),
    'cdiis': (scf.diis.CDIIS, 8, 1),
    'cdiis_s12': (scf.diis.CDIIS, 12, 1),
    'cdiis_s15': (scf.diis.CDIIS, 15, 1),
    'cdiis_s20': (scf.diis.CDIIS, 20, 1),
    'cdiis_c0': (scf.diis.CDIIS, 8, 0),
    'cdiis_s12_c0': (scf.diis.CDIIS, 12, 0),
    'hyb_lin': (hybrid(blend='linear'), 8, 1),
    'hyb_lin_s12': (hybrid(blend='linear'), 12, 1),
    'hyb_lin_c0': (hybrid(blend='linear'), 8, 0),
    'hyb_lin_s12_c0': (hybrid(blend='linear'), 12, 0),
    'hyb_log': (hybrid(blend='log'), 8, 1),
    'hyb_rms': (hybrid(err='rms'), 8, 1),
    'hyb_rms_s12': (hybrid(err='rms'), 12, 1),
    'hyb_log_s12_c0': (hybrid(blend='log'), 12, 0),
}
names = args.variants.split(',') if args.variants else list(VARIANTS)

for name in args.mols.split(','):
    atom, charge, spin = mols[name]
    mol = gto.M(atom=atom, basis=args.basis, charge=charge, spin=spin, verbose=0)
    grids = eri = None
    out = []
    for v in names:
        cls, space, start = VARIANTS[v][:3]
        guess = VARIANTS[v][3] if len(VARIANTS[v]) > 3 else args.guess
        if args.xc.upper() == 'HF':
            mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
        else:
            mf = (dft.RKS if spin == 0 else dft.UKS)(mol, xc=args.xc)
        mf.init_guess = guess
        mf.DIIS, mf.diis_space, mf.diis_start_cycle = cls, space, start
        mf.max_cycle = 100
        if grids is not None and guess != 'pre_scf':
            mf.grids = grids
        if eri is not None:
            mf._eri = eri
        t0 = time.perf_counter()
        mf.kernel()
        rec = {'mol': name, 'variant': v, 'xc': args.xc, 'basis': args.basis, 'guess': guess,
               'cycles': mf.cycles, 'converged': bool(mf.converged), 'e_tot': float(mf.e_tot),
               'time': time.perf_counter() - t0}
        grids = getattr(mf, 'grids', None)
        if getattr(mf, '_eri', None) is not None:
            eri = mf._eri
        with open(args.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        out.append(f"{rec['cycles']}{'' if rec['converged'] else '!'}")
    print(f'{name:18s} ' + ' '.join(f'{x:>5s}' for x in out), flush=True)
print(' ' * 19 + ' '.join(f'{v}' for v in names))
