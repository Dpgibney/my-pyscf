"""SCF cycles per initial guess and DIIS variant (It21). For each molecule of bench/geom and the bench set, runs
the SCF at the default thresholds (conv_tol 1e-9) for every (guess, diis) and appends one JSON line per run to
OUT: cycles, converged, energy, the energy of the guess, wall time, and the trace (E, |g|) per cycle.
Molecules share their grids and in-memory ERIs between runs.
Usage: scf_guess_study.py OUT [--xc B3LYP] [--basis def2-svp] [--guesses minao,atom,...] [--diis cdiis,...]
                          [--mols name,...]"""
import argparse
import glob
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib, scf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--xc', default='B3LYP')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--guesses', default='minao,atom,huckel,mod_huckel,sap,vsap')
ap.add_argument('--diis', default='cdiis')
ap.add_argument('--mols', default=None)
ap.add_argument('--max-cycle', type=int, default=100)
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
if args.mols:
    mols = {k: mols[k] for k in args.mols.split(',')}


def diis_class(name):
    if name == 'cdiis':
        return scf.diis.CDIIS
    if name == 'adiis':
        return scf.diis.ADIIS
    if name == 'ediis':
        return scf.diis.EDIIS
    return getattr(scf.diis, name)  # classes added in It21


for name, (atom, charge, spin) in mols.items():
    mol = gto.M(atom=atom, basis=args.basis, charge=charge, spin=spin, verbose=0)
    grids = eri = None
    for diis in args.diis.split(','):
        for guess in args.guesses.split(','):
            mf = (dft.RKS if spin == 0 else dft.UKS)(mol, xc=args.xc)
            mf.max_cycle = args.max_cycle
            mf.init_guess = guess
            mf.DIIS = diis_class(diis)
            if grids is not None:
                mf.grids = grids
            if eri is not None:
                mf._eri = eri
            trace = []

            def callback(envs):
                if envs['cycle'] == 0:
                    trace.append((float(envs['last_hf_e']), None))
                trace.append((float(envs['e_tot']), float(envs['norm_gorb'])))
            mf.callback = callback
            rec = {'mol': name, 'natm': mol.natm, 'nao': mol.nao, 'xc': args.xc, 'basis': args.basis,
                   'spin': spin, 'guess': guess, 'diis': diis}
            t0 = time.perf_counter()
            try:
                mf.kernel()
                rec.update(cycles=mf.cycles, converged=bool(mf.converged), e_tot=float(mf.e_tot),
                           e_guess=trace[0][0] if trace else None, trace=trace)
                if spin:
                    rec['s2'] = float(mf.spin_square()[0])
            except Exception as err:  # a guess that does not support this case
                rec.update(error=repr(err))
            rec['time'] = time.perf_counter() - t0
            grids = mf.grids
            if getattr(mf, '_eri', None) is not None:
                eri = mf._eri
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{name:18s} {diis:8s} {guess:10s} cycles {rec.get('cycles')} conv {rec.get('converged')} "
                  f"E {rec.get('e_tot', float('nan')):.10f} t {rec['time']:.1f} s {rec.get('error', '')}",
                  flush=True)
