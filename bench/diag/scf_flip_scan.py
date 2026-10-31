"""Does the flip-pair trigger (a clean swap of one occupied orbital after cycle 4, and the same orbital swapping
back later) fire on open-shell molecules where CDIIS behaves? Plain CDIIS with the detector as a callback that
only records; prints the default run's cycles and the cycle the detector would have fired at.
Usage: scf_flip_scan.py XC mol [mol ...]"""
import sys, os, glob, time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE, BENZENE
from pyscf import gto, dft

FIRST, SWAP = 4, 0.1
geom = os.path.join(here, '..', 'geom')
mols = {}
for f in glob.glob(os.path.join(geom, '*.xyz')) + glob.glob(os.path.join(geom, 'hard', '*.xyz')):
    lines = open(f).read().splitlines()
    c, s = map(int, lines[1].split()[:2])
    mols[os.path.basename(f)[:-4]] = ('\n'.join(lines[2:]), c, s)
mols['caffeine+'] = (CAFFEINE, 1, 1)
mols['benzene+'] = (BENZENE, 1, 1)

xc = sys.argv[1]
for name in sys.argv[2:]:
    atom, charge, spin = mols[name]
    mol = gto.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)
    s1e = mol.intor_symmetric('int1e_ovlp')
    mf = dft.UKS(mol, xc=xc)
    mf.max_cycle = 100
    hist = {0: [], 1: []}
    last = [None, None]
    fired = []
    swaps = []
    def callback(envs):
        c, occ = envs['mo_coeff'], envs['mo_occ']
        cyc = envs['cycle'] + 1
        for sp in range(2):
            cur = c[sp][:, occ[sp] > 0]
            if last[sp] is not None and cyc >= FIRST:
                u, sv, vt = numpy.linalg.svd(last[sp].T @ s1e @ cur)
                if sv[-1] < SWAP:
                    out_, in_ = last[sp] @ u[:, -1], cur @ vt[-1]
                    swaps.append((cyc, sp))
                    if not fired and any(abs(in_ @ s1e @ o1) > 0.9 for _, o1 in hist[sp]):
                        fired.append((cyc, sp))
                    hist[sp].append((cyc, out_))
            last[sp] = cur
    mf.callback = callback
    t0 = time.perf_counter()
    mf.kernel()
    print(f'{name:18s} {xc:6s} cycles {mf.cycles:3d} conv {mf.converged!s:5s} swaps(cycle,spin) {swaps[:6]} '
          f'trigger {fired[0] if fired else None}  t {time.perf_counter() - t0:.1f} s', flush=True)
