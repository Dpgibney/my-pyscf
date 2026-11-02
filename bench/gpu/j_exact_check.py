"""J matrices from GPU4PySCF builds (jk_at_dm.py --out) against the exact J of PySCF on the CPU, without screening
(scf.hf.get_jk with vhfopt=None), at the same density. Kernel changes that alter each quartet's arithmetic, not only
the order of the sums, move J by more than reordering noise; this shows whether a build is closer to or further from
the exact integrals than the reference build. The exact J is cached in EXACT.npy (computed if missing).
Usage: j_exact_check.py --dm DM.npy --basis def2-tzvp --exact EXACT.npy LABEL=J.npy [LABEL=J.npy ...]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
from pyscf import gto, scf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--exact', required=True)
ap.add_argument('builds', nargs='+')
args = ap.parse_args()

dm = numpy.load(args.dm)
if os.path.exists(args.exact):
    jx = numpy.load(args.exact)
else:
    mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
    t0 = time.time()
    jx = scf.hf.get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=False)[0]
    numpy.save(args.exact, jx)
    print(f'exact J (no screening) in {time.time() - t0:.0f} s')
print(f'max |J| {abs(jx).max():.2f}')
print(f'{"build":24s} {"max err":>9s} {"rms err":>9s} {"E_J err (Eh)":>13s} {"|err| > 1e-12":>14s}')
for b in args.builds:
    label, path = b.split('=')
    j = numpy.load(path)
    if j.ndim == 3:   # several densities (--ndm): the first is the density itself
        j = j[0]
    e = j - jx
    print(f'{label:24s} {abs(e).max():9.2e} {numpy.sqrt((e ** 2).mean()):9.2e} '
          f'{0.5 * numpy.einsum("ij,ji", dm, e):13.2e} {(abs(e) > 1e-12).sum():14d}')
