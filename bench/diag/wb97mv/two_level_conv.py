'''mf.two_level: the convergence of the cheap model's own SCF (mf.two_level_conv_factor x conv_tol) and the second-order
test of the references (two_level.SECOND_ORDER_TEST), caffeine wB97M-V/def2-TZVPD, grids (99,590)/(50,194), exact J and K and
RIJCOSX(aug_etb). A line per run: time, Fock builds of the model and of the cheap one, the energy.
Usage: two_level_conv.py exact|rijcosx [--smd] TOL[,nosecond] ...'''
import sys, time, io, re
from pyscf import gto, dft, df, lib
from pyscf.scf import two_level
from molecules import CAFFEINE
T0 = time.perf_counter()
args = sys.argv[1:]
model = args.pop(0)
smd = '--smd' in args
args = [a for a in args if a != '--smd']
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
e0 = None
for spec in args:
    tol = float(spec.split(',')[0])
    two_level.SECOND_ORDER_TEST = 'nosecond' not in spec
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    if model == 'rijcosx':
        mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
    if smd:
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    mf.two_level = True; mf.two_level_conv_factor = tol / mf.conv_tol
    out = io.StringIO(); mf.stdout = out; mf.verbose = 4
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    log = out.getvalue()
    refs = re.findall(r'^reference .*$', log, re.M)
    summary = re.findall(r'^two-level SCF: .*$', log, re.M)[-1]
    if e0 is None:
        e0 = e
    print(f'[{time.perf_counter() - T0:5.0f} s] {model}{" SMD" if smd else ""}, cheap SCF to {tol:g}, second-order test '
          f'{two_level.SECOND_ORDER_TEST}: {dt:.1f} s, conv {mf.converged}, E {e:.10f} ({e - e0:+.1e}); {summary}', flush=True)
    for r in refs:
        print('        ' + r, flush=True)
