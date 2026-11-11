# prototype: the cheap model of mf.two_level with SGX-K on grid level 0 throughout (no switch to level 1)
import sys, time, io, re
from pyscf import gto, dft, df, lib
from pyscf.scf import two_level, hf
from molecules import CAFFEINE
T0 = time.perf_counter()
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
two_level.CHEAP_CONV_TOL = 1e-5
def run(model, level0):
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    if model == 'rijcosx':
        mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
    mf.two_level = True
    if level0:
        if model == 'rijcosx':
            c = dft.RKS(mol, xc='wb97m-v')
            c.grids = hf._pre_scf_grids(mf.grids, mol, mf.two_level_grids_level)
            c.nlcgrids = hf._pre_scf_grids(mf.nlcgrids, mol, mf.two_level_nlcgrids_level)
            cheap = c.RIJCOSX(auxbasis=df.aug_etb(mol))
            cheap.conv_check = False
        else:
            cheap = two_level.cheap_scf(mf)
        cheap.with_df.grids_level_f = 0
        if model == 'rijcosx':
            mf.with_df.grids_level_i = mf.with_df.grids_level_f   # the model's own builds on its final level
        cheap.with_df.grids_level_energy = None
        mf.two_level_scf = cheap
    out = io.StringIO(); mf.stdout = out; mf.verbose = 4
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    log = out.getvalue()
    print(f'[{time.perf_counter() - T0:5.0f} s] {model}, cheap SGX-K {"level 0 only" if level0 else "levels 0 then 1"}: {dt:.1f} s, conv {mf.converged}, '
          f'E {e:.10f}; ' + re.findall(r'^two-level SCF: .*$', log, re.M)[-1], flush=True)
    for r in re.findall(r'^reference .*$', log, re.M):
        print('        ' + r, flush=True)
for spec in sys.argv[1:]:
    model, lv = spec.split(',')
    run(model, lv == '0')
