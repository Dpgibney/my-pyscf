"""1 thread: SCF-like nr_rks (MO-tagged dm, default level-3 grid) on caffeine/def2-SVP, split by phase."""
import sys, time, cProfile, pstats, io
import numpy as np
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft, lib
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
for xc in sys.argv[1:] or ['B3LYP']:
    mf = dft.RKS(mol, xc=xc)
    mf.grids.build(with_non0tab=True)
    mo_e, mo = mf.eig(mf.get_hcore(), mf.get_ovlp())
    mo_occ = mf.get_occ(mo_e, mo)
    dm = lib.tag_array(mf.make_rdm1(mo, mo_occ), mo_coeff=mo, mo_occ=mo_occ)
    ni = mf._numint
    ni.nr_rks(mol, mf.grids, xc, dm)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(2):
        ni.nr_rks(mol, mf.grids, xc, dm)
    pr.disable()
    t = (time.perf_counter() - t0) / 2
    st = pstats.Stats(pr)
    tt = {}
    for (fn, line, name), (cc, nc, tottime, cum, callers) in st.stats.items():
        tt[name] = tt.get(name, 0) + tottime / 2
    print(f'{xc}: ngrids {mf.grids.weights.size}; 1-thread nr_rks {t:.3f} s per call; dot {tt.get("dot",0):.3f} '
          f'_rho_kernel {tt.get("_rho_kernel",0):.3f} _eval_xc {tt.get("_eval_xc",0):.3f}', flush=True)
