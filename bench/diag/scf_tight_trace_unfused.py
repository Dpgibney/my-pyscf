"""Tight SCF on n-C20H42/def2-SVP with per-cycle energy / gradient norm, as in bench validate."""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import n_alkane
from pyscf import gto, dft
if "--unfused" in sys.argv:
    from pyscf.dft import numint as _nm
    _nm._ao_comps_contiguous = lambda ao: False

xc = sys.argv[1] if len(sys.argv) > 1 else 'B3LYP'
maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 30
mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=4, output=None)
mf = dft.RKS(mol, xc=xc)
mf.chkfile = None
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = maxc
mf.kernel()
print(f'@@ converged={mf.converged} cycles={mf.cycles} e_tot={mf.e_tot:.12f}')
