"""Time the XC part of the RKS gradient (grad.rks.get_vxc) and NumInt.nr_rks on a large molecule.
Usage: grad_vxc_time.py [c20|c20tz] [XC]   prints an @@ JSON line with the results for comparisons
"""
import json
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import n_alkane
from pyscf import gto, dft
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint
args = [a for a in sys.argv[1:] if not a.startswith('--')]
if '--nofuse' in sys.argv:      # commit B off: unfused sparse kernels
    numint._ao_comps_contiguous = lambda ao: False
if '--noroute' in sys.argv:     # commit C off: _dot_ao_ao/_dot_ao_dm without the gather kernels
    numint._screened_gemm = lambda *a: False

which = args[0] if len(args) > 0 else 'c20'
xc = args[1] if len(args) > 1 else 'B3LYP'
basis = {'c20': 'def2-svp', 'c20tz': 'def2-tzvp'}[which]
mol = gto.M(atom=n_alkane(20), basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))   # MO-tagged, as in SCF and gradients
ni = mf._numint

def best(fn, n=2):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), r

t_vxc, (nelec, exc, vmat) = best(lambda: ni.nr_rks(mol, mf.grids, xc, dm))
t_grad, (_, gvxc) = best(lambda: rks_grad.get_vxc(ni, mol, mf.grids, xc, dm))
print(f'{which} {xc}: nr_rks {t_vxc:.3f} s, grad get_vxc {t_grad:.3f} s')
print('@@' + json.dumps({'exc': exc, 'vxc_sum': float(abs(vmat).sum()),
                         'gvxc_sum': float(abs(gvxc).sum()), 'gvxc_00': float(gvxc[0, 0, 0]),
                         't_nr_rks': t_vxc, 't_grad_vxc': t_grad}))
