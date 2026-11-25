'''UKS Hessians of the CH2Cl radical in several configurations, saved to npz (run with the frozen tree for the
references and with the working tree for the comparison)'''
import sys, time, numpy
import pyscf
from pyscf import gto, dft, lib, hessian  # noqa
out = sys.argv[1]
cases = sys.argv[2].split(',') if len(sys.argv) > 2 else None
import os
RKS = os.environ.get('HESS_REF_RKS') == '1'
if os.environ.get('HESS_REF_DD') == '0':
    # the screening of the energy-type two-electron derivative integrals as before the two-density bound
    from pyscf.hessian import rhf as _rhf_hess
    from pyscf.grad import rhf as _rhf_grad
    _rhf_hess.JK_ENERGY_DD_SCREEN = False
    _rhf_hess.JK_ENERGY_TOL = 1e-14
    _rhf_hess.MAKE_H1_TOL = 1e-14
    _rhf_grad.JK_ENERGY_DD_SCREEN = False
if RKS:
    mol = gto.M(atom='O 0 0 0; H 0.1 0.93 0.25; Cl 1.55 -0.5 0.1; C -1.2 -0.6 0.2; H -1.9 0.2 0.5; H -1.5 -1.3 -0.6; H -1.0 -1.2 1.1', basis='def2-svp', charge=1, verbose=0)
else:
    mol = gto.M(atom='C 0 0 0; H 0 0.93 0.45; H 0.1 -0.93 0.5; Cl 0 0.1 -1.75', basis='def2-svp', spin=1, verbose=0)
res = {}
CASES = {
    # name: (xc, solvent, grid_response, sgx_response, auxbasis, two_level)
    'wb97mv':            ('wb97m-v', False, False, False, None, False),
    'wb97mv_gr':         ('wb97m-v', False, True, False, None, False),
    'wb97mv_gr_sgx':     ('wb97m-v', False, True, True, None, False),
    'wb97mv_smd_gr_sgx': ('wb97m-v', True, True, True, None, True),
    'wb97mv_smd_gr_sgx_aux': ('wb97m-v', True, True, True, 'aug_etb', True),
    'b3lyp_gr_sgx':      ('b3lyp', False, True, True, None, False),
    'pbe_gr':            ('pbe', False, True, False, None, False),
    'm06l_smd':          ('m06l', True, False, False, None, False),
    'svwn_gr':           ('lda,vwn', False, True, False, None, False),
}
for name, (xc, solvent, gr, sgx, aux, two) in CASES.items():
    if cases and name not in cases:
        continue
    mf = (dft.RKS if RKS else dft.UKS)(mol, xc=xc)
    mf.grids.atom_grid = (50, 194)
    mf.nlcgrids.atom_grid = (30, 110)
    if solvent:
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    mf.conv_tol = 1e-11
    if two:
        mf.two_level = True
        mf.set_second_grids(2); mf.set_second_nlcgrids(1)
    mf.kernel()
    hobj = mf.Hessian()
    hobj.grid_response = gr
    hobj.sgx_response = sgx
    hobj.sgx_response_auxbasis = aux
    hobj.cphf_residual_tol = float(os.environ.get('HESS_REF_TOL', 1e-7))
    hobj.verbose = int(os.environ.get('HESS_REF_VERBOSE', 0))
    t = time.perf_counter()
    h = hobj.kernel()
    res[name] = h
    res[name + '_e'] = mf.e_tot
    print(f'{name:24s} E {mf.e_tot:.10f}  Hessian {time.perf_counter() - t:6.1f} s  max |H| {abs(h).max():.4f}', flush=True)
numpy.savez(out, **res)
print(pyscf.__file__)
