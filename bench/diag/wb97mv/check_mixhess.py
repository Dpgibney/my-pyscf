import time, numpy
from pyscf import gto, dft
from pyscf.sgx import sgx_jk
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svpd', verbose=0)
for xc in ('wb97m-v', 'cam-b3lyp'):
    mf = dft.RKS(mol, xc=xc); mf.conv_tol = 1e-11
    if mf.do_nlc(): mf.nlcgrids.level = 0
    mf.kernel()
    out = []
    for flag in (False, True):
        sgx_jk.MIXED_RANGE = flag
        h = mf.Hessian(); h.sgx_response = True; h.cphf_residual_tol = 1e-6
        t = time.perf_counter(); out.append((h.kernel(), time.perf_counter() - t))
    print(f'{xc}: Hessian with sgx_response, two passes vs one: max|dH| {abs(out[0][0] - out[1][0]).max():.1e} (max|H| {abs(out[0][0]).max():.2f}); t {out[0][1]:.1f} {out[1][1]:.1f}', flush=True)
