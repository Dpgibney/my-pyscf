import time, numpy, sys
from pyscf import gto, scf, dft, lib
from pyscf.grad import rks as rks_grad
from molecules import CAFFEINE
XCS = sys.argv[1].split(';')
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svp', verbose=0)
for xc in XCS:
    for kind in ('direct', 'df', 'rijcosx'):
        mf = dft.RKS(mol, xc=xc)
        if mf.do_nlc(): mf.nlcgrids.level = 1
        if kind == 'df': mf = mf.density_fit()
        if kind == 'rijcosx': mf = mf.RIJCOSX()
        mf.conv_tol = 1e-11
        mf.kernel()
        out = []
        for flag in (False, True):
            rks_grad.XC_DM = flag
            t = time.perf_counter(); g = mf.nuc_grad_method().kernel(); out.append((g, time.perf_counter() - t))
        print(f'{xc:8s} {kind:8s} max|dg| {abs(out[0][0] - out[1][0]).max():.1e}  max|g| {abs(out[0][0]).max():.3f}  t {out[0][1]:.2f} -> {out[1][1]:.2f}', flush=True)
