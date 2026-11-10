import time, numpy
from pyscf import gto, dft, lib
from pyscf.hessian import rks as rks_hess, rhf as rhf_hess
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svpd', verbose=0)
for name, make in [('lda', lambda: dft.RKS(mol, xc='lda,vwn')), ('pbe', lambda: dft.RKS(mol, xc='pbe')),
                   ('b3lyp', lambda: dft.RKS(mol, xc='b3lyp')), ('r2scan', lambda: dft.RKS(mol, xc='r2scan')),
                   ('wb97m-v', lambda: dft.RKS(mol, xc='wb97m-v')), ('wb97m-v PCM', lambda: dft.RKS(mol, xc='wb97m-v').PCM()),
                   ('b3lyp df', lambda: dft.RKS(mol, xc='b3lyp').density_fit())]:
    mf = make(); mf.conv_tol = 1e-11
    if mf.do_nlc(): mf.nlcgrids.level = 0
    mf.kernel()
    out = []
    for flag in (False, True):
        rks_hess.H1_OCC = flag
        h = mf.Hessian(); h.cphf_residual_tol = 1e-7
        used = []
        split0 = rks_hess._make_h1_split
        def spy(*a, **k):
            r = split0(*a, **k); used.append(r[1] is not None and abs(r[1]).max() > 0); return r
        rhf_hess._MAKE_H1_SPLIT[rks_hess.make_h1] = spy
        t = time.perf_counter(); out.append((h.kernel(), time.perf_counter() - t, list(used)))
        rhf_hess._MAKE_H1_SPLIT[rks_hess.make_h1] = split0
    print(f'{name:12s} max|dH| {abs(out[0][0] - out[1][0]).max():.1e} (max|H| {abs(out[0][0]).max():.2f}); t {out[0][1]:.1f} -> {out[1][1]:.1f}; occ form used {out[1][2]}', flush=True)
