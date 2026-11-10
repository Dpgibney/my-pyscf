'''UKS gradients with the XC and VV10 terms contracted on the grid (grad.uks._get_vxc_de) against the matrices
(grad.rks.XC_DM = False): direct, density-fitted and RIJCOSX, a radical cation and a triplet.'''
import time, numpy
from pyscf import gto, dft
from pyscf.grad import rks as rks_grad
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
mols = [('cation doublet', gto.M(atom=atom, basis='def2-svpd', charge=1, spin=1, verbose=0)),
        ('triplet', gto.M(atom='O 0 0 0; O 0 0.1 1.22; H 3 0 0; H 3 0.1 0.75', basis='def2-svpd', spin=2, verbose=0))]
worst = 0
for mname, mol in mols:
    for xc in ('lda,vwn', 'pbe', 'r2scan', 'b3lyp', 'wb97m-v'):
        for kind in ('direct', 'df', 'rijcosx'):
            if kind != 'direct' and xc in ('lda,vwn', 'r2scan'):
                continue
            mf = dft.UKS(mol, xc=xc)
            if xc == 'wb97m-v':
                mf.nlcgrids.level = 1
            if kind == 'df':
                mf = mf.density_fit()
            elif kind == 'rijcosx':
                mf = mf.RIJCOSX()
            mf.conv_tol = 1e-11
            mf.kernel()
            out = []
            for flag in (True, False):
                rks_grad.XC_DM = flag
                t = time.perf_counter(); g = mf.Gradients().kernel(); out.append((g, time.perf_counter() - t))
            rks_grad.XC_DM = True
            d = abs(out[0][0] - out[1][0]).max(); worst = max(worst, d)
            print(f'{mname:15s} {xc:8s} {kind:8s} max|g(grid) - g(matrices)| {d:.1e}  ({out[0][1]:.2f} s, {out[1][1]:.2f} s)', flush=True)
print('largest difference %.1e' % worst)
