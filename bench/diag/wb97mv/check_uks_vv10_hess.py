'''The UKS Hessian with VV10 (the RKS VV10 terms for the total density): a closed-shell molecule as UKS against RKS,
and a radical cation against finite differences of analytical gradients (B3LYP alongside, for the size of the
difference that the missing grid-weight derivatives of the analytical Hessian leave).'''
import numpy
from pyscf import gto, dft
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
def make(cls, mol, xc):
    mf = cls(mol, xc=xc); mf.conv_tol = 1e-12; mf.grids.atom_grid = (99, 590)
    if mf.do_nlc():
        mf.nlcgrids.atom_grid = (50, 194)
    return mf
neutral = gto.M(atom=atom, basis='def2-svp', verbose=0)
out = []
for cls in (dft.RKS, dft.UKS):
    mf = make(cls, neutral, 'wb97m-v'); mf.kernel()
    h = mf.Hessian(); h.cphf_residual_tol = 1e-8
    out.append(h.kernel())
print('closed shell wb97m-v: max|H(UKS) - H(RKS)| %.1e (max|H| %.2f)' % (abs(out[0] - out[1]).max(), abs(out[0]).max()), flush=True)
cation = gto.M(atom=atom, basis='def2-svp', charge=1, spin=1, verbose=0)
for xc in ('b3lyp', 'wb97m-v', 'wb97x-v'):
    mf = make(dft.UKS, cation, xc); mf.kernel()
    h = mf.Hessian(); h.cphf_residual_tol = 1e-8
    hana = h.kernel()
    coords = cation.atom_coords()
    worst = 0
    for ia, x in ((0, 2), (3, 0), (5, 1)):
        g = []
        for sign in (1, -1):
            c = coords.copy(); c[ia, x] += sign * 1e-3
            m = cation.set_geom_(c, unit='Bohr', inplace=False)
            mf1 = make(dft.UKS, m, xc); mf1.kernel(dm0=mf.make_rdm1())
            g.append(mf1.Gradients().kernel())
        hfd = (g[0] - g[1]) / 2e-3
        worst = max(worst, abs(hana[ia, :, x, :] - hfd).max())
    print(f'cation {xc}: max|H analytical - finite difference of gradients| {worst:.1e}', flush=True)
