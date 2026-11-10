import numpy
from pyscf import gto, dft
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
def make(cls, mol, xc):
    mf = cls(mol, xc=xc); mf.conv_tol = 1e-12; mf.grids.atom_grid = (99, 590)
    if mf.do_nlc():
        mf.nlcgrids.atom_grid = (50, 194)
    return mf
neutral = gto.M(atom=atom, basis='def2-svp', verbose=0)
for xc in ('b3lyp', 'wb97x-v'):
    mf = make(dft.RKS, neutral, xc); mf.kernel()
    h = mf.Hessian(); h.cphf_residual_tol = 1e-8
    hana = h.kernel()
    coords = neutral.atom_coords()
    worst = 0
    for ia, x in ((0, 2), (3, 0), (5, 1)):
        g = []
        for sign in (1, -1):
            c = coords.copy(); c[ia, x] += sign * 1e-3
            m = neutral.set_geom_(c, unit='Bohr', inplace=False)
            mf1 = make(dft.RKS, m, xc); mf1.kernel(dm0=mf.make_rdm1())
            g.append(mf1.Gradients().kernel())
        worst = max(worst, abs(hana[ia, :, x, :] - (g[0] - g[1]) / 2e-3).max())
    print(f'RKS neutral {xc}: max|H analytical - finite difference of gradients| {worst:.1e}', flush=True)
