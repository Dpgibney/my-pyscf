'''How much the grid response moves the harmonic frequencies: wB97M-V/def2-TZVPD, grids (99,590)/(50,194), geometries
optimized with geomeTRIC (gradient with grid_response) first, the Hessian with and without grid_response'''
import time, numpy
from pyscf import gto, dft, grad, hessian
from pyscf.hessian import thermo
from pyscf.geomopt.geometric_solver import optimize
MOLS = {
    'water': 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
    'methanol': 'C -0.047 0.665 0; O -0.047 -0.756 0; H -1.087 0.983 0; H 0.432 1.069 0.892; H 0.432 1.069 -0.892; H 0.858 -1.075 0',
    'H2S': 'S 0 0 0.103; H 0 0.966 -0.825; H 0 -0.966 -0.825',
    'CH3Cl': 'C 0 0 0; Cl 0 0 1.78; H 1.03 0 -0.36; H -0.515 0.892 -0.36; H -0.515 -0.892 -0.36',
    'PH3': 'P 0 0 0.12; H 1.19 0 -0.6; H -0.595 1.03 -0.6; H -0.595 -1.03 -0.6',
}
def mk(mol):
    mf = dft.RKS(mol, xc='wb97m-v'); mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-11
    return mf
for name, atom in MOLS.items():
    t = time.time()
    mol = gto.M(atom=atom, basis='def2-tzvpd', verbose=0)
    g = mk(mol).Gradients(); g.grid_response = True
    mol = optimize(g, convergence_gmax=1e-5, convergence_grms=5e-6, convergence_drms=1e-5, convergence_dmax=2e-5)
    mf = mk(mol); mf.kernel()
    w = {}
    for gr in (False, True):
        h = mf.Hessian(); h.grid_response = gr
        w[gr] = thermo.harmonic_analysis(mol, h.kernel(), imaginary_freq=False)['freq_wavenumber']
    d = w[False] - w[True]
    print(f'{name:9s} with grid response: ' + ' '.join(f'{x:.1f}' for x in w[True]) + f'; without - with: '
          + ' '.join(f'{x:+.1f}' for x in d) + f' cm-1 (max {abs(d).max():.1f}); {time.time()-t:.0f} s', flush=True)
