'''wB97M-V/def2-TZVPD at grids (99,590)/(50,194) with and without grid_response: H[Cl z, Cl z] of HCl, and the
harmonic frequencies at the RIJCOSX saddle point of Cl- + CH3Cl in SMD water (ts_rijcosx.py)'''
import time, numpy
from pyscf import gto, dft, grad, hessian
from pyscf.hessian import thermo
def mk(mol, smd=False):
    mf = dft.RKS(mol, xc='wb97m-v'); mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    if smd:
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    mf.conv_tol = 1e-11
    return mf
mol = gto.M(atom='H 0 0 0; Cl 0 0 1.28', basis='def2-tzvpd', verbose=0)
mf = mk(mol); mf.kernel()
for gr in (False, True):
    h = mf.Hessian(); h.grid_response = gr
    t = time.time(); e = h.kernel()
    print(f'HCl grid_response={gr}: H[Cl z, Cl z] {e[1,1,2,2]:.6f} (energy second difference 0.329178), '
          f'translational sum {e[1,:,2,2].sum():+.1e}, {time.time()-t:.1f} s', flush=True)
mol = gto.M(atom=__file__.rsplit('/', 1)[0] + '/sn2_ts_rijcosx.xyz', basis='def2-tzvpd', charge=-1, verbose=0)
mf = mk(mol, True); mf.two_level = True; mf.kernel()
for gr in (False, True):
    h = mf.Hessian(); h.grid_response = gr
    t = time.time(); e = h.kernel()
    w = thermo.harmonic_analysis(mol, e, imaginary_freq=False)['freq_wavenumber']
    print(f'SN2 saddle point, SMD water, grid_response={gr}: ' + ' '.join(f'{x:.1f}' for x in w) +
          f'; {time.time()-t:.0f} s', flush=True)
print('RIJCOSX FD Hessian there:  -546.3 197.2 197.8 217.0 977.3 977.7 1108.8 1369.1 1370.4 3194.2 3393.5 3393.9')
