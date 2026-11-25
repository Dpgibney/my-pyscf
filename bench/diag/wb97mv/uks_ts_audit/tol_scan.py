'''The Hessian against the screening threshold of its exact two-electron derivative integrals (JK_ENERGY_TOL of
hess_jk_energy, MAKE_H1_TOL of make_h1's int2e_ip1 J/K): wB97M-V/def2-TZVP, coarse XC grids, gas phase,
CPHF to 1e-8. Differences from the default (1e-14).
Usage: tol_scan.py XYZ SPIN [tols]'''
import sys, time, numpy
from pyscf import gto, dft, lib, hessian  # noqa
from pyscf.hessian import rhf as rhf_hess, thermo
mol = gto.M(atom=sys.argv[1], basis='def2-tzvp', spin=int(sys.argv[2]), verbose=0)
tols = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1e-14, 1e-12, 1e-11, 1e-10, 1e-9]
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc='wb97m-v')
mf.grids.atom_grid = (50, 194); mf.nlcgrids.atom_grid = (30, 110)
mf.conv_tol = 1e-11
mf.kernel()
print(f'{mol.natm} atoms, {mol.nao} AOs, E = {mf.e_tot:.10f}, {lib.num_threads()} threads', flush=True)
ref = fref = None
for tol in tols:
    rhf_hess.JK_ENERGY_TOL = tol
    rhf_hess.MAKE_H1_TOL = tol
    hobj = mf.Hessian()
    hobj.cphf_residual_tol = 1e-8
    stats = []
    t = time.perf_counter()
    h = hobj.kernel()
    dt = time.perf_counter() - t
    f = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
    if ref is None:
        ref, fref = h, f
        print(f'tol {tol:.0e}: Hessian {dt:6.1f} s (reference)', flush=True)
    else:
        print(f'tol {tol:.0e}: Hessian {dt:6.1f} s  max |dH| {abs(h - ref).max():.1e}  max |dfreq| {abs(f - fref).max():.4f} cm-1', flush=True)
