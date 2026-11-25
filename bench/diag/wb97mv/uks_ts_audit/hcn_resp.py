'''HCN <-> HNC saddle point (the geometry of the example's default run, or XYZ): frequencies with the exact CPHF
response, with sgx_response (levels 1 and 0), and with RI-J in the response too. The example's model.'''
import sys, numpy
from pyscf import gto, dft, lib, hessian  # noqa
from pyscf.hessian import thermo
xyz = sys.argv[1]
spin = int(sys.argv[2]) if len(sys.argv) > 2 else 0
mol = gto.M(atom=xyz, basis='def2-tzvp', spin=spin, verbose=0)
mf = (dft.UKS if spin else dft.RKS)(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf.two_level = True
mf.set_second_grids(2); mf.set_second_nlcgrids(1)
mf.kernel()
ref = None
for label, sgx, aux, level, tol in (('exact response', False, None, None, None), ('exact response, residual 1e-7', False, None, None, 1e-7),
                                    ('sgx_response', True, None, None, None), ('sgx_response + RI-J aug_etb', True, 'aug_etb', None, None),
                                    ('sgx_response + RI-J, residual 1e-7', True, 'aug_etb', None, 1e-7),
                                    ('sgx_response level 2 + RI-J', True, 'aug_etb', 2, None)):
    hobj = mf.Hessian()
    hobj.grid_response = True
    hobj.sgx_response = sgx
    hobj.sgx_response_auxbasis = aux
    if level is not None:
        hobj.sgx_response_level = level
    if tol is not None:
        hobj.cphf_residual_tol = tol
    h = hobj.kernel()
    f = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
    if ref is None:
        ref = (h, f)
    print(f'{label:36s} ' + ' '.join(f'{x:9.2f}' for x in f) + f'   max |dH| {abs(h - ref[0]).max():.1e}  max |dfreq| {abs(f - ref[1]).max():.2f}', flush=True)
