"""Translational invariance of the SGX gradient (pyscf/sgx/test/test_dm_screening.py setup)."""
import numpy
from pyscf import gto, scf
from pyscf.sgx.sgx import sgx_fit

ATOM = [["O", (0., 0., 0.)], [1, (0., -0.757, 0.587)], [1, (0., 0.757, 0.587)]]
# the test's molecule
mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]
import pyscf.sgx.test.test_dm_screening as t
mol = gto.M(atom=t.ATOM, basis='6-31g', verbose=0)
mol2 = gto.M(atom=t.ATOM, basis='6-31g', verbose=0, spin=1, charge=1)
for spinpol in (False, True):
    for fit_ovlp, optk in ((False, False), (False, True), (True, True)):
        mf = sgx_fit(scf.UHF(mol2) if spinpol else scf.RHF(mol))
        t._set_df_args(mf, fit_ovlp, optk)
        mf.kernel()
        g = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True).kernel()
        print(f'{"UHF" if spinpol else "RHF"} fit_ovlp={fit_ovlp!s:5s} optk={optk!s:5s} '
              f'e={mf.e_tot:.12f} |sum g|={numpy.abs(g.sum(axis=0)).sum():.2e}')
