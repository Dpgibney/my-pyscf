'''Wall times of the grid-response parts of the wB97M-V Hessian, (99,590)/(50,194), in one process: the XC energy
terms, the VV10 energy with and without, make_h1 with and without. Usage: hess_grid_response_parts.py XYZ BASIS'''
import sys, time
from pyscf import gto, dft, grad, hessian, lib
from pyscf.hessian import rks as rks_hess
mol = gto.M(atom=sys.argv[1], basis=sys.argv[2], verbose=0)
mf = dft.RKS(mol, xc='wb97m-v'); mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf.two_level = True; mf.kernel()
print(f'{mol.natm} atoms, {mol.nao} AOs, max_memory {mf.max_memory}, XC grid {mf.grids.weights.size}, VV10 grid {mf.nlcgrids.weights.size}', flush=True)
mo, occ = mf.mo_coeff, mf.mo_occ
for rep in range(2):
    for gr in (False, True):
        h = mf.Hessian(); h.grid_response = gr
        t = time.time(); e = rks_hess._get_exc_deriv2_grid_response(h, mo, occ, 4000) if gr else None; t_xc = time.time() - t
        t = time.time(); rks_hess._get_enlc_deriv2(h, mo, occ, 4000); t_nlc = time.time() - t
        t = time.time(); rks_hess._make_h1_split(h, mo, occ); t_h1 = time.time() - t
        print(f'rep {rep} grid_response={gr}: XC energy grid terms {t_xc:.1f} s, VV10 energy {t_nlc:.1f} s, make_h1 {t_h1:.1f} s', flush=True)
