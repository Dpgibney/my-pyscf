'''The solvent's Hessian terms and Fock derivative for a molecule, each method, saved to an npz: to compare two trees.
Usage: solv_cmp.py XYZ OUT.npz'''
import sys, time, numpy
from pyscf import gto, dft, lib
from pyscf.solvent import pcm
from pyscf.solvent.hessian import pcm as pcm_hess
from pyscf.solvent.grad import pcm as pcm_grad
mol = gto.M(atom=sys.argv[1], basis='def2-svp', spin=1, verbose=0)
out = {}
dm = None
for method in ('C-PCM', 'IEF-PCM', 'SS(V)PE', 'SMD'):
    mf = dft.UKS(mol, xc='pbe').density_fit()
    if method == 'SMD':
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    else:
        mf = mf.PCM(); mf.with_solvent.method = method; mf.with_solvent.eps = 78.3553
    mf.grids.atom_grid = (50, 194)
    mf.conv_tol = 1e-11
    mf.kernel(dm0=dm)
    dm = mf.make_rdm1()
    d = dm[0] + dm[1]
    s = mf.with_solvent
    t = time.perf_counter()
    out[method + ' grad'] = s.grad(d)
    out[method + ' vmat'] = pcm_hess.analytical_grad_vmat(s, d)
    out[method + ' nuc'] = pcm_hess.analytical_hess_nuc(s, d)
    out[method + ' qv'] = pcm_hess.analytical_hess_qv(s, d)
    out[method + ' solver'] = pcm_hess.analytical_hess_solver(s, d)
    out[method + ' e'] = mf.e_tot
    print(f'{method}: E = {mf.e_tot:.10f}, {s.surface["grid_coords"].shape[0]} surface points, {time.perf_counter() - t:.1f} s, '
          f'{lib.num_threads()} threads', flush=True)
numpy.savez(sys.argv[2], **out)
