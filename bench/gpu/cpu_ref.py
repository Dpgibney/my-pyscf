'''The CPU tree's energy, gradient and Hessian of a molecule with the settings of the GPU profile runs (prod_parts.py:
wB97M-V, (99,590)/(50,194), SMD water, conv_tol 1e-10; Hessian with grid_response, exact J/K response, the XC kernel of
the response on SG-1 as GPU4PySCF's cphf_grids), saved to an npz with prod_parts.py's keys.
Usage: cpu_ref.py XYZ SPIN BASIS OUT [--cphf-grids sg1|none|LEVEL]'''
import sys, time, numpy
from pyscf import gto, dft, lib, grad, hessian  # noqa
xyz, spin, basis, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cg = 'sg1'
for a in sys.argv[5:]:
    if a.startswith('--cphf-grids='):
        cg = a.split('=', 1)[1]
mol = gto.M(atom=xyz, basis=basis, spin=spin, verbose=0)
mf = (dft.UKS if spin else dft.RKS)(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf = mf.SMD(); mf.with_solvent.solvent = 'water'
mf.conv_tol = 1e-10
mf.two_level = True
if cg != 'none':
    mf.set_second_grids(cg if cg == 'sg1' else int(cg))
t = time.perf_counter(); e = mf.kernel(); t1 = time.perf_counter()
g = mf.Gradients().kernel(); t2 = time.perf_counter()
hobj = mf.Hessian(); hobj.grid_response = True
h = hobj.kernel(); t3 = time.perf_counter()
numpy.savez(out, e=e, grad=g, hess=h)
print(f'{xyz} {basis} spin {spin}: E = {e:.12f}; SCF {t1-t:.1f} s, gradient {t2-t1:.1f} s, Hessian {t3-t2:.1f} s; {lib.num_threads()} threads', flush=True)
