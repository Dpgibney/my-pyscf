import sys, time, numpy
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/jk')
from pyscf import gto, scf
from pyscf.scf import _vhf
from pyscf.hessian import rhf as rhf_hess
import hess_tri
lib = _vhf.libcvhf
for n in ['int2e_ip1ip2','int2e_ipip1ipip2','int2e_ipvip1ipvip2','int2e_ip1ip2_sph','CVHFip1ip2_prescreen','CVHFipvip1_prescreen','CVHFipip1_prescreen']:
    print(n, hasattr(lib, n))
mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59; C 2 0 0; H 2.5 0.9 0; H 2.5 -0.9 0', basis='def2-svp', verbose=0, charge=0, spin=0)
mf = scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel()
h = mf.Hessian()
print(rhf_hess._make_vhfopt(mol, None, 'ip1ip2', 'int2e_ip1ip2'))
t0=time.perf_counter()
_, ej0, ek0 = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
t1=time.perf_counter()
ej1, ek1 = hess_tri.ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
t2=time.perf_counter()
print('orig %.2f tri %.2f'%(t1-t0,t2-t1), abs(ej1-ej0).max(), abs(ek1-ek0).max())
atl=[2,0,5,1,4,3]
_, ej0, ek0 = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
ej1, ek1 = hess_tri.ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
print('perm', abs(ej1-ej0).max(), abs(ek1-ek0).max())
atl=[3,1]
_, ej0, ek0 = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
ej1, ek1 = hess_tri.ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
print('subset', abs(ej1-ej0).max(), abs(ek1-ek0).max())
