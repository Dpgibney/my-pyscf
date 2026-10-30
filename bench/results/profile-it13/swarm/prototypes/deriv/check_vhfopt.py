import sys
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, lib
from pyscf.scf import _vhf
from pyscf.hessian import rhf as rhf_hess
import numpy
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
lib_ = _vhf.libcvhf
for name in ('int2e_ip1ip2', 'int2e_ipip1ipip2', 'int2e_ipvip1ipvip2', 'int2e_ip1ip2_sph',
             'int2e_ipip1ipip2_sph', 'int2e_ipvip1ipvip2_sph', 'CVHFip1ip2_prescreen',
             'CVHFipip1_prescreen', 'CVHFipvip1_prescreen', 'CVHFgrad_jk_prescreen'):
    print(name, hasattr(lib_, name))
dm0 = numpy.eye(mol.nao)
for key, vi in (('ipip1', 'int2e_ipip1ipip2'), ('ip1ip2', 'int2e_ip1ip2'), ('ipvip1', 'int2e_ipvip1ipvip2')):
    print(key, rhf_hess._make_vhfopt(mol, dm0, key, vi))
