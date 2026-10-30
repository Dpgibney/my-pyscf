"""Gradient J/K (int2e_ip1, s2kl) in the spherical basis vs evaluated and contracted in the Cartesian
basis with D_cart = C D C^T and V = C^T V_cart C (C = mol.cart2sph_coeff()). 1 thread."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE
from pyscf import gto, scf, lib
from pyscf.grad import rhf as rhf_grad
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[sys.argv[1]]
mol = gto.M(atom=atom, basis=sys.argv[2] if len(sys.argv) > 2 else 'def2-svp', verbose=0)
dm = scf.RHF(mol).get_init_guess(key='minao')
c = mol.cart2sph_coeff()
molc = mol.copy(); molc.cart = True; molc.build(False, False)
dmc = c @ dm @ c.T
def best(f, n=2):
    b = 9e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t0)
    return b, r
ts, (vj, vk) = best(lambda: rhf_grad.get_jk(mol, dm))
tc, (vjc, vkc) = best(lambda: rhf_grad.get_jk(molc, dmc))
vj2 = numpy.einsum('pi,xpq,qj->xij', c, vjc, c)
vk2 = numpy.einsum('pi,xpq,qj->xij', c, vkc, c)
print(f'{sys.argv[1]}: nao sph {mol.nao} cart {molc.nao}; grad J/K spherical {ts:.3f} s, cartesian {tc:.3f} s')
print(f'max |dvj| {abs(vj2-vj).max():.1e} (|vj| {abs(vj).max():.1f}), max |dvk| {abs(vk2-vk).max():.1e}')
de_s = numpy.einsum('xij,ij->x', vj - .5*vk, dm); de_c = numpy.einsum('xij,ij->x', vj2 - .5*vk2, dm)
print('trace check', abs(de_s - de_c).max())
