"""The qcint kernels of branch pcm-kernel (int1e_grids_dm_contract_cart, int1e_grids_charge_contract_cart) against
int1e_grids_cart, for every ordered shell pair of formaldehyde (argv[1]: the basis), a random density and random charges
on 600 points in three exponent groups. Needs a qcint build of that branch (LD_LIBRARY_PATH)."""
import sys, ctypes
import numpy
from pyscf import gto, lib
from pyscf.scf import _vhf
lib_ = _vhf.libcvhf
basis = sys.argv[1] if len(sys.argv) > 1 else 'def2-tzvp'
mol = gto.M(atom='C 0 0 -0.53; O 0 0 0.67; H 0 0.94 -1.12; H 0 -0.94 -1.12', basis=basis, cart=True, verbose=0)
rng = numpy.random.default_rng(3)
ng = 600
coords = mol.atom_coords().mean(axis=0) + rng.normal(scale=3.5, size=(ng, 3))
zetas = numpy.array([30., 55., 90.])
group = rng.integers(0, 3, ng)
order = numpy.argsort(group, kind='stable')
coords = coords[order]
group = group[order]
gl = numpy.append(0, numpy.cumsum(numpy.bincount(group, minlength=3)))
batch_loc = numpy.append(numpy.hstack([numpy.arange(gl[k], gl[k+1], 150) for k in range(3)]), ng).astype(numpy.int32)
batch_zeta = numpy.ascontiguousarray(zetas[group[batch_loc[:-1]]])
nao = mol.nao
ints = numpy.zeros((ng, nao, nao))
for k in range(3):
    idx = numpy.where(group == k)[0]
    with mol.with_rinv_zeta(zetas[k]):
        ints[idx] = mol.intor('int1e_grids_cart', grids=coords[idx])
dm = rng.normal(size=(nao, nao)); dm = dm + dm.T
q = rng.normal(size=ng)
v_ref = numpy.einsum('gij,ij->g', ints, dm)
V_ref = numpy.einsum('gij,g->ij', ints, q)
env = numpy.append(mol._env, coords.ravel())
env[gto.NGRIDS] = ng
env[gto.PTR_GRIDS] = mol._env.size
ao_loc = mol.ao_loc_nr(cart=True)
fdm = lib_.int1e_grids_dm_contract_cart
fq = lib_.int1e_grids_charge_contract_cart
v = numpy.zeros(ng)
V = numpy.zeros((nao, nao))
worst = 0
for ish in range(mol.nbas):
    for jsh in range(mol.nbas):
        i0, i1, j0, j1 = ao_loc[ish], ao_loc[ish+1], ao_loc[jsh], ao_loc[jsh+1]
        shls = (ctypes.c_int*4)(ish, jsh, 0, ng)
        blk = numpy.ascontiguousarray(dm[i0:i1, j0:j1].T)
        vp = numpy.zeros(ng)
        args = (shls, mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                env.ctypes.data_as(ctypes.c_void_p), None, None, ctypes.c_int(batch_loc.size-1),
                batch_loc.ctypes.data_as(ctypes.c_void_p), batch_zeta.ctypes.data_as(ctypes.c_void_p))
        fdm(vp.ctypes.data_as(ctypes.c_void_p), blk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1), ctypes.c_int(ng), *args)
        vp_ref = numpy.einsum('gij,ij->g', ints[:, i0:i1, j0:j1], dm[i0:i1, j0:j1])
        worst = max(worst, abs(vp - vp_ref).max() / max(abs(vp_ref).max(), 1e-300))
        v += vp
        vij = numpy.zeros((j1-j0, i1-i0))
        fq(vij.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1), ctypes.c_int(ng), *args)
        V[i0:i1, j0:j1] = vij.T
print(f'{basis}: {mol.nbas} shells, {nao} Cartesian AOs, {ng} points; largest relative difference of one pair {worst:.1e}')
print(f'potential: max |dv| {abs(v - v_ref).max():.2e} (max |v| {abs(v_ref).max():.2e}); '
      f'Fock term: max |dV| {abs(V - V_ref).max():.2e} (max |V| {abs(V_ref).max():.2e})')
