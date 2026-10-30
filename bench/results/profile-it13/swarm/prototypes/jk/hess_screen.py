"""The Hessian's ipip1/ip1ip2/ipvip1 J/K with the screening _make_vhfopt was meant to give them
(it returns None: hasattr(libcvhf, 'int2e_ip1ip2') is False, only the suffixed names exist), alone and
with the triangle restriction of hess_tri.py. Times (1 thread) and differences of ej, ek.
Usage: hess_screen.py SYSTEM"""
import sys, time, ctypes
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/jk')
from molecules import BENZENE, CAFFEINE, WATER
from pyscf import gto, scf, lib
from pyscf.scf import _vhf
from pyscf.hessian import rhf as rhf_hess
import hess_tri

def make_vhfopt_fixed(mol, dm0, key, vhf_intor):
    libcvhf = _vhf.libcvhf
    vhfopt = _vhf._VHFOpt(mol, 'int2e_'+key, 'CVHF'+key+'_prescreen', dmcondname='CVHFnr_dm_cond1')
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q_cond = numpy.empty((2, nbas, nbas))
    with mol.with_integral_screen(vhfopt.direct_scf_tol**2):
        fqcond = libcvhf.CVHFnr_int2e_pp_q_cond if vhf_intor == 'int2e_ip1ip2' else libcvhf.CVHFnr_int2e_pppp_q_cond
        fqcond(getattr(libcvhf, mol._add_suffix(vhf_intor)), lib.c_null_ptr(), q_cond[0].ctypes,
               ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
               mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        libcvhf.CVHFnr_int2e_q_cond(getattr(libcvhf, mol._add_suffix('int2e')), lib.c_null_ptr(),
               q_cond[1].ctypes, ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
               mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    vhfopt.q_cond = q_cond
    # dm_cond of dm0: every dm the Hessian passes is a block of dm0 at its global AO indices
    vhfopt.set_dm(dm0, mol._atm, mol._bas, mol._env)
    vhfopt.set_dm = lambda *args: None
    return vhfopt

if __name__ == '__main__':
    system = sys.argv[1]
    atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'water': WATER,
            'ethylene': 'C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237'}[system]
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    hobj = mf.Hessian()
    t0 = time.perf_counter()
    e1, ej0, ek0 = rhf_hess._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    t1 = time.perf_counter()
    rhf_hess._make_vhfopt = make_vhfopt_fixed
    hess_tri._make_vhfopt = make_vhfopt_fixed
    e1, ej1, ek1 = rhf_hess._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    t2 = time.perf_counter()
    ej2, ek2 = hess_tri.ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    t3 = time.perf_counter()
    print(f'{system}: unscreened {t1-t0:.2f} s, screened {t2-t1:.2f} s, screened+triangle {t3-t2:.2f} s')
    print('screened vs unscreened: max |dej|', abs(ej1-ej0).max(), 'max |dek|', abs(ek1-ek0).max(),
          ' (max |ej|', abs(ej0).max(), ')')
    print('screened+triangle vs screened: max diff', abs(ej2-ej1).max(), abs(ek2-ek1).max())
