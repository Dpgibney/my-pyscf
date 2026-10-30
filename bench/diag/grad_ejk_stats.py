"""Orbits and integral evaluations of CVHFgrad_jk_energy (It20) against the int2e_ip1 evaluations the
s2kl driver of grad.rhf.get_jk makes for the same density.
Usage: grad_ejk_stats.py SYSTEM [BASIS]"""
import ctypes
import os
import sys
import time
import numpy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import WATER, BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, scf, lib  # noqa: E402
from pyscf.scf import _vhf  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402

system = sys.argv[1]
basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-svp'
atom = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20h42': n_alkane(20)}[system]
mol = gto.M(atom=atom, basis=basis, verbose=0)
dm = scf.RHF(mol).get_init_guess(key='minao')
dm = (dm + dm.T) * .5

vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen', dmcondname='CVHFnr_dm_cond1')
vhfopt.q_cond = rhf_grad._calc_q_cond(mol, vhfopt)
vhfopt.set_dm(dm, mol._atm, mol._bas, mol._env)
intor = mol._add_suffix('int2e_ip1')
ao_loc = _vhf.make_loc(mol._bas, intor)
de = numpy.zeros((mol.natm, 3))
stats = numpy.zeros(3)
t0 = time.perf_counter()
_vhf.libcvhf.CVHFgrad_jk_energy(
    de.ctypes, getattr(_vhf.libcvhf, intor), vhfopt._cintopt, ctypes.byref(vhfopt._this),
    dm.ctypes, dm[None].ctypes, ctypes.c_int(1), ctypes.c_double(1.), ctypes.c_double(.5),
    ctypes.c_int(1), ctypes.c_int(1), stats.ctypes,
    ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
t1 = time.perf_counter()
norbit, neval, npass = stats
print(f'{system}/{basis}: nbas {mol.nbas}, orbits screened in {norbit:.4g}, evaluations: s2kl {npass:.4g}, '
      f'new {neval:.4g} ({neval / npass:.3f}), per orbit {neval / norbit:.2f}; {t1 - t0:.2f} s')
