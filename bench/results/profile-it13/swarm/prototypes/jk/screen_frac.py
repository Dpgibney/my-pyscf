"""Fraction of the Hessian's derivative-integral quartets (weighted by block size) that the intended
Schwarz x density screening at direct_scf_tol=1e-14 would keep. Usage: screen_frac.py SYSTEM"""
import sys, ctypes
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, scf, lib
from pyscf.scf import _vhf

system = sys.argv[1]
basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-svp'
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20h42': n_alkane(20), 'c12h26': n_alkane(12)}[system]
mol = gto.M(atom=atom, basis=basis, verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
libcvhf = _vhf.libcvhf
ao_loc = mol.ao_loc_nr()
nbas = mol.nbas
nf = numpy.diff(ao_loc).astype(float)
thr = 1e-14

def qc(intor, f):
    q = numpy.empty((nbas, nbas))
    with mol.with_integral_screen(thr**2):
        f(getattr(libcvhf, mol._add_suffix(intor)), lib.c_null_ptr(), q.ctypes, ao_loc.ctypes,
          mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    return q

q0 = qc('int2e', libcvhf.CVHFnr_int2e_q_cond)
qpp = qc('int2e_ip1ip2', libcvhf.CVHFnr_int2e_pp_q_cond)
qpppp = qc('int2e_ipip1ipip2', libcvhf.CVHFnr_int2e_pppp_q_cond)
dmc = numpy.empty((nbas, nbas))
libcvhf.CVHFnr_dm_cond1(dmc.ctypes, dm0.ctypes, ctypes.c_int(1), ao_loc.ctypes, mol._atm.ctypes,
                        ctypes.c_int(mol.natm), mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
dmax = dmc.max()
w = numpy.outer(nf, nf)

def frac_schwarz(qa, qb):
    '''weighted fraction of (ij,kl) with qa_ij*qb_kl*max|D| > thr'''
    a = qa.ravel(); wa = w.ravel()
    b = qb.ravel(); wb = w.ravel()
    order = numpy.argsort(b)
    bs = b[order]; wbs = numpy.cumsum(wb[order][::-1])[::-1]; wbs = numpy.append(wbs, 0)
    with numpy.errstate(divide='ignore'):
        m = numpy.searchsorted(bs, thr / (a * dmax), side='right')
    kept = (wa * wbs[m]).sum()
    return kept / (wa.sum() * wb.sum())

print(f'{system}/{basis}: nao {mol.nao}, nbas {nbas}, max dm_cond {dmax:.2f}')
print(f'  ip1ip2 (q_pp * q_pp): kept {frac_schwarz(qpp, qpp):.3f} of the quartet volume')
print(f'  ipip1/ipvip1 (q_pppp * q_int2e): kept {frac_schwarz(qpppp, q0):.3f}')
print(f'  plain int2e (q * q): kept {frac_schwarz(q0, q0):.3f}')
