"""1 thread: SCF with the stock eigensolver (scipy evr) vs driver='evd' in SCF._eigh: energies, cycles,
and the time spent in _eigh."""
import sys, time
import numpy, scipy.linalg
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, WATER, CAFFEINE
from pyscf import gto, dft, scf
from pyscf.scf import hf

orig = hf.SCF._eigh
T = {'t': 0.0, 'n': 0}


def timed(f):
    def g(self, h, s, overwrite=False, x=None):
        t0 = time.perf_counter()
        r = f(self, h, s, overwrite, x)
        T['t'] += time.perf_counter() - t0; T['n'] += 1
        return r
    return g


def eigh_evd(self, h, s, overwrite=False, x=None):
    if x is None:
        return orig(self, h, s, overwrite, x)
    h = x.conj().T.dot(h).dot(x)
    e, c = scipy.linalg.eigh(h, overwrite_a=True, driver='evd')
    c = hf._adjust_phase_(x.dot(c))
    return e, c


cases = [('benzene B3LYP', BENZENE, 0, 0, 'B3LYP'), ('benzene r2SCAN', BENZENE, 0, 0, 'r2SCAN'),
         ('benzene+ UKS PBE', BENZENE, 1, 1, 'PBE'), ('water B3LYP', WATER, 0, 0, 'B3LYP')]
if len(sys.argv) > 1:
    cases.append(('caffeine PBE', CAFFEINE, 0, 0, 'PBE'))
for name, atom, charge, spin, xc in cases:
    mol = gto.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)
    res = {}
    for label, f in (('evr', orig), ('evd', eigh_evd)):
        for tight in (False, True):
            hf.SCF._eigh = timed(f)
            T['t'] = 0.; T['n'] = 0
            mf = dft.RKS(mol, xc=xc) if spin == 0 else dft.UKS(mol, xc=xc)
            mf.chkfile = None
            if tight:
                mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-8; mf.max_cycle = 100
            t0 = time.perf_counter()
            mf.kernel()
            res[label, tight] = (mf.e_tot, mf.cycles, mf.converged, time.perf_counter() - t0, T['t'], T['n'], mf.make_rdm1())
    hf.SCF._eigh = orig
    for tight in (False, True):
        a, b = res['evr', tight], res['evd', tight]
        print(f'{name} {"tight" if tight else "default"}: E evr {a[0]:.12f} ({a[1]} cyc, conv {a[2]}), evd {b[0]:.12f} '
              f'({b[1]} cyc, conv {b[2]}), dE {b[0]-a[0]:.1e}, max|dDM| {abs(b[6]-a[6]).max():.1e}; '
              f'_eigh {a[4]*1e3:.1f} ms/{a[5]} calls vs {b[4]*1e3:.1f} ms/{b[5]}')
    sys.stdout.flush()
