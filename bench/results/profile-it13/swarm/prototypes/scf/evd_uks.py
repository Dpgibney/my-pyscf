"""1 thread: is the benzene+ UKS SCF path sensitive to rounding at all? evr vs evd, and evr with the
initial guess perturbed at the 1e-15 relative level."""
import sys, time
import numpy, scipy.linalg
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, dft
from pyscf.scf import hf

orig = hf.SCF._eigh


def eigh_evd(self, h, s, overwrite=False, x=None):
    if x is None:
        return orig(self, h, s, overwrite, x)
    h = x.conj().T.dot(h).dot(x)
    e, c = scipy.linalg.eigh(h, overwrite_a=True, driver='evd')
    return e, hf._adjust_phase_(x.dot(c))


for xc in ('r2SCAN', 'PBE'):
    mol = gto.M(atom=BENZENE, basis='def2-svp', charge=1, spin=1, verbose=0)
    dm0 = dft.UKS(mol, xc=xc).get_init_guess()
    rng = numpy.random.default_rng(3)
    for label, f, eps in (('evr', orig, 0), ('evd', eigh_evd, 0), ('evr, dm0*(1+1e-15 noise)', orig, 1e-15),
                          ('evr, dm0*(1+1e-14 noise)', orig, 1e-14)):
        hf.SCF._eigh = f
        mf = dft.UKS(mol, xc=xc)
        mf.chkfile = None
        noise = rng.standard_normal(dm0.shape) * eps
        noise = noise + noise.transpose(0, 2, 1)
        dm = dm0 * (1 + noise)
        mf.kernel(dm0=dm)
        print(f'benzene+ UKS {xc} {label}: E {mf.e_tot:.12f} cycles {mf.cycles} converged {mf.converged}')
        sys.stdout.flush()
    hf.SCF._eigh = orig
