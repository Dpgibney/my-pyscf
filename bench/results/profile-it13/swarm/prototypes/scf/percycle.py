"""1-thread costs of the per-cycle pieces outside J/K and the XC kernels (caffeine/SVP, B3LYP-like dm)."""
import sys, time
import numpy, scipy.linalg
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto, lib, scf, dft
from pyscf.dft import numint


def tm(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r


systems = [('caffeine/SVP', CAFFEINE, 'def2-svp'), ('caffeine/TZVP', CAFFEINE, 'def2-tzvp'),
           ('C20H42/SVP', n_alkane(20), 'def2-svp')]
if len(sys.argv) > 1:
    systems.append(('C20H42/TZVP', n_alkane(20), 'def2-tzvp'))
for name, atom, basis in systems:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    nao = mol.nao
    mf = dft.RKS(mol, xc='B3LYP')
    t_ovlp, s = tm(lambda: mf.get_ovlp(mol), 3)
    t_hcore, h = tm(lambda: mf.get_hcore(mol), 3)
    t_guess, dm = tm(lambda: mf.get_init_guess(mol, 'minao'), 2)
    t_cld, x = tm(lambda: mf.check_linear_dependency(s), 3)
    t_ovc, _ = tm(lambda: mol.get_overlap_cond())
    t_cond, _ = tm(lambda: mol.condense_to_shell(dm, 'absmax'))
    # Fock-like matrix
    f = h + 0.1 * dm
    t_eig, (e, c) = tm(lambda: mf.eig(f, s, x=x))
    fx = x.T.dot(f).dot(x)
    t_evr, _ = tm(lambda: scipy.linalg.eigh(fx))
    t_evd, _ = tm(lambda: scipy.linalg.eigh(fx, driver='evd'))
    t_tr, _ = tm(lambda: x.T.dot(f).dot(x))
    mo_occ = mf.get_occ(e, c)
    t_rdm1, dm1 = tm(lambda: mf.make_rdm1(c, mo_occ))
    t_grad, _ = tm(lambda: mf.get_grad(c, mo_occ, f))
    t_fock_diis = None
    # DIIS: CDIIS update with a few vectors (as in get_fock with diis)
    from pyscf.scf import diis as scf_diis
    adiis = scf_diis.CDIIS(mf, None)
    adiis.Corth = x
    rng = numpy.random.default_rng(0)
    for k in range(6):
        r1 = rng.random((nao, nao)) * 1e-3; r2 = rng.random((nao, nao)) * 1e-3
        adiis.update(s, dm + r1 + r1.T, f + r2 + r2.T)
    t_diis, _ = tm(lambda: adiis.update(s, dm, f), 3)
    t_e1, _ = tm(lambda: numpy.einsum('ij,ji->', h, dm1).real)
    print(f'{name}: nao {nao} nbas {mol.nbas}')
    print(f'  once: get_ovlp {t_ovlp*1e3:.1f} ms, get_hcore {t_hcore*1e3:.1f} ms, minao guess {t_guess*1e3:.1f} ms, '
          f'check_linear_dependency {t_cld*1e3:.1f} ms')
    print(f'  per cycle: get_overlap_cond {t_ovc*1e3:.2f} ms, condense_to_shell(absmax) {t_cond*1e3:.2f} ms')
    print(f'  per cycle: mf.eig {t_eig*1e3:.1f} ms (eigh evr {t_evr*1e3:.1f}, evd {t_evd*1e3:.1f}, '
          f'XtFX {t_tr*1e3:.1f}), make_rdm1 {t_rdm1*1e3:.1f}, get_grad {t_grad*1e3:.1f}, '
          f'CDIIS update (7 vecs) {t_diis*1e3:.1f}, e1 einsum {t_e1*1e3:.2f} ms')
    sys.stdout.flush()
