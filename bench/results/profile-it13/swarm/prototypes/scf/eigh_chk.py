"""1-thread: eigh driver costs for the SCF sizes (evr = scipy default, evd), and the cost of the
per-cycle chkfile write (mf.dump_chk) and save_mol that a default SCF (mf.chkfile set) performs."""
import sys, time, os, tempfile
import numpy, scipy.linalg
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto, lib, scf, dft
from pyscf.scf import chkfile


def tm(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r


rng = numpy.random.default_rng(1)
for n in (246, 494, 872, 1200):
    a = rng.standard_normal((n, n)); a = a + a.T
    t_evr, (e1, c1) = tm(lambda: scipy.linalg.eigh(a), 3)
    t_evd, (e2, c2) = tm(lambda: scipy.linalg.eigh(a, driver='evd'), 3)
    t_np, (e3, c3) = tm(lambda: numpy.linalg.eigh(a), 3)
    print(f'n={n}: scipy evr {t_evr*1e3:.1f} ms, scipy evd {t_evd*1e3:.1f} ms, numpy.linalg.eigh (evd) {t_np*1e3:.1f} ms, '
          f'max|de| evr-evd {abs(e1-e2).max():.1e}')
    sys.stdout.flush()

for name, atom, basis in [('caffeine/SVP', CAFFEINE, 'def2-svp'), ('caffeine/TZVP', CAFFEINE, 'def2-tzvp'),
                          ('C20H42/SVP', n_alkane(20), 'def2-svp')]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    nao = mol.nao
    mf = dft.RKS(mol, xc='B3LYP')
    mf.chkfile = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False).name
    mo_coeff = rng.standard_normal((nao, nao))
    mo_energy = numpy.sort(rng.standard_normal(nao))
    mo_occ = numpy.zeros(nao); mo_occ[:mol.nelectron // 2] = 2
    envs = dict(e_tot=-1.0, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    t_save, _ = tm(lambda: chkfile.save_mol(mol, mf.chkfile), 3)
    t_dump, _ = tm(lambda: mf.dump_chk(envs), 5)
    os.unlink(mf.chkfile)
    print(f'{name}: chkfile.save_mol {t_save*1e3:.1f} ms (once), mf.dump_chk {t_dump*1e3:.2f} ms per cycle')
    sys.stdout.flush()
