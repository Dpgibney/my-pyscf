'''A production-size wB97M-V/def2-TZVPD energy and gradient in SMD through the public options: grids (99,590) and
(50,194), exact J and K (or RIJCOSX), the two-level SCF, then the analytical gradient. PySCF's own output at verbose 4 with its
timers on (the wall time of each step), and a line per phase marked ">>>" with the wall time, the resident memory
and its peak since the previous such line.
Usage: prod_scf_grad.py XYZ [--charge 0] [--solvent water] [--rijcosx] [--plain] [--dm0 DM.npy] [--save PREFIX]
  --rijcosx RI-J in the even-tempered fitting basis (df.aug_etb) and seminumerical K instead of the exact J and K
  --plain   the plain SCF instead of mf.two_level
  --dm0     start the SCF from a saved density matrix (PREFIX_dm.npy of an earlier run)'''
import argparse, time, numpy
import pyscf
from pyscf import gto, dft, df, lib, grad
from pyscf.lib import logger
logger.TIMER_LEVEL = 4      # timer lines at verbose 4
p = argparse.ArgumentParser()
p.add_argument('xyz'); p.add_argument('--charge', type=int, default=0); p.add_argument('--solvent', default='water')
p.add_argument('--basis', default='def2-tzvpd'); p.add_argument('--plain', action='store_true')
p.add_argument('--dm0'); p.add_argument('--save'); p.add_argument('--rijcosx', action='store_true')
a = p.parse_args()
T0 = time.perf_counter()


def memory():
    '''resident memory now and its peak since the last call, GB'''
    d = {}
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(('VmRSS', 'VmHWM')):
                d[line[:5]] = int(line.split()[1]) / 1048576
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')        # the peak starts again from here
    except OSError:
        pass
    return d['VmRSS'], d['VmHWM']


PEAKS = []


def say(msg, phase=None):
    rss, peak = memory()
    if phase:
        PEAKS.append((phase, peak))
    print(f'>>> [{time.perf_counter() - T0:6.0f} s] {msg}; resident {rss:.1f} GB, peak since the last line {peak:.1f} GB',
          flush=True)


mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, verbose=4)
mf = dft.RKS(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
if a.rijcosx:
    mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
mf = mf.SMD()
mf.with_solvent.solvent = a.solvent
mf.two_level = not a.plain
say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads, max_memory {mf.max_memory:.0f} MB; {mol.natm} atoms, '
    f'{mol.nelectron} electrons, {mol.nao} AOs; wB97M-V/{a.basis} in SMD {a.solvent}, '
    f'{"RIJCOSX (aug_etb)" if a.rijcosx else "exact J and K"}, '
    f'{"plain" if a.plain else "two-level"} SCF')

t = time.perf_counter()
mf.with_solvent.build()
surface = mf.with_solvent._intermediates['D']
n = surface.shape[0]
say(f'solvent surface: {time.perf_counter() - t:.1f} s, {n} points, {n * n * 8 / 1e9:.2f} GB for each [n,n] matrix',
    'solvent surface')

t = time.perf_counter()
e = mf.kernel(dm0=numpy.load(a.dm0) if a.dm0 else None)
t_scf = time.perf_counter() - t
assert mf.with_solvent._intermediates['D'] is surface, 'the SCF built the solvent surface again'
say(f'SCF: {t_scf:.1f} s, {mf.cycles} {"cycles" if a.plain else "Fock builds of the model"}, converged {mf.converged}, '
    f'E = {e:.10f} Eh; XC grid {mf.grids.weights.size} points, VV10 grid {mf.nlcgrids.weights.size} points', 'SCF')
if a.save:
    numpy.save(a.save + '_dm.npy', mf.make_rdm1())

t = time.perf_counter()
g = mf.Gradients().kernel()
t_grad = time.perf_counter() - t
say(f'gradient: {t_grad:.1f} s, max |g| {abs(g).max():.3e}, rms {numpy.sqrt((g**2).mean()):.3e} Eh/Bohr, '
    f'sum over atoms {abs(g.sum(axis=0)).max():.1e}', 'gradient')
if a.save:
    numpy.savez(a.save + '_result.npz', e_tot=e, grad=g, t_scf=t_scf, t_grad=t_grad)
print('>>> summary: SCF %.0f s, gradient %.0f s, total %.0f s; peak resident memory by phase: %s'
      % (t_scf, t_grad, time.perf_counter() - T0, ', '.join(f'{k} {v:.1f} GB' for k, v in PEAKS)), flush=True)
