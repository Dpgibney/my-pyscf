'''The exact J and K of a range-separated hybrid as two passes: full-range J + K and long-range K (now), or
long-range J + K and short-range J + K (1/r = erf(wr)/r + erfc(wr)/r), where the short-range pass screens better:
the time of each pass and the identity. Usage: sr_lr_split.py caffeine|XYZ [DM.npy] [--charge N]'''
import sys, time, numpy
from pyscf import gto, dft, scf
import molecules
a = sys.argv[1:]
charge = int(a[a.index('--charge') + 1]) if '--charge' in a else 0
mol = gto.M(atom=getattr(molecules, a[0].upper(), None) or a[0], basis='def2-tzvpd', charge=charge, verbose=0)
dm = numpy.load(a[1]) if len(a) > 1 and a[1].endswith('.npy') else scf.hf.init_guess_by_minao(mol)
mf = dft.RKS(mol, xc='wb97m-v'); mf.semi_direct = False
w = 0.3
def timed(label, fn):
    t = time.perf_counter(); r = fn(); print(f'{label:34s} {time.perf_counter() - t:7.1f} s', flush=True); return r
print(f'{mol.natm} atoms, {mol.nao} AOs, direct_scf_tol {mf.direct_scf_tol}')
vj, vk = timed('full-range J + K', lambda: mf.get_jk(mol, dm))
vkl = timed('long-range K', lambda: mf.get_k(mol, dm, omega=w))
vjl, vkl2 = timed('long-range J + K', lambda: mf.get_jk(mol, dm, omega=w))
vjs, vks = timed('short-range J + K', lambda: mf.get_jk(mol, dm, omega=-w))
print(f'J_sr + J_lr - J: {abs(vjs + vjl - vj).max():.1e}; K_sr + K_lr - K: {abs(vks + vkl2 - vk).max():.1e}; '
      f'K_lr of the two calls: {abs(vkl - vkl2).max():.1e}; max|J_sr| {abs(vjs).max():.2f}, max|J| {abs(vj).max():.2f}')
