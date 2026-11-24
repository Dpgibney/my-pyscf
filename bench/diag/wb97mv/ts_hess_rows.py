'''Rows of the Hessian at a saddle point of ts_rijcosx.py (PREFIX.xyz, PREFIX.npz): the exact-J/K analytical Hessian,
central differences of the exact-J/K gradient, and the RIJCOSX finite-difference Hessian saved in PREFIX.npz, along a
few Cartesian coordinates; with the solvent and, with --gas, without it.
Usage: ts_hess_rows.py PREFIX --charge -1 [--solvent water] [--gas] [--coords 0z,1z,1x,3x]'''
import argparse, time, numpy
from pyscf import gto, dft, grad, hessian  # noqa: F401
p = argparse.ArgumentParser()
p.add_argument('prefix'); p.add_argument('--charge', type=int, default=0); p.add_argument('--solvent', default='water')
p.add_argument('--basis', default='def2-tzvpd'); p.add_argument('--gas', action='store_true')
p.add_argument('--coords', default='0z,1z,1x,3x'); p.add_argument('--step', type=float, default=1e-3)
a = p.parse_args()
T0 = time.perf_counter()
XYZ = 'xyz'


def say(msg):
    print(f'>>> [{time.perf_counter() - T0:6.0f} s] {msg}', flush=True)


rows = [(int(c[:-1]), XYZ.index(c[-1])) for c in a.coords.split(',')]
d = numpy.load(a.prefix + '.npz')
h_rij = d['hess']                     # [natm,natm,3,3], RIJCOSX with the solvent
mol = gto.M(atom=a.prefix + '.xyz', basis=a.basis, charge=a.charge, verbose=0)
for solvent in ([a.solvent, None] if a.gas else [a.solvent]):
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    if solvent:
        mf = mf.SMD()
        mf.with_solvent.solvent = solvent
    mf.two_level = True
    mf.conv_tol = 1e-11
    mf.kernel()
    t = time.perf_counter()
    h = mf.Hessian().kernel()
    say(f'{solvent or "gas"}: analytical Hessian {time.perf_counter() - t:.0f} s')
    gs = mf.Gradients().as_scanner()
    x0 = mol.atom_coords()
    for i, k in rows:
        g = []
        for s in (a.step, -a.step):
            x = x0.copy()
            x[i, k] += s
            g.append(gs(mol.set_geom_(x, unit='Bohr', inplace=False))[1])
        fd = (g[0] - g[1]) / (2 * a.step)       # d g / d x[i,k], [natm,3]
        an = h[i, :, k, :]
        line = (f'{solvent or "gas"} row {mol.atom_symbol(i)}{i}{XYZ[k]}: max |analytical - FD(exact)| '
                f'{abs(an - fd).max():.1e}, diagonal analytical {an[i, k]:.6f} FD(exact) {fd[i, k]:.6f}')
        if solvent:
            line += (f' FD(RIJCOSX) {h_rij[i, i, k, k]:.6f}; max |FD(exact) - FD(RIJCOSX)| '
                     f'{abs(fd - h_rij[i, :, k, :]).max():.1e}')
        say(line)
say(f'total {time.perf_counter() - T0:.0f} s')
