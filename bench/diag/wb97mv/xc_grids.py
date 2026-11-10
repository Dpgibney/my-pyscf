'''E_xc (and the XC potential matrix) of a fixed converged density on several XC grids against a large reference
grid; and the VV10 energy on several NLC grids. Usage: xc_grids.py MOL NPZ'''
import argparse
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.dft import gen_grid
from molecules import CAFFEINE, BENZENE, n_alkane

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('npz')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
GEOM = {'caffeine': CAFFEINE, 'benzene': BENZENE}
atom = GEOM[args.mol] if args.mol in GEOM else n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
z = numpy.load(args.npz)
mf = dft.RKS(mol, xc=args.xc)
mf.mo_coeff, mf.mo_occ = z['mo_coeff'], z['mo_occ']
dm = mf.make_rdm1()
ni = mf._numint


def grid(spec, prune=gen_grid.nwchem_prune):
    g = gen_grid.Grids(mol)
    if isinstance(spec, tuple):
        g.atom_grid = spec
    else:
        g.level = spec
    g.prune = prune
    g.build(with_non0tab=True)
    return g


def xc(g):
    t = time.perf_counter()
    n, e, v = ni.nr_rks(mol, g, args.xc, dm, max_memory=8000)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    n, e, v = ni.nr_rks(mol, g, args.xc, dm, max_memory=8000)
    return n, e, v, time.perf_counter() - t, t1


ref = grid((150, 974), None)
n0, e0, v0, t0, _ = xc(ref)
print(f'reference (150,974) unpruned: {ref.weights.size} points, nelec {n0:.8f}, E_xc {e0:.10f}, {t0:.1f} s', flush=True)
print(f'{"grid":34s} {"points":>8s} {"pts/atom":>8s} {"dN":>10s} {"dE_xc":>11s} {"max|dV|":>9s} {"tr(D dV)":>10s} {"s/call":>7s}')
tests = [('level 1', 1, gen_grid.nwchem_prune), ('level 2', 2, gen_grid.nwchem_prune),
         ('level 3 (PySCF default)', 3, gen_grid.nwchem_prune), ('level 4', 4, gen_grid.nwchem_prune),
         ('level 5', 5, gen_grid.nwchem_prune),
         ('(75,302) nwchem', (75, 302), gen_grid.nwchem_prune),
         ('(99,590) nwchem (production)', (99, 590), gen_grid.nwchem_prune),
         ('(99,590) treutler', (99, 590), gen_grid.treutler_prune),
         ('(99,590) unpruned', (99, 590), None),
         ('(75,302) unpruned', (75, 302), None),
         ('(50,194) sg1', (50, 194), gen_grid.sg1_prune)]
for name, spec, prune in tests:
    g = grid(spec, prune)
    n, e, v, t, t1 = xc(g)
    print(f'{name:34s} {g.weights.size:8d} {g.weights.size // mol.natm:8d} {n - n0:+10.2e} {e - e0:+11.3e} '
          f'{abs(v - v0).max():9.2e} {numpy.einsum("ij,ji", dm, v - v0):+10.2e} {t:7.2f}', flush=True)

print('\nVV10 energy by NLC grid (same density)')
gref = grid((99, 590), None)
t = time.perf_counter()
n, er, vr = ni.nr_nlc_vxc(mol, gref, args.xc, dm, max_memory=8000)
print(f'reference (99,590) unpruned: {gref.weights.size} points, E_nlc {er:.10f}, {time.perf_counter() - t:.1f} s', flush=True)
for name, spec, prune in [('level 0', 0, gen_grid.nwchem_prune), ('level 1', 1, gen_grid.nwchem_prune),
                          ('level 2', 2, gen_grid.nwchem_prune),
                          ('(50,194) nwchem (production)', (50, 194), gen_grid.nwchem_prune),
                          ('(50,194) sg1', (50, 194), gen_grid.sg1_prune),
                          ('(50,194) unpruned', (50, 194), None),
                          ('level 3', 3, gen_grid.nwchem_prune)]:
    g = grid(spec, prune)
    t = time.perf_counter()
    n, e, v = ni.nr_nlc_vxc(mol, g, args.xc, dm, max_memory=8000)
    print(f'{name:34s} {g.weights.size:8d} {g.weights.size // mol.natm:8d} dE_nlc {e - er:+11.3e} max|dV| {abs(v - vr).max():9.2e} '
          f'{time.perf_counter() - t:7.2f} s', flush=True)
print('done')
