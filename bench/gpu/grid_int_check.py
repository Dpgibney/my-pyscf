'''The two contractions of gpu4pyscf.sgx.int1e_grids (the potential of a density at points, the matrix
of charges at points) against libcint and against gpu4pyscf.gto.int3c1e, and their time against the
latter's on the surface of a solvated molecule.

1. A small molecule, random points (some at and near the nuclei, some far), random symmetric density
   and charges: point charges against mol.intor('int1e_grids'), Gaussian charges of random exponents
   against int3c2e with one normalized s function per point; both against int3c1e.int1e_grids.
2. --time: caffeine (or --mol) in --tbasis with the SMD surface of the molecule: the two contractions
   of the reaction field, int3c1e against these kernels in double and single precision.

    python bench/gpu/grid_int_check.py [--basis def2-tzvp] [--time [--mol caffeine] [--tbasis def2-tzvp]]
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from pyscf import df, gto
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.sgx import int1e_grids

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--ngrids', type=int, default=300)
p.add_argument('--atom', default='O 0 0 0.12; C 0.05 0.1 -1.08; H 0.3 0.94 -1.62; H -0.2 -0.94 -1.66')
p.add_argument('--time', action='store_true')
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--tbasis', default='def2-tzvp')
args = p.parse_args()

def sync():
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter()

mol = pyscf.M(atom=args.atom, basis=args.basis, verbose=0)
nao = mol.nao
rng = np.random.default_rng(11)
ng = args.ngrids
xyz = mol.atom_coords()
coords = np.vstack([
    xyz,
    xyz[rng.integers(0, mol.natm, ng//4)] + rng.normal(size=(ng//4, 3)) * .05,
    xyz[rng.integers(0, mol.natm, ng//2)] + rng.normal(size=(ng//2, 3)) * 1.5,
    rng.normal(size=(ng, 3)) * 12.,
])[:ng]
dm = rng.normal(size=(nao, nao))
dm = dm + dm.T
charges = rng.normal(size=ng)
expnt = np.exp(rng.uniform(np.log(.3), np.log(200.), ng))
print('%s  %s  nao %d  ngrids %d' % (args.atom, args.basis, nao, ng))

opt = int1e_grids.Int1eGridsOpt(mol)
old = int3c1e.VHFOpt(mol)
old.build(1e-14)
for label, ex in (('point charges', None), ('Gaussian charges', expnt)):
    if ex is None:
        a = mol.intor('int1e_grids', grids=coords).transpose(1, 2, 0)
    else:
        fake = gto.fakemol_for_charges(coords, expnt=ex)
        a = df.incore.aux_e2(mol, fake, intor='int3c2e')
    v_ref = np.einsum('mng,mn->g', a, dm)
    m_ref = a.dot(charges)
    cex = None if ex is None else cp.asarray(ex)
    v_old = int3c1e.int1e_grids(mol, cp.asarray(coords), dm=cp.asarray(dm), charge_exponents=cex, intopt=old).get()
    m_old = int3c1e.int1e_grids(mol, cp.asarray(coords), charges=cp.asarray(charges), charge_exponents=cex, intopt=old).get()
    print('%s: max |v| %.2f, max |V| %.2f; int3c1e against libcint: v %.1e, V %.1e' % (
        label, abs(v_ref).max(), abs(m_ref).max(), abs(v_old - v_ref).max(), abs(m_old - m_ref).max()))
    for dtype in ('double', 'float'):
        v = opt.density_contracted(cp.asarray(dm), coords, cex, dtype).get()
        m = opt.charge_contracted(cp.asarray(charges), coords, cex, dtype).get()
        tol = 1e-13 if dtype == "double" else 2e-5
        ok = abs(v - v_ref).max() < tol * abs(v_ref).max() and abs(m - m_ref).max() < tol * abs(m_ref).max()
        print('   %-6s against libcint: v %.1e, V %.1e   against int3c1e: v %.1e, V %.1e   %s' % (
            dtype, abs(v - v_ref).max(), abs(m - m_ref).max(), abs(v - v_old).max(), abs(m - m_old).max(),
            'ok' if ok else 'FAILED'))

if args.time:
    from gpu4pyscf import dft
    atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
            'water': molecules.WATER}.get(args.mol, args.mol)
    mol = pyscf.M(atom=atom, basis=args.tbasis, charge=args.charge, verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').SMD()
    mf.with_solvent.solvent = 'water'
    pcm = mf.with_solvent
    pcm.build()
    coords = pcm.surface['grid_coords']
    cex = pcm.surface['charge_exp']**2
    ng = len(coords)
    dm = cp.asarray(mf.get_init_guess())
    q = cp.asarray(np.random.default_rng(3).normal(size=ng) * 1e-2)
    t0 = sync()
    opt = int1e_grids.Int1eGridsOpt(mol)
    t1 = sync()
    print('%s %s: %d AOs, %d surface points; pair data %.2f s: %d shell pairs, %d primitive pairs' % (
        args.mol, args.tbasis, mol.nao, ng, t1 - t0, sum(c['npairs'] for c in opt.classes),
        sum(c['nprims'] for c in opt.classes)))

    def best(f, n=3):
        f()
        t = []
        for i in range(n):
            t0 = sync()
            out = f()
            t.append(sync() - t0)
        return out, min(t)
    v_old, tv_old = best(lambda: int3c1e.int1e_grids(mol, coords, dm=dm, charge_exponents=cex, intopt=pcm.intopt))
    m_old, tm_old = best(lambda: int3c1e.int1e_grids(mol, coords, charges=q, charge_exponents=cex, intopt=pcm.intopt))
    print('int3c1e:        potential of the density %.3f s, matrix of the charges %.3f s' % (tv_old, tm_old))
    for dtype in ('double', 'float'):
        v, tv = best(lambda: opt.density_contracted(dm, coords, cex, dtype))
        m, tm = best(lambda: opt.charge_contracted(q, coords, cex, dtype))
        print('these, %-6s:  potential of the density %.3f s, matrix of the charges %.3f s;  '
              'max |v - v_int3c1e| %.1e (max |v| %.1f), max |V - V_int3c1e| %.1e (max |V| %.2e)' % (
                  dtype, tv, tm, float(abs(v - v_old).max()), float(abs(v_old).max()),
                  float(abs(m - m_old).max()), float(abs(m_old).max())))
