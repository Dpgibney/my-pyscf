'''The two-level SCF with a seminumerical K in its cheap model (mf.two_level_sgx_level) against the
plain SCF and the two-level SCF whose cheap model builds the exact K.

    python bench/gpu/two_level_cosx.py [--mol caffeine] [--basis def2-svp] [--xc b3lyp] [--smd]
        [--grid 99,590] [--nlcgrid 50,194] [--runs plain,exact,sgx1,sgx1f] [--verbose 4]

runs: plain, exact (two_level, exact K in the cheap model), sgxN (two_level_sgx_level N, double),
sgxNf (single-precision kernels), cosxN (the approximate model mf.COSX() alone, level N). A two-level
run takes +jfit, +jkfit or +etb for a fitted J in its cheap model (two_level_auxbasis
def2-universal-jfit, def2-universal-jkfit, or PySCF's even-tempered aug_etb): sgx1f+jfit.
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.sgx import sgx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--spin', type=int, default=0)
p.add_argument('--basis', default='def2-svp')
p.add_argument('--xc', default='b3lyp')
p.add_argument('--smd', action='store_true')
p.add_argument('--grid', default=None)
p.add_argument('--nlcgrid', default=None)
p.add_argument('--conv', type=float, default=1e-10)
p.add_argument('--runs', default='plain,exact,sgx1,sgx1f')
p.add_argument('--verbose', type=int, default=0)
p.add_argument('--set', action='append', default=[], help='two_level module constants, NAME=VALUE')
args = p.parse_args()

from gpu4pyscf.scf import two_level
import ast
for item in args.set:
    k, v = item.split('=')
    setattr(two_level, k, ast.literal_eval(v))

atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
        'water': molecules.WATER}.get(args.mol, args.mol)
mol = pyscf.M(atom=atom, basis=args.basis, charge=args.charge, spin=args.spin, verbose=args.verbose)
print('%s %s %s%s: %d AOs%s' % (args.mol, args.basis, args.xc, ' SMD water' if args.smd else '',
                               mol.nao, '' if args.grid is None else ', grids %s / %s' % (args.grid, args.nlcgrid)))

def model():
    mf = dft.RKS(mol, xc=args.xc) if mol.spin == 0 else dft.UKS(mol, xc=args.xc)
    if args.grid:
        mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
    if args.nlcgrid:
        mf.nlcgrids.atom_grid = tuple(int(x) for x in args.nlcgrid.split(','))
    if args.smd:
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    mf.conv_tol = args.conv
    return mf

e_ref = None
print('run         E                     E - plain   cycles (references)   SCF s')
AUX = {'jfit': 'def2-universal-jfit', 'jkfit': 'def2-universal-jkfit'}
for name in args.runs.split(','):
    run, _, aux = name.partition('+')
    mf = model()
    if aux == 'etb':
        from pyscf import df as cpu_df
        mf.two_level_auxbasis = cpu_df.aug_etb(mol)
    elif aux:
        mf.two_level_auxbasis = AUX[aux]
    if run == 'plain':
        pass
    elif run == 'exact':
        mf.two_level = True
    elif run.startswith('sgx'):
        mf.two_level = True
        mf.two_level_sgx_level = int(run[3])
        if run.endswith('f'):
            mf.two_level_sgx_dtype = 'float'
    elif run.startswith('cosx'):
        mf = mf.COSX()
        mf.with_sgx.grids_level = int(run[4])
        if run.endswith('f'):
            mf.with_sgx.kernel_dtype = 'float'
    else:
        raise ValueError(run)
    cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cp.cuda.get_current_stream().synchronize()
    t = time.perf_counter() - t0
    if run == 'plain':
        e_ref = e
    if getattr(mf, 'two_level', False):
        cheap = getattr(mf, '_two_level_counts', None)
        cyc = '%s refs' % mf.cycles if cheap is None else '%d + %d (%d refs)' % (cheap[0], cheap[1], mf.cycles)
    else:
        cyc = '%d' % mf.cycles
    print('%-11s %.12f  %s  %-20s %8.1f  %s' % (
        name, e, '          ' if e_ref is None else '%+.2e' % (e - e_ref), cyc, t,
        '' if mf.converged else 'NOT CONVERGED'), flush=True)
