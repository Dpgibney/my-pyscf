"""GPU4PySCF's J/K builders against molecule size, basis and density (node1). For each molecule a direct PBE SCF
gives a converged density. At that density it times, as the median of REPEAT calls after one untimed call each:
  J-engine  get_j, the MD J-engine (what HF and DFT use for J since #442)
  Rys J     jk.get_j, the Rys J-only kernel
  Rys K     get_k, the Rys K-only kernel (what HF and hybrids use for K)
  Rys J+K   get_jk, both in one Rys pass (what hybrids used before #442)
K-only screening keeps a shell quartet (ij|kl) on the exchange-type density blocks (ik, jk, il, jl); J+K also keeps
it on the Coulomb-type blocks (ij, kl), so it evaluates every quartet that J needs.
With --ddm it also times them on density differences of a B3LYP SCF (the SCF's incremental builds use those).
Usage: jk_scaling.py [--mols caffeine,c10,c20,c30,c40] [--basis def2-svp] [--repeat 3] [--ddm] [--out FILE]"""
import argparse
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks as gpu_rks  # noqa: E402
from gpu4pyscf.scf import jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mols', default='caffeine,c10,c20,c30,c40')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--ddm', action='store_true')
ap.add_argument('--out', default=None)
args = ap.parse_args()


def geometry(name):
    if name.startswith('c') and name[1:].isdigit():
        return n_alkane(int(name[1:]))
    return {'caffeine': CAFFEINE, 'benzene': BENZENE}[name]


def bench(f):
    out = f()
    ts = []
    for _ in range(args.repeat):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = f()
        cupy.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    return out, float(numpy.median(ts))


def builders(mf, mol, dm):
    '''Times of the four builders at dm, and the largest differences between their J and K matrices'''
    vk, t_k = bench(lambda: mf.get_k(mol, dm))
    vhfopt = next(iter(mf._opt_gpu.values()))
    vj, t_je = bench(lambda: mf.get_j(mol, dm))
    vj_rys, t_rj = bench(lambda: jk.get_j(mol, dm, 1, vhfopt))
    (vj2, vk2), t_jk = bench(lambda: mf.get_jk(mol, dm))
    return {'t_jengine': t_je, 't_rys_j': t_rj, 't_rys_k': t_k, 't_rys_jk': t_jk,
            'split_minus_onepass': t_je + t_k - t_jk,
            'dj_rys': float(abs(vj_rys - vj).max()), 'dj_jk': float(abs(vj2 - vj).max()),
            'dk_jk': float(abs(vk2 - vk).max())}


def show(label, r):
    print(f"{label:34s} J-engine {r['t_jengine']:7.3f}  Rys J {r['t_rys_j']:7.3f}  Rys K {r['t_rys_k']:7.3f}  "
          f"Rys J+K {r['t_rys_jk']:7.3f}  split - 1-pass {r['split_minus_onepass']:+7.3f} s  "
          f"(dJ {r['dj_jk']:.0e}, dK {r['dk_jk']:.0e}, dJ Rys {r['dj_rys']:.0e})", flush=True)


for name in args.mols.split(','):
    mol = gto.M(atom=geometry(name), basis=args.basis, verbose=0)
    mf = gpu_rks.RKS(mol, xc='pbe')
    mf.conv_tol = 1e-8
    t0 = time.perf_counter()
    mf.kernel()
    t_scf = time.perf_counter() - t0
    r = builders(mf, mol, cupy.asarray(mf.make_rdm1()))
    r.update({'mol': name, 'basis': args.basis, 'natm': mol.natm, 'nao': mol.nao, 'density': 'converged PBE',
              'pbe_scf_s': round(t_scf, 1), 'pbe_cycles': mf.cycles})
    show(f'{name} {args.basis} nao {mol.nao} (PBE dm)', r)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps(r) + '\n')

    if args.ddm:
        snaps = {}

        def keep(envs):
            if envs['cycle'] + 1 in (2, 6, 10):
                snaps[envs['cycle'] + 1] = cupy.asarray(envs['dm']) - cupy.asarray(envs['dm_last'])
        mfh = gpu_rks.RKS(mol, xc='b3lyp')
        mfh.callback = keep
        mfh.kernel()
        for cyc, ddm in sorted(snaps.items()):
            r = builders(mf, mol, ddm)
            r.update({'mol': name, 'basis': args.basis, 'natm': mol.natm, 'nao': mol.nao,
                      'density': f'B3LYP ddm, cycle {cyc}', 'ddm_max': float(abs(ddm).max())})
            show(f"  ddm cycle {cyc:2d} (max |ddm| {r['ddm_max']:.0e})", r)
            if args.out:
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
