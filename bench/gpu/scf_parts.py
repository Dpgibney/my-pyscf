"""Per-part SCF profile (node1): the wall time of each SCF component, for PySCF on the CPU and GPU4PySCF on the GPU,
with the same molecule, basis, functional and threshold. Each part's time is exclusive (time spent in nested timed
calls is subtracted) and is measured between device synchronizations, so asynchronous GPU work is charged to the
call that queued it. The first call of each method (which builds screening data, in-memory ERIs or grids) is
reported separately from the per-cycle cost.
Devices: cpu (PySCF defaults: in-memory ERIs when they fit), cpu-direct (PySCF forced to direct SCF, as GPU4PySCF
always is), gpu (GPU4PySCF, profiled on its second SCF so that CUDA start-up is excluded; the verbose-6 log of that
run supplies GPU4PySCF's own sub-timers of the XC part).
Usage: scf_parts.py [--mol caffeine|benzene] [--basis def2-svp] [--xc pbe,b3lyp,r2scan] [--df]
                    [--device cpu,cpu-direct,gpu] [--tol 1e-10] [--verbose 6] [--logdir .] [--out parts.jsonl]
--verbose 4 drops GPU4PySCF's sub-timers, which synchronize after every J/K kernel launch (+6% for B3LYP's K)."""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from gpu4pyscf.dft import rks as gpu_rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='pbe,b3lyp,r2scan')
ap.add_argument('--df', action='store_true')
ap.add_argument('--device', default='cpu,cpu-direct,gpu')
ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--verbose', type=int, default=6)
ap.add_argument('--logdir', default='.')
ap.add_argument('--out', default=None)
args = ap.parse_args()

# (part, path from mf to the method's owner, method)
TIMED = [('init guess', '', 'get_init_guess'), ('hcore+ovlp', '', 'get_hcore'), ('hcore+ovlp', '', 'get_ovlp'),
         ('grids', 'grids', 'build'), ('XC', '_numint', 'nr_rks'),
         ('J', '', 'get_j'), ('K', '', 'get_k'), ('J+K', '', 'get_jk'), ('J+K', 'with_df', 'get_jk'),
         ('veff rest', '', 'get_veff'), ('Fock+DIIS', '', 'get_fock'), ('eig', '', 'eig'),
         ('occ+density', '', 'get_occ'), ('occ+density', '', 'make_rdm1'), ('energy', '', 'energy_tot'),
         ('gradient', '', 'get_grad')]
PARTS = list(dict.fromkeys(part for part, _, _ in TIMED)) + ['untimed']
# GPU4PySCF's own timers (wall time, summed over the run) that split the XC part
SUBTIMERS = {'eval rho on Device 0': 'XC: density on grid', 'eval vxc on Device 0': 'XC: functional (libxc)',
             'eval integration on 0': 'XC: Vxc integration'}


def sync():
    cupy.cuda.Device().synchronize()


def instrument(mf):
    '''Wrap each method of TIMED; returns {method key: [part, calls, first-call s, total s]} (exclusive times)'''
    rec, stack = {}, []

    def wrap(key, f):
        def timed(*a, **kw):
            sync()
            t0 = time.perf_counter()
            stack.append(0.)
            try:
                return f(*a, **kw)
            finally:
                sync()
                dt = time.perf_counter() - t0
                inner = stack.pop()
                if stack:
                    stack[-1] += dt
                r = rec[key]
                r[1] += 1
                if r[1] == 1:
                    r[2] = dt - inner
                r[3] += dt - inner
        timed.__wrapped__ = f
        return timed

    for part, path, name in TIMED:
        owner = mf
        for attr in filter(None, path.split('.')):
            owner = getattr(owner, attr, None)
        method = getattr(owner, name, None) if owner is not None else None
        if callable(method):
            key = f'{path}.{name}'
            rec[key] = [part, 0, 0., 0.]
            # an owner shared with an earlier run keeps that run's wrapper: wrap the original instead
            setattr(owner, name, wrap(key, getattr(method, '__wrapped__', method)))
    return rec


def make(device, mol, xc):
    mf = (gpu_rks.RKS if device == 'gpu' else dft.RKS)(mol, xc=xc)
    if args.df:
        mf = mf.density_fit()
    if device == 'cpu-direct':
        mf._is_mem_enough = lambda: False
    mf.conv_tol = args.tol
    return mf


def run(device, mol, xc):
    if device == 'gpu':
        make(device, mol, xc).kernel()  # warm-up: CUDA context, module loading, memory pool
    mf = make(device, mol, xc)
    rec = instrument(mf)
    tag = f"{args.mol}_{args.basis}_{xc}{'_df' if args.df else ''}_{device}_v{args.verbose}"
    log = os.path.join(args.logdir, f'parts_{tag}.log')
    with open(log, 'w') as f:
        # GPU4PySCF's numint logs through mol (new_logger(mol, None)), so mol gets the level too
        mf.verbose = mol.verbose = args.verbose
        mf.stdout = mol.stdout = f
        sync()
        t0 = time.perf_counter()
        mf.kernel()
        sync()
        total = time.perf_counter() - t0
        mol.verbose, mol.stdout = 0, sys.stdout
    parts = defaultdict(lambda: [0, 0., 0.])  # part: calls, first-call s, total s
    for part, calls, first, tot in rec.values():
        if calls:
            p = parts[part]
            p[0] += calls
            p[1] += first
            p[2] += tot
    parts['untimed'] = [0, 0., total - sum(p[2] for p in parts.values())]
    sub = defaultdict(float)
    if device == 'gpu':
        # GPU-event time: GPU4PySCF's timer reads the host clock before synchronizing, so a label's host wall
        # time also contains waiting for the work queued under the previous label
        pat = re.compile(r'CPU time for (.+?)\s+[-\d.]+ sec, wall time\s+[-\d.]+ sec, GPU time\s+([-\d.]+) ms')
        for line in open(log):
            m = pat.search(line)
            if m and m.group(1).strip() in SUBTIMERS:
                sub[SUBTIMERS[m.group(1).strip()]] += float(m.group(2)) / 1e3
    res = {'mol': args.mol, 'basis': args.basis, 'xc': xc, 'df': args.df, 'device': device, 'nao': mol.nao,
           'verbose': args.verbose,
           'e_tot': float(mf.e_tot), 'converged': bool(mf.converged), 'cycles': int(mf.cycles),
           'ngrids': int(mf.grids.weights.size), 'total_s': round(total, 3),
           'parts': {k: {'calls': v[0], 'first_s': round(v[1], 4), 'total_s': round(v[2], 4)}
                     for k, v in parts.items()},
           'gpu_subtimers_s': {k: round(v, 4) for k, v in sub.items()}}
    if device == 'gpu':
        res['gpu_pool_gib'] = round(cupy.get_default_memory_pool().total_bytes() / 2**30, 3)
    return res


mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
for xc in args.xc.split(','):
    results = []
    for device in args.device.split(','):
        r = run(device, mol, xc)
        results.append(r)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(r) + '\n')
    print(f"\n{args.mol} {args.basis} {xc}{' DF' if args.df else ''}, nao {mol.nao}, conv_tol {args.tol:g}")
    print('  ' + ' ' * 22 + ''.join(f"{r['device']:>24s}" for r in results))
    print('  ' + f"{'E (Eh)':22s}" + ''.join(f"{r['e_tot']:24.10f}" for r in results))
    print('  ' + f"{'cycles':22s}" + ''.join(f"{r['cycles']:24d}" for r in results))
    print('  ' + f"{'SCF wall (s)':22s}" + ''.join(f"{r['total_s']:24.2f}" for r in results))
    print('  ' + f"{'part: first | per cycle':22s}" + ''.join(f"{'first s  per-cycle s':>24s}" for _ in results))
    for part in PARTS:
        if not any(part in r['parts'] for r in results):
            continue
        cells = []
        for r in results:
            p = r['parts'].get(part)
            if p is None:
                cells.append(f"{'-':>24s}")
            elif part == 'untimed':
                cells.append(f"{'':>9s}{p['total_s'] / r['cycles']:15.3f}")
            else:
                cells.append(f"{p['first_s']:9.3f}{(p['total_s'] - p['first_s']) / r['cycles']:15.3f}")
        print(f'  {part:22s}' + ''.join(cells))
    for r in results:
        for k, v in r['gpu_subtimers_s'].items():
            print(f"  {r['device']} {k:30s} {v / r['cycles']:8.3f} s per cycle")
        if 'gpu_pool_gib' in r:
            print(f"  {r['device']} CuPy memory pool after the run: {r['gpu_pool_gib']} GiB")
    sys.stdout.flush()
