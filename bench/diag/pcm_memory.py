"""Where the memory of a large PCM SCF goes: an RKS SCF with PCM on n-CnH(2n+2) (default n-C26H54/def2-TZVP, 1130
AOs), with a sampler thread reading the process RSS every 0.2 s and labelling each sample with the phase being run
(wrappers around the PCM steps, get_jk and the XC). Prints a timeline of new peaks, the peak RSS and the wall time of
each phase, and the SCF result. A watchdog stops the run (exit 3, with the Python stacks) when the RSS passes
--limit-gb, so that the machine never swaps or meets the OOM killer. Run with PYTHONPATH set to the tree;
PYSCF_MAX_MEMORY as usual.
Usage: pcm_memory.py [--n 26] [--basis def2-tzvp] [--xc b3lyp] [--method C-PCM] [--limit-gb 13] [--cycles N]"""
import argparse
import faulthandler
import functools
import os
import sys
import threading
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy  # noqa: E402
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402
from pyscf.dft import numint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=26)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--method', default='C-PCM')
ap.add_argument('--limit-gb', type=float, default=13.0)
ap.add_argument('--cycles', type=int, default=None, help='max SCF cycles (default: PySCF\'s)')
args = ap.parse_args()

T0 = time.perf_counter()


def rss_gb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024**2
    return 0.


stack = ['setup']
peaks = {}
counts = {}
seconds = {}
state = {'peak': 0.}


def label():
    return stack[-1]


def phase(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            stack.append(name)
            counts[name] = counts.get(name, 0) + 1
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                seconds[name] = seconds.get(name, 0.) + time.perf_counter() - t0
                stack.pop()
        return wrapper
    return deco


def sampler():
    while True:
        r = rss_gb()
        ph = label()
        if r > peaks.get(ph, 0.):
            peaks[ph] = r
        if r > state['peak'] + 0.25:
            state['peak'] = r
            print(f'  [{time.perf_counter() - T0:8.1f} s] RSS {r:6.2f} GB  in {ph}', flush=True)
        if r > args.limit_gb:
            print(f'\nWATCHDOG: RSS {r:.2f} GB > {args.limit_gb} GB in {ph} at {time.perf_counter() - T0:.1f} s; '
                  f'stacks:', flush=True)
            faulthandler.dump_traceback(all_threads=True)
            sys.stdout.flush()
            os._exit(3)
        time.sleep(0.2)


threading.Thread(target=sampler, daemon=True).start()

# phase labels: the PCM steps, the J/K and the XC
pcm.gen_surface = phase('pcm gen_surface')(pcm.gen_surface)
pcm.get_D_S = phase('pcm get_D_S (surface matrices)')(pcm.get_D_S)
pcm.PCM.build = phase('pcm build')(pcm.PCM.build)
pcm.PCM._get_v = phase('pcm _get_v (potential on the surface)')(pcm.PCM._get_v)
pcm.PCM._get_vmat = phase('pcm _get_vmat (Fock term)')(pcm.PCM._get_vmat)
pcm.PCM._get_vind = phase('pcm _get_vind (solves)')(pcm.PCM._get_vind)
numint.NumInt.nr_rks = phase('xc nr_rks')(numint.NumInt.nr_rks)
dft.rks.RKS.get_jk = phase('scf get_jk')(dft.rks.RKS.get_jk)

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
mol.verbose = 4
print(f'C{args.n}H{2 * args.n + 2} {args.basis}: {mol.natm} atoms, nao {mol.nao}; {args.xc}, {args.method}; '
      f'{lib.num_threads()} threads, max_memory {mol.max_memory} MB; RSS {rss_gb():.2f} GB', flush=True)
mf = dft.RKS(mol, xc=args.xc).PCM()
mf.with_solvent.method = args.method
mf.verbose = 4
if args.cycles:
    mf.max_cycle = args.cycles
stack.append('scf')
t0 = time.perf_counter()
e = mf.kernel()
stack.pop()
print(f'\nE {e:.10f}, converged {mf.converged}, {mf.cycles} cycles, SCF {time.perf_counter() - t0:.1f} s; '
      f'surface points {mf.with_solvent.surface["grid_coords"].shape[0]}', flush=True)
print('peak RSS by phase (GB; calls):')
for ph, r in sorted(peaks.items(), key=lambda kv: -kv[1]):
    print(f'  {r:6.2f}  {ph}  ({counts.get(ph, 0)})')
print(f'RSS at the end {rss_gb():.2f} GB')
print('wall time by phase (s; calls; a phase includes those it calls: _get_vind calls _get_v and _get_vmat, '
      'build calls get_D_S and gen_surface):')
for ph, t in sorted(seconds.items(), key=lambda kv: -kv[1]):
    print(f'  {t:8.1f}  {ph}  ({counts.get(ph, 0)})')
