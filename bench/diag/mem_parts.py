# Where a RIJCOSX SCF's memory goes, by part of the cycle: RSS sampled every 0.05 s by phase (J, K, XC, VV10, the
# rest), the RSS before and after each call (what stays), and the largest live numpy allocations (tracemalloc).
# n-C20H42/def2-TZVP, 3 cycles, VV10 on nlcgrids level 1. Usage: mem_parts.py CASE, CASE one of wb97x, wb97m-v,
# wb97m-v-novv10 (mf.nlc = 0), wb97m-v-oldvv10 (nr_nlc_vxc on its block_loop path: the fused kernels off for the
# VV10 call only)
import sys, os, time, threading, tracemalloc, linecache
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane
import pyscf
from pyscf import gto, dft, lib
from pyscf.sgx import sgx_jk
from pyscf.df import df_jk
case = sys.argv[1]
xc = 'wb97x' if case == 'wb97x' else 'wb97m-v'
mol = gto.M(atom=n_alkane(20), basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc=xc).RIJCOSX()
mf.with_df.grids_level_energy = None
mf.nlcgrids.level = 1
if case == 'wb97m-v-novv10':
    mf.nlc = 0
mf.max_cycle = 3
page = os.sysconf('SC_PAGE_SIZE')
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * page / 1e9
state = {'phase': 'other'}
peaks = {}
done = threading.Event()
def sampler():
    while not done.wait(0.05):
        r = rss(); p = state['phase']
        peaks[p] = max(peaks.get(p, 0), r)
threading.Thread(target=sampler, daemon=True).start()
events = []
def wrap(obj, name, phase):
    fn = getattr(obj, name)
    def w(*a, **k):
        before = rss(); old = state['phase']; state['phase'] = phase
        try:
            return fn(*a, **k)
        finally:
            state['phase'] = old
            events.append((phase, before, rss()))
    setattr(obj, name, w)
wrap(mf._numint, 'nr_rks', 'xc')
if case == 'wb97m-v-oldvv10':
    # the block_loop path of nr_nlc_vxc: the fused kernels off for the VV10 call only
    from pyscf.dft import numint
    fn = mf._numint.nr_nlc_vxc
    def old_nlc(*a, **k):
        numint.FUSED_XC = False
        try:
            return fn(*a, **k)
        finally:
            numint.FUSED_XC = True
    mf._numint.nr_nlc_vxc = old_nlc
wrap(mf._numint, 'nr_nlc_vxc', 'vv10')
wrap(sgx_jk, 'get_k_only', 'k')
wrap(df_jk, 'get_j', 'j')
tracemalloc.start(8)
t = time.perf_counter()
mf.kernel()
snap = tracemalloc.take_snapshot()
done.set()
print(f'{case}: {mf.cycles} cycles, {time.perf_counter() - t:.0f} s, RSS at the end {rss():.2f} GB, max_memory {mf.max_memory} MB')
print('  peak RSS by phase: ' + ', '.join(f'{p} {v:.2f}' for p, v in sorted(peaks.items(), key=lambda kv: -kv[1])))
first = {}
for p, b, a in events:
    first.setdefault(p, []).append((b, a))
for p, lst in first.items():
    print(f'  {p:5s} calls: RSS before -> after: ' + ', '.join(f'{b:.2f}->{a:.2f}' for b, a in lst[:4]))
stats = snap.statistics('traceback')
total = sum(s.size for s in stats)
print(f'  live allocations traced by tracemalloc: {total / 1e9:.2f} GB; the largest:')
for s in stats[:8]:
    fr = [f for f in s.traceback if 'pyscf' in f.filename][:3]
    where = ' <- '.join(f'{os.path.relpath(f.filename, os.path.dirname(pyscf.__file__))}:{f.lineno}' for f in fr)
    print(f'    {s.size / 1e9:6.3f} GB  {where}')
