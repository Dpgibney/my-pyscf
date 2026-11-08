'''The VV10 part of the RKS nuclear gradient (grad.rks.get_nlc_vxc) on the fused
kernels against its block_loop path (FUSED_XC off for the call only): the
matrices, time and the peak RSS of each call (sampled every 0.02 s).'''
import sys, os, time, threading
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane, CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

page = os.sysconf('SC_PAGE_SIZE')
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * page / 1e9

def call(fused, *args, **kw):
    peak = [rss()]
    done = threading.Event()
    def sampler():
        while not done.wait(0.02):
            peak[0] = max(peak[0], rss())
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    saved = numint.FUSED_XC
    numint.FUSED_XC = fused
    before = rss()
    t0 = time.perf_counter()
    try:
        r = rks_grad.get_nlc_vxc(*args, **kw)
    finally:
        numint.FUSED_XC = saved
        done.set(); th.join()
    return r, time.perf_counter() - t0, before, peak[0]

cases = [('caffeine', 'def2-tzvp'), ('c20h42', 'def2-tzvp')]
if len(sys.argv) > 1:
    cases = [c for c in cases if c[0] in sys.argv[1:]]
for name, basis in cases:
    if name == 'c20h42':
        mol = gto.M(atom=n_alkane(20), basis=basis, verbose=0)
    else:
        mol = gto.M(atom=CAFFEINE, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.nlcgrids.level = 1
    dm = mf.get_init_guess()
    mf.initialize_grids(mol, dm)
    mo_e, mo = mf.eig(mf.get_hcore(), mf.get_ovlp())
    mo_occ = numpy.zeros(mol.nao); mo_occ[:mol.nelectron//2] = 2
    dm = lib.tag_array(mf.make_rdm1(mo, mo_occ), mo_coeff=mo, mo_occ=mo_occ)
    ni = mf._numint
    max_memory = max(2000, mf.max_memory*.9 - lib.current_memory()[0])
    print(f'{name}/{basis} nao {mol.nao} nlc grid {mf.nlcgrids.weights.size} '
          f'max_memory {max_memory:.0f} MB')
    res = {}
    for fused in (False, True, False, True):
        (exc, v), t, before, peak = call(fused, ni, mol, mf.nlcgrids, 'wb97m-v', dm,
                                        max_memory=max_memory)
        res[fused] = v
        print(f'  {"fused" if fused else "block_loop":10s} {t:6.2f} s  RSS before {before:.2f} GB, '
              f'peak {peak:.2f} GB (+{peak-before:.2f})')
    print(f'  max|dv| {abs(res[True]-res[False]).max():.1e} (|v| {abs(res[False]).max():.2e})')
    sys.stdout.flush()
