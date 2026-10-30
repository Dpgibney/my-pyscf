"""Per-task cost of CVHFnr_direct_drv's parallel loop (one task = one (j,k,l) block of AO_BLOCK_SIZE=64 AOs
with all i of the slice), timed at 1 thread by restricting shls_slice to each task's blocks, then a
dynamic-schedule simulation on 4 fast + 6 slow cores. Usage: task_balance.py SYSTEM [block]"""
import sys, time, itertools, heapq
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE
from pyscf import gto, scf, lib
from pyscf.scf import _vhf
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess

system = sys.argv[1]
blk = int(sys.argv[2]) if len(sys.argv) > 2 else 64
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
ao_loc = mol.ao_loc_nr()
nbas = mol.nbas

def partition(sh0, sh1, size):
    loc = [sh0]
    last = ao_loc[sh0]
    for ish in range(sh0 + 1, sh1):
        if ao_loc[ish+1] - last > size:
            loc.append(ish)
            last = ao_loc[ish]
    loc.append(sh1)
    return loc

def simulate(costs, fast=4, slow=6, slow_speed=0.72):
    '''dynamic schedule in task order, speeds relative to a fast core'''
    speeds = [1.0]*fast + [slow_speed]*slow
    heap = [(0.0, s) for s in speeds]
    heapq.heapify(heap)
    for c in costs:
        t, s = heapq.heappop(heap)
        heapq.heappush(heap, (t + c / s, s))
    makespan = max(t for t, s in heap)
    ideal = sum(costs) / sum(speeds)
    return makespan, ideal

vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen', dmcondname='CVHFnr_dm_cond1')
vhfopt.q_cond = rhf_grad._calc_q_cond(mol, vhfopt)
vhfopt.set_dm(dm0, mol._atm, mol._bas, mol._env)
vhfopt.set_dm = lambda *args: None

def dsl(script, dm, sl):
    """slice dm (given on the full AO range or already on rows/cols of the slice) by the script letters"""
    idx = {'i': (0, 1), 'j': (2, 3), 'k': (4, 5), 'l': (6, 7)}
    a, b = script.split('->')[0]
    ra = slice(ao_loc[sl[idx[a][0]]], ao_loc[sl[idx[a][1]]])
    rb = slice(ao_loc[sl[idx[b][0]]], ao_loc[sl[idx[b][1]]])
    return numpy.asarray(dm[ra, rb], order='C')

def grad_task(shls_slice):
    scripts = ('lk->s1ij', 'jk->s1il')
    dms = [dsl(s, dm0, shls_slice) for s in scripts]
    _vhf.direct_bindm('int2e_ip1_sph', 's2kl', scripts, dms, 3,
                      mol._atm, mol._bas, mol._env, vhfopt=vhfopt, shls_slice=shls_slice)

def run(label, ish0, ish1, call, s2kl=True):
    jl = partition(0, nbas, blk)
    costs = []
    for j, k, l in itertools.product(range(len(jl)-1), repeat=3):
        if s2kl and k < l:
            continue
        sl = (ish0, ish1, jl[j], jl[j+1], jl[k], jl[k+1], jl[l], jl[l+1])
        t0 = time.perf_counter(); call(sl); costs.append(time.perf_counter() - t0)
    tot = sum(costs)
    mk, ideal = simulate(costs)
    print(f'{label}: {len(costs)} tasks, 1-thread sum {tot:.3f} s, largest {max(costs):.3f} s; '
          f'10-thread makespan {mk:.3f} s vs ideal {ideal:.3f} s (speedup {tot/mk:.2f}x of {tot/ideal:.2f}x)')

import os
parts = os.environ.get('PARTS', 'grad,hess')
print(system, 'nao', mol.nao, 'block', blk)
if 'grad' in parts:
    run('gradient int2e_ip1 (all i)', 0, nbas, grad_task)
aoslices = mol.aoslice_by_atom()
opt = rhf_hess._make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
hatoms = [int(a) for a in os.environ.get('HATOMS', f'0,{mol.natm-1}').split(',')]
for ia in (hatoms if 'hess' in parts else ()):
    shl0, shl1, p0, p1 = aoslices[ia]
    def ip1ip2(sl):
        sc = ['ji->s1kl', 'li->s1kj', 'lj->s1ki']
        rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                         sum([[s, dsl(s, dm0, sl)] for s in sc], []),
                         shls_slice=sl, vhfopt=opt)
    run(f'hessian ip1ip2 atom {ia}', shl0, shl1, ip1ip2, s2kl=False)
    def ip1(sl):
        sc = ['ji->s2kl', 'lk->s1ij', 'li->s1kj', 'jk->s1il']
        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                         sum([[s, -dsl(s, dm0, sl)] for s in sc], []), shls_slice=sl)
    run(f'make_h1 int2e_ip1 atom {ia}', shl0, shl1, ip1)
