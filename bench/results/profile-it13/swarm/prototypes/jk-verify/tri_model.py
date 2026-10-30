import sys, time, itertools, heapq, os
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
blk = int(os.environ.get('BLK', '64'))
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
ao_loc = mol.ao_loc_nr(); nbas = mol.nbas
aoslices = mol.aoslice_by_atom()
def partition(sh0, sh1, size):
    loc = [sh0]; last = ao_loc[sh0]
    for ish in range(sh0 + 1, sh1):
        if ao_loc[ish+1] - last > size:
            loc.append(ish); last = ao_loc[ish]
    loc.append(sh1); return loc
def simulate(costs, fast=4, slow=6, sp=0.72):
    heap = [(0.0, s) for s in [1.0]*fast + [sp]*slow]; heapq.heapify(heap)
    for c in costs:
        t, s = heapq.heappop(heap); heapq.heappush(heap, (t + c/s, s))
    return max(t for t, s in heap)
def dsl(script, sl):
    idx = {'i': (0, 1), 'j': (2, 3), 'k': (4, 5), 'l': (6, 7)}
    a, b = script.split('->')[0]
    ra = slice(ao_loc[sl[idx[a][0]]], ao_loc[sl[idx[a][1]]])
    rb = slice(ao_loc[sl[idx[b][0]]], ao_loc[sl[idx[b][1]]])
    return numpy.asarray(dm0[ra, rb], order='C')
def ip1ip2(sl):
    sc = ['ji->s1kl', 'li->s1kj', 'lj->s1ki']
    rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1', sum([[s, dsl(s, sl)] for s in sc], []), shls_slice=sl)
def ipvip1(sl):
    sc = ['lk->s1ij', 'li->s1kj']
    rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl', sum([[s, dsl(s, sl)] for s in sc], []), shls_slice=sl)
res = {}
for ia in range(mol.natm):
    shl0, shl1 = aoslices[ia][:2]
    bsh1 = shl1
    for name, f, s2kl in (('ip1ip2', ip1ip2, False), ('ipvip1', ipvip1, True)):
        for mode in ('full', 'tri'):
            jl = partition(0, nbas, blk)
            if mode == 'full':
                jr = kr = lr = jl
            else:
                cut = partition(0, bsh1, blk)
                if name == 'ip1ip2': jr, kr, lr = jl, cut, jl
                else: jr, kr, lr = cut, jl, jl
            costs = []
            for j, k, l in itertools.product(range(len(jr)-1), range(len(kr)-1), range(len(lr)-1)):
                if s2kl and kr[k] < lr[l]:  # k block below l block: skipped (approx for same partition)
                    continue
                sl = (shl0, shl1, jr[j], jr[j+1], kr[k], kr[k+1], lr[l], lr[l+1])
                t0 = time.perf_counter(); f(sl); costs.append(time.perf_counter()-t0)
            res[(ia, name, mode)] = (sum(costs), simulate(costs), len(costs))
for mode in ('full', 'tri'):
    for name in ('ip1ip2', 'ipvip1'):
        s1 = sum(res[(ia, name, mode)][0] for ia in range(mol.natm))
        s10 = sum(res[(ia, name, mode)][1] for ia in range(mol.natm))
        print(f'blk {blk} {name} {mode}: 1-thread {s1:.2f} s, modelled 10-thread {s10:.2f} s')
for ia in range(mol.natm):
    print(ia, mol.atom_symbol(ia), ' '.join('%s/%s %.2f/%.2f n%d' % (n, m, res[(ia,n,m)][0], res[(ia,n,m)][1], res[(ia,n,m)][2]) for n in ('ip1ip2','ipvip1') for m in ('full','tri')))
