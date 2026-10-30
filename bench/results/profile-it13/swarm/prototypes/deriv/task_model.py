'''Model the OpenMP decomposition of CVHFnr_direct_drv (nr_direct.c:398-466) for the Hessian's
per-atom calls: tasks = (j,k,l) AO blocks of <= AO_BLOCK_SIZE functions (schedule(dynamic,1)), the i
blocks (one atom) looped inside a task. Task cost ~ number of AO quartets (di*dj*dk*dl summed),
which reproduced the measured 1-thread ratio of the k restriction (0.551 model vs 0.56 measured).
Makespan by greedy list scheduling on 10 equal workers in blk_id order.'''
import sys, heapq
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, BENZENE
from pyscf import gto

def partition(ao_loc, sh0, sh1, bs):
    if sh0 >= sh1:
        return []
    loc = [sh0]; last = ao_loc[sh0]
    for ish in range(sh0+1, sh1):
        if ao_loc[ish+1] - last > bs:
            loc.append(ish); last = ao_loc[ish]
    loc.append(sh1)
    return list(zip(loc[:-1], loc[1:]))

def call_tasks(mol, shls_slice, aosym, bs):
    ao_loc = mol.ao_loc_nr()
    nfn = lambda b: ao_loc[b[1]] - ao_loc[b[0]]
    bi = partition(ao_loc, *shls_slice[0:2], bs)
    bj = partition(ao_loc, *shls_slice[2:4], bs)
    bk = partition(ao_loc, *shls_slice[4:6], bs)
    bl = partition(ao_loc, *shls_slice[6:8], bs)
    ni = sum(nfn(b) for b in bi)
    tasks = []
    for j in bj:
        for k in bk:
            for l in bl:
                if aosym == 's2kl':
                    if k[0] < l[0]:
                        tasks.append(0.); continue
                    if k[0] == l[0]:
                        # lower triangle of the diagonal block
                        f = ao_loc[k[0]:k[1]+1]
                        w = sum((f[a+1]-f[a])*(f[b+1]-f[b]) for a in range(len(f)-1) for b in range(a+1))
                        tasks.append(ni*nfn(j)*w); continue
                tasks.append(ni*nfn(j)*nfn(k)*nfn(l))
    return tasks

def makespan(tasks, nthreads=10):
    h = [0.]*nthreads
    for t in tasks:
        x = heapq.heappop(h); heapq.heappush(h, x + t)
    return max(h)

for name, atom in (('benzene', BENZENE), ('caffeine', CAFFEINE)):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    aosl = mol.aoslice_by_atom()
    nbas = mol.nbas
    print(f'== {name}: nao {mol.nao}, natm {mol.natm}')
    for bs in (64, 32, 16):
        tot = {}
        for label, aosym, restrict in (('ip1ip2 full', 's1', None), ('ip1ip2 k<=A', 's1', 'k'),
                                       ('ipvip1 full', 's2kl', None), ('ipvip1 j<=A', 's2kl', 'j'),
                                       ('ip1 make_h1', 's2kl', None)):
            work = span = 0.; ntask = []
            for ia in range(mol.natm):
                sh0, sh1 = aosl[ia][:2]
                sl = [sh0, sh1, 0, nbas, 0, nbas, 0, nbas]
                if restrict == 'k':
                    sl[5] = sh1
                elif restrict == 'j':
                    sl[3] = sh1
                t = call_tasks(mol, sl, aosym, bs)
                work += sum(t); span += makespan(t)
                ntask.append(sum(1 for x in t if x > 0))
            tot[label] = (work, span)
            print(f'  block {bs:2d} {label:12s}: tasks/call {min(ntask)}-{max(ntask)}, '
                  f'10-thread efficiency {work/10/span:.2f}')
        # whole-molecule ipip1 (one call)
        t = call_tasks(mol, [0, nbas]*4, 's2kl', bs)
        print(f'  block {bs:2d} ipip1 (1 call): tasks {sum(1 for x in t if x > 0)}, '
              f'efficiency {sum(t)/10/makespan(t):.2f}')
