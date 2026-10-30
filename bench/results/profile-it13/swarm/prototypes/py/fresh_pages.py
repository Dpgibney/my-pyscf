import sys, time, numpy, resource
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft
from pyscf.dft import numint
mol = gto.M(atom=molecules.CAFFEINE, basis='def2-svp', verbose=0)
grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()
coords = grids.coords[:50000]
def faults():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_minflt
for deriv in (0, 1, 2):
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    buf = numpy.empty((comp, mol.nao, coords.shape[0]))   # eval_ao's layout
    numint.eval_ao(mol, coords, deriv=deriv, out=buf)      # touch
    tf, tr, ff, fr = [], [], 0, 0
    for _ in range(3):
        f0 = faults(); t0 = time.perf_counter(); ao = numint.eval_ao(mol, coords, deriv=deriv); tf.append(time.perf_counter()-t0); ff = faults()-f0; del ao
        f0 = faults(); t0 = time.perf_counter(); ao = numint.eval_ao(mol, coords, deriv=deriv, out=buf); tr.append(time.perf_counter()-t0); fr = faults()-f0; del ao
    mb = comp*mol.nao*coords.shape[0]*8/1e6
    print(f'deriv {deriv}: {mb:.0f} MB; fresh {min(tf)*1e3:.0f} ms ({ff} minor faults), reused out= {min(tr)*1e3:.0f} ms ({fr} faults); diff {(min(tf)-min(tr))*1e3:.0f} ms')
# a fresh large numpy temporary: allocation + first touch alone
for mb in (100, 1000):
    n = int(mb*1e6/8)
    t0 = time.perf_counter(); a = numpy.empty(n); a[:] = 1.0; t1 = time.perf_counter(); a[:] = 2.0; t2 = time.perf_counter()
    print(f'{mb} MB: first write {1e3*(t1-t0):.0f} ms, second write {1e3*(t2-t1):.0f} ms')
    del a
