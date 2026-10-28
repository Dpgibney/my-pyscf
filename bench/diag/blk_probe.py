import sys
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
orig = numint.NumInt.block_loop
def probe(self, mol, grids, nao=None, deriv=0, max_memory=2000, non0tab=None, blksize=None, buf=None):
    n = 0; sizes = set(); sparse = 0
    for ao, mask, weight, coords in orig(self, mol, grids, nao, deriv, max_memory, non0tab, blksize, buf):
        n += 1; sizes.add(weight.size); sparse += mask is not None
        yield ao, mask, weight, coords
    print(f'  block_loop deriv={deriv} max_memory={max_memory:.0f}MB rss={lib.current_memory()[0]:.0f}MB blocks={n} sparse={sparse} sizes={sorted(sizes)[-2:]}')
numint.NumInt.block_loop = probe
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='r2SCAN'); mf.chkfile = None; mf.max_cycle = 2
mf.kernel()
