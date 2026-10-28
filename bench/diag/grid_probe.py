import sys
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='PBE')
g = mf.grids
attrs = ['level', 'atom_grid', 'prune', 'radi_method', 'becke_scheme', 'radii_adjust', 'atomic_radii', 'cutoff', 'alignment']
print({a: (getattr(getattr(g, a), '__name__', getattr(g, a)) if not isinstance(getattr(g, a, None), (dict,)) else 'dict') for a in attrs if hasattr(g, a)})
g.build()
print('ngrids', g.weights.size, 'sum(w)', round(g.weights.sum(), 6))
