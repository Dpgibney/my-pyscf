"""PCM's two integral steps (_get_v, _get_vmat) on n-C26H54/def2-TZVP (1130 AOs, 9998 surface points), timed with
the block sizes that their max_memory check gives: with the RSS over max_memory (a 3 GB ballast array standing in for
the modified tree's ERI cache: the floor of 8 points), and with 4000 MB (the original tree's runs) and 8000 MB of
max_memory. Mind the machine's RAM: a block is (max_memory - RSS)*0.9 of it. Run with PYTHONPATH set to the tree."""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy  # noqa: E402
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402

mol = gto.M(atom=n_alkane(26), basis='def2-tzvp', verbose=0)
p = pcm.PCM(mol)
t0 = time.perf_counter()
p.build()
print(f'build {time.perf_counter() - t0:.1f} s, RSS {lib.current_memory()[0]:.0f} MB', flush=True)
dm = dft.RKS(mol).get_init_guess()[None]
ngrids = p.surface['grid_coords'].shape[0]
q = numpy.ones((1, ngrids)) * 1e-3
nao = mol.nao
for label, ballast_gb, mm in (('RSS over max_memory (3 GB ballast), max_memory 4000 MB', 3, 4000),
                              ('max_memory 4000 MB (the original tree)', 0, 4000),
                              ('max_memory 8000 MB', 0, 8000)):
    ballast = numpy.ones(int(ballast_gb * 1e9 / 8)) if ballast_gb else None
    p.max_memory = mm
    free = mm - lib.current_memory()[0]
    blk = int(max(free * .9e6 / 8 / nao**2, 8))
    t0 = time.perf_counter()
    p._get_v(dm)
    t1 = time.perf_counter()
    p._get_vmat(q)
    t2 = time.perf_counter()
    print(f'{label}: RSS {lib.current_memory()[0]:.0f} MB, block {blk} points ({(ngrids + blk - 1) // blk} blocks); '
          f'_get_v {t1 - t0:.1f} s, _get_vmat {t2 - t1:.1f} s', flush=True)
    del ballast
