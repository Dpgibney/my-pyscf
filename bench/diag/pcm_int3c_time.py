"""PCM's _get_v on n-C26H54/def2-TZVP split into its two parts, for comparing trees: the int3c2e integrals
(df.incore.aux_e2 on blocks of surface points, as _get_v makes them) and the einsum contraction with the density.
Also prints the libcint the process loaded. Run with PYTHONPATH set to the tree.
Usage: pcm_int3c_time.py [nblocks (default 20)] [points per block (default 123, the original tree's SCF)]"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy  # noqa: E402
from molecules import n_alkane  # noqa: E402
from pyscf import gto, df, dft, lib  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402

nblk = int(sys.argv[1]) if len(sys.argv) > 1 else 20
blksize = int(sys.argv[2]) if len(sys.argv) > 2 else 123
mol = gto.M(atom=n_alkane(26), basis='def2-tzvp', verbose=0)
p = pcm.PCM(mol)
p.build()
grid_coords = p.surface['grid_coords']
exponents = p.surface['charge_exp']
dm = dft.RKS(mol).get_init_guess()
int3c2e = mol._add_suffix('int3c2e')
cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
t_int = t_ein = 0.
blocks = list(lib.prange(0, grid_coords.shape[0], blksize))[:nblk]
for p0, p1 in blocks:
    t0 = time.perf_counter()
    fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
    fakemol.cart = mol.cart
    v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
    t1 = time.perf_counter()
    numpy.einsum('ijL,ij->L', v_nj, dm)
    t2 = time.perf_counter()
    t_int += t1 - t0
    t_ein += t2 - t1
    del v_nj
with open('/proc/self/maps') as f:
    libs = sorted({line.split()[-1] for line in f if 'libcint' in line})
print(f'{len(blocks)} blocks of {blksize} points: int3c2e {t_int:.2f} s, einsum {t_ein:.2f} s '
      f'(per block {t_int / len(blocks):.3f} + {t_ein / len(blocks):.3f} s); {", ".join(libs)}', flush=True)
