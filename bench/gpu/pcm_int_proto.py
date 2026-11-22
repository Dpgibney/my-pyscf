"""The two surface integrals of a PCM/SMD SCF cycle through two routes (node1): the potential of the density at the
surface charges, v_g = sum_ij D_ij (ij|g), and the Fock contribution of the charges, V_ij = sum_g q_g (ij|g), with
Gaussian surface charges g. Route "gint": gto.int3c1e.int1e_grids, what solvent.pcm uses. Route "md": the 3-center
J engine of df.j_engine_3c2e (contract_int3c2e_dm, contract_int3c2e_auxvec), which hf.get_hcore uses for the nuclear
attraction. Prints the times (best of --repeat, between device synchronizations, after one warm call) and the largest
differences. Run with PYTHONPATH set to the tree.
Usage: pcm_int_proto.py [--mol caffeine] [--basis def2-svp] [--repeat 3] [--ndm 1]"""
import argparse
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.gto.int3c1e import int1e_grids  # noqa: E402
from gpu4pyscf.df.j_engine_3c2e import Int3c2eOpt, contract_int3c2e_dm, contract_int3c2e_auxvec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--cutoff', type=float, default=1e-14)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE, 'water': WATER}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc='pbe').SMD()
mf.with_solvent.solvent = 'water'
mf.max_cycle = 3
mf.kernel()
pcm = mf.with_solvent
dm = mf.make_rdm1()
charge_exp = pcm.surface['charge_exp']
grid_coords = pcm.surface['grid_coords']
q = pcm._intermediates['q_sym']
n = grid_coords.shape[0]


def timed(fn):
    fn()
    best = 1e99
    for _ in range(args.repeat):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = fn()
        cupy.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - t0)
    return out, best


v0, t_v0 = timed(lambda: int1e_grids(mol, grid_coords, dm=dm, charge_exponents=charge_exp**2, intopt=pcm.intopt))
f0, t_f0 = timed(lambda: int1e_grids(mol, grid_coords, charges=q, charge_exponents=charge_exp**2, intopt=pcm.intopt))

t0 = time.perf_counter()
auxmol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
opt = Int3c2eOpt(mol, auxmol).build(cutoff=args.cutoff)
cupy.cuda.Device().synchronize()
t_build = time.perf_counter() - t0
v1, t_v1 = timed(lambda: contract_int3c2e_dm(mol, auxmol, dm, hermi=1, int3c2e_opt=opt))
f1, t_f1 = timed(lambda: contract_int3c2e_auxvec(mol, auxmol, q, int3c2e_opt=opt))

print(f'{args.mol} {args.basis}: nao={mol.nao} surface points={n}')
print(f'potential at the charges: gint {t_v0:.3f} s, md {t_v1:.3f} s ({t_v0 / t_v1:.2f}x), '
      f'max |d| {float(abs(v1 - v0).max()):.2e} of {float(abs(v0).max()):.2e}')
print(f'Fock matrix of the charges: gint {t_f0:.3f} s, md {t_f1:.3f} s ({t_f0 / t_f1:.2f}x), '
      f'max |d| {float(abs(f1 - f0).max()):.2e} of {float(abs(f0).max()):.2e}')
print(f'md set-up (once per surface): {t_build:.3f} s')
e0 = float(cupy.dot(v0.ravel(), q))
e1 = float(cupy.dot(v1.ravel(), q))
print(f'q.v: gint {e0:.12f}, md {e1:.12f}, difference {e1 - e0:.2e}; '
      f'D.V: gint {float((dm * f0).sum()):.12f}, md {float((dm * f1).sum()):.12f}')
