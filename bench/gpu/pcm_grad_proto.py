"""The integral term of the PCM/SMD nuclear gradient through two routes (node1): solvent.grad.pcm.grad_qv, which
contracts gto.int3c1e_ip's derivative integrals of the surface charges (the legacy gint kernels), and the kernel of the
density-fitted Coulomb gradient, libvhf_rys.sum_ejk_int3c2e_ip1, with the surface charges as the auxiliary vector:
it gives sum_ij,g D_ij q_g d(ij|g)/dR per atom of the basis functions and, by translational invariance, per charge.
Prints the times (best of --repeat after a warm call) and the difference of the two gradients, with the factors of
the second route found by a least-squares fit (they should come out as integers).
Usage: pcm_grad_proto.py [--mol caffeine] [--basis def2-svp] [--repeat 3]"""
import argparse
import ctypes
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.solvent.grad import pcm as pcm_grad  # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt, int3c2e_scheme, libvhf_rys  # noqa: E402
from gpu4pyscf.gto.mole import _split_l_ctr_groups  # noqa: E402
from gpu4pyscf.lib.cupy_helper import transpose_sum  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE, 'water': WATER}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc='pbe').SMD()
mf.with_solvent.solvent = 'water'
mf.kernel()
pcm = mf.with_solvent
dm = mf.make_rdm1()
q = pcm._intermediates['q_sym']
charge_exp = pcm.surface['charge_exp']
grid_coords = pcm.surface['grid_coords']
gridslice = pcm.surface['gslice_by_atom']
nsurf = grid_coords.shape[0]


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


de_ref, t_ref = timed(lambda: pcm_grad.grad_qv(pcm, dm))

t0 = time.perf_counter()
fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
opt = Int3c2eOpt(mol, fakemol).build()
cupy.cuda.Device().synchronize()
t_build = time.perf_counter() - t0
smol, saux = opt.mol, opt.auxmol


def rys_route():
    dms = smol.apply_C_mat_CT(dm)
    dms = transpose_sum(dms, inplace=True)
    dms *= .5
    auxvec = saux.C_dot_mat(q)
    naux = len(auxvec)
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(gout_width=54, deriv=(1, 0, 0))
    lmax = smol.uniq_l_ctr[:, 0].max()
    laux = saux.uniq_l_ctr[:, 0].max()
    shm_size_max = shm_size[:laux+1, :lmax+1, :lmax+1].max()
    bas_ij_idx, shl_pair_offsets = smol.aggregate_shl_pairs(opt.bas_ij_cache, nsp_per_block[0]*16)
    _, aux_counts = _split_l_ctr_groups(saux.uniq_l_ctr, saux.l_ctr_counts, [16]*(laux+1))
    ksh_offsets_cpu = numpy.append(0, numpy.cumsum(aux_counts))
    ksh_offsets_gpu = cupy.asarray(ksh_offsets_cpu + smol.nbas, dtype=numpy.int32)
    envs = opt.int3c2e_envs
    ej = cupy.zeros((mol.natm, 3))
    ej_aux = cupy.zeros((nsurf, 3))
    err = libvhf_rys.sum_ejk_int3c2e_ip1(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p), ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p), ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1), ctypes.c_double(0.), ctypes.c_double(1.), ctypes.c_double(1.),
        ctypes.byref(envs), ctypes.c_int(int(shm_size_max)),
        ctypes.c_int(len(shl_pair_offsets) - 1), ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(mol.natm))
    assert err == 0
    ej_grid = cupy.asarray([cupy.sum(ej_aux[p0:p1], axis=0) for p0, p1 in gridslice])
    return ej.get(), ej_grid.get()


(ej, ej_grid), t_new = timed(rys_route)
# de_ref = a * ej + b * ej_grid
a_mat = numpy.stack([ej.ravel(), ej_grid.ravel()], axis=1)
coef, *_ = numpy.linalg.lstsq(a_mat, numpy.asarray(de_ref).ravel(), rcond=None)
fit = (a_mat @ coef).reshape(de_ref.shape)
print(f'{args.mol} {args.basis}: nao={mol.nao} surface points={nsurf}')
print(f'grad_qv {t_ref:.3f} s; Rys 3-center kernel {t_new:.3f} s ({t_ref / t_new:.2f}x), set-up once {t_build:.3f} s')
print(f'fitted factors: {coef[0]:.12f} (functions), {coef[1]:.12f} (charges); '
      f'max |d| after the fit {abs(fit - de_ref).max():.2e} of {abs(de_ref).max():.2e}')
ints = numpy.round(coef)
print(f'with factors {ints}: max |d| {abs((a_mat @ ints).reshape(de_ref.shape) - de_ref).max():.2e}')
