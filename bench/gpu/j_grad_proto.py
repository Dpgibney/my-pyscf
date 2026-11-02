"""Prototype of the J engine's nuclear gradient (part 24), against GPU4PySCF's Rys gradient of the Coulomb energy
(grad.rhf._jk_energy_per_atom with j_factor=1, k_factor=0) at a saved density (node1; needs a library with
MD_build_j_ip1 and Rt_ip1_dot_dm, exp_prim_screen.py --j-grad on).
  1. the pairs' Hermite densities (dm_to_Rt) in a layout of one order more per pair, the top components zero;
  2. MD_build_j_ip1 over the J engine's tasks: every pair's Hermite potential one order above its density;
  3. Rt_ip1_dot_dm: the pairs' derivative Hermite densities against those potentials, summed per atom.
Usage: j_grad_proto.py --dm DM.npy [--basis def2-svp] [--repeat 3] [--out GRADS.npz]"""
import argparse
import ctypes
import math
import os
import sys
import time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy as cp  # noqa: E402
from pyscf import gto  # noqa: E402
from gpu4pyscf.lib.cupy_helper import condense  # noqa: E402
from gpu4pyscf.scf import j_engine  # noqa: E402
from gpu4pyscf.scf.j_engine import (_VHFOpt, _make_pair_qd_cond, _dm_to_Rt, _md_j_engine_quartets_scheme,  # noqa: E402
                                    _fill_sms_tilex, _scale_sp_ctr_coeff, libvhf_md, LMAX)
from gpu4pyscf.grad import rhf as rhf_grad  # noqa: E402
from gpu4pyscf.scf import jk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--out', default=None, help='save both gradients (npz: jengine, rys)')
args = ap.parse_args()

ANG_OF = 1


def herm(l):
    return [(t, u, v) for t in range(l + 1) for u in range(l + 1 - t) for v in range(l + 1 - t - u)]


def j_engine_grad(jopt, dm):
    mol = jopt.sorted_mol
    dms = jopt.apply_coeff_C_mat_CT(cp.asarray(dm)[None])
    ao_loc = mol.ao_loc
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_cutoff = math.log(jopt.direct_scf_tol)
    l_counts = np.bincount(mol._bas[:, ANG_OF])[:LMAX + 1]
    n_groups = len(l_counts)
    l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
    pair_mappings = _make_pair_qd_cond(mol, jopt.bas_pair_cache, dm_cond)
    pair_lst = []
    task_offsets = {}
    p1 = 0
    for key in ((i, j) for i in range(n_groups) for j in range(i + 1)):
        pair_lst.append(pair_mappings[key][0])
        task_offsets[key] = p1
        p1 += pair_mappings[key][0].size
    pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)
    ls = cp.asarray(mol._bas[:, ANG_OF], dtype=np.int32)
    ll = (ls[:, None] + ls).ravel()[pair_lst]
    nf3 = (ll + 1) * (ll + 2) * (ll + 3) // 6
    nf3_1 = (ll + 2) * (ll + 3) * (ll + 4) // 6
    pair_loc = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32), nf3), dtype=np.int32)
    pair_loc1 = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32), nf3_1), dtype=np.int32)
    dm_xyz = _dm_to_Rt(mol, dms, pair_lst, pair_loc, jopt.rys_envs)[0]
    # the densities in the layout of one order more per pair, class by class
    dm1 = cp.zeros(int(pair_loc1[-1]))
    for (i, j), p0 in task_offsets.items():
        n = pair_mappings[i, j][0].size
        if n == 0:
            continue
        l = i + j
        pos = cp.asarray([herm(l + 1).index(c) for c in herm(l)])
        a0, a1 = int(pair_loc[p0]), int(pair_loc[p0 + n])
        b0, b1 = int(pair_loc1[p0]), int(pair_loc1[p0 + n])
        dm1[b0:b1].reshape(n, -1)[:, pos] = dm_xyz[a0:a1].reshape(n, -1)
    vj1 = cp.zeros_like(dm1)
    n_sm = cp.cuda.Device().attributes['MultiProcessorCount']
    _env = _scale_sp_ctr_coeff(mol)
    for i in range(n_groups):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    if i == k and j < l:
                        continue
                    pair_ij_mapping, q_cond_ij, qd_ij = pair_mappings[i, j]
                    pair_kl_mapping, q_cond_kl, qd_kl = pair_mappings[k, l]
                    npairs_ij, npairs_kl = pair_ij_mapping.size, pair_kl_mapping.size
                    if npairs_ij == 0 or npairs_kl == 0:
                        continue
                    shls_slice = l_ctr_bas_loc[[i, i + 1, j, j + 1, k, k + 1, l, l + 1]]
                    scheme = _fill_sms_tilex(_md_j_engine_quartets_scheme((i + 1, j, k + 1, l)),
                                             npairs_ij, npairs_kl, (i, j) == (k, l), n_sm)
                    pair_ij_loc = pair_loc1[task_offsets[i, j]:]
                    pair_kl_loc = pair_loc1[task_offsets[k, l]:]
                    err = libvhf_md.MD_build_j_ip1(
                        ctypes.cast(vj1.data.ptr, ctypes.c_void_p), ctypes.cast(dm1.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(1), ctypes.c_int(int(pair_loc1[-1])),
                        ctypes.byref(jopt.rys_envs), (ctypes.c_int * 6)(*scheme), (ctypes.c_int * 8)(*shls_slice),
                        ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                        ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                        ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                        ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p), ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
                        ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                        ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes, ctypes.c_int(mol.nbas),
                        _env.ctypes)
                    assert err == 0, (i, j, k, l)
    grad = cp.zeros((mol.natm, 3))
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)   # held until the kernel has run
    err = libvhf_md.Rt_ip1_dot_dm(
        ctypes.cast(grad.data.ptr, ctypes.c_void_p), ctypes.cast(vj1.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p), ctypes.byref(jopt.rys_envs),
        ctypes.cast(pair_lst.data.ptr, ctypes.c_void_p), ctypes.cast(pair_loc1.data.ptr, ctypes.c_void_p),
        ctypes.c_int(pair_lst.size), ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p))
    assert err == 0
    return grad.get()


mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = np.load(args.dm)
jopt = _VHFOpt(mol, 1e-13).build()
g = j_engine_grad(jopt, dm)
ts = []
for _ in range(args.repeat):
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    g = j_engine_grad(jopt, dm)
    ts.append(time.perf_counter() - t0)
kopt = jk._VHFOpt(mol, 1e-13).build()
ref = rhf_grad._jk_energy_per_atom(kopt, dm, 1., 0.)
tr = []
for _ in range(args.repeat):
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    ref = rhf_grad._jk_energy_per_atom(kopt, dm, 1., 0.)
    cp.cuda.Device().synchronize()
    tr.append(time.perf_counter() - t0)
print(f'{args.basis}: J engine gradient {np.median(ts):.3f} s, Rys (j_factor 1, k_factor 0) {np.median(tr):.3f} s')
ratio = ref / np.where(abs(g) > 1e-8, g, np.nan)
print(f'  ratio Rys / J engine: median {np.nanmedian(ratio):.6f}, spread {np.nanmin(ratio):.6f} .. {np.nanmax(ratio):.6f}')
c = np.nanmedian(ratio)
print(f'  max |ref| {abs(ref).max():.4e}; max |ref - {c:.4f} g| {abs(ref - c * g).max():.3e}; '
      f'sum of the J engine gradient over atoms {abs((c * g).sum(axis=0)).max():.2e} (Rys {abs(ref.sum(axis=0)).max():.2e})')
if args.out:
    np.savez(args.out, jengine=c * g, rys=ref)
