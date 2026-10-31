"""How many ket pairs (tasks) each bra pair gets in GPU4PySCF's K build, and how many thread slots the kernels pad.
Replays the screening of _fill_vk_tasks (gvhf-rys/create_tasks.cu) on the host for every shell-group quartet, at a
saved density (jk_at_dm.py --save-dm). A block takes one bra pair and runs its ntasks ket pairs nsq_per_block at a
time, so the last pass leaves nsq_per_block - (ntasks mod nsq_per_block) slots idle. Reported per class, for the
block shape each kernel launches with (stride-1 unrolled: 256 or 512 quartets; multi-thread unrolled: 64 quartets
x 4 threads; generic rys_k_kernel: threads_scheme_for_k):
  slots      sum over bra pairs of ceil(ntasks / nsq) * nsq
  pad        idle share of those slots
  idle warps share of warps with no real quartet (they still run the full loops in kernels with barriers)
With --model 256,128,64,32 it also gives the stride-1 kernels' idle warp time for those block sizes.
With --skip it counts the primitive quartets the patched kernels compute, against those a per-quartet skip would
leave: the multi-thread and generic kernels skip only bra pairs below e^-60, block-wide, and run every padding slot
and every ket primitive pair.
Usage: task_census.py --dm DM.npy [--basis def2-svp] [--mol caffeine] [--nsq1 256] [--model 256,128,64] [--skip]"""
import argparse
import math
import os
import sys
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from gpu4pyscf.scf import jk  # noqa: E402
from gpu4pyscf.lib.cupy_helper import condense  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--nsq1', type=int, default=256, help='block size of the stride-1 kernels (512 if <= 128 registers)')
ap.add_argument('--model', default=None, help='comma-separated block sizes: padding of the stride-1 kernels at each')
ap.add_argument('--skip', action='store_true', help='primitive-quartet work of the multi-thread and generic '
                'kernels that per-quartet skips (no padding, ket pairs below e^-60) would remove')
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
opt = jk._VHFOpt(mol).build()
smol = opt.sorted_mol
nbas = smol.nbas
dms = opt.apply_coeff_C_mat_CT(cupy.asarray(numpy.load(args.dm))[None])
dm_cond = cupy.log(condense('absmax', dms, smol.ao_loc) + 1e-300).astype(numpy.float32).get()
dm_penalty = float(dm_cond.max())
log_cutoff = numpy.float32(math.log(opt.direct_scf_tol))
uniq_l = smol.uniq_l_ctr[:, 0]
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
symb = [lib.param.ANGULAR[l] for l in uniq_l]
ngroups = numpy.count_nonzero(uniq_l <= jk.LMAX)
cache = {k: [cupy.asnumpy(x) for x in v] for k, v in opt.bas_pair_cache.items()}
STRIDE1 = {0, 125, 130, 131, 150, 155, 156, 250, 255, 256, 260, 275, 280, 300, 375, 380, 385, 400, 425}
MULTI = {261, 281, 285, 305, 381, 405}
SHM_SIZE = jk.SHM_SIZE


def generic_nsq(li, lj, lk, ll, ijprim):
    '''nsq_per_block of threads_scheme_for_k (first launch, tile_chunk <= 256)'''
    nt = [((l + 1) * (l + 2) // 2 + 2) // 3 for l in (li, lj, lk, ll)]
    cart_idx_size = sum(nt) * 9
    g_size = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1)
    nroots = (li + lj + lk + ll) // 2 + 1
    unit = nroots * 2 + g_size * 3 + 6
    counts = (SHM_SIZE - cart_idx_size * 4 - ijprim * 8) // (unit * 8)
    gout_stride = min(nt[0] * nt[1] * nt[2] * nt[3], 256)
    nsq = min(counts, 256 // gout_stride)
    return (nsq & ~7 if nsq > 8 else nsq), gout_stride


shell_r = smol.atom_coords()[smol._bas[:, 0]]
grp_exp = [numpy.array([smol.bas_exp(b) for b in range(loc[g], loc[g + 1])]) for g in range(ngroups)]
grp_cof = [numpy.array([abs(smol._libcint_ctr_coeff(b)[:, 0]) for b in range(loc[g], loc[g + 1])])
           for g in range(ngroups)]


def kept_prim_pairs(pairs, gi, gj):
    '''primitive pairs per shell pair with |c_i c_j exp(-theta R^2)| >= e^-60, the kernels' PRIM_CUTOFF'''
    ish, jsh = pairs // nbas, pairs % nbas
    ai, aj = grp_exp[gi][ish - loc[gi]], grp_exp[gj][jsh - loc[gj]]
    ci, cj = grp_cof[gi][ish - loc[gi]], grp_cof[gj][jsh - loc[gj]]
    rr = ((shell_r[ish] - shell_r[jsh])**2).sum(axis=1)
    theta = ai[:, :, None] * aj[:, None, :] / (ai[:, :, None] + aj[:, None, :])
    x = theta * rr[:, None, None] - numpy.log(ci[:, :, None] * cj[:, None, :])
    return (x <= 60).sum(axis=(1, 2))


rows = {}
for i in range(ngroups):
    for j in range(i + 1):
        for k in range(i + 1):
            for l in range(k + 1):
                if (i, j) not in cache or (k, l) not in cache:
                    continue
                pij, qij = cache[i, j][0].astype(numpy.int64), cache[i, j][1]
                pkl, qkl = cache[k, l][0].astype(numpy.int64), cache[k, l][1]
                if pij.size == 0 or pkl.size == 0:
                    continue
                li, lj, lk, ll = (int(uniq_l[g]) for g in (i, j, k, l))
                ijkl = li * 125 + lj * 25 + lk * 5 + ll
                ish, jsh = pij // nbas, pij % nbas
                ksh, lsh = pkl // nbas, pkl % nbas
                kl_cutoff = (log_cutoff - qij)[:, None]
                d_cutoff = kl_cutoff - qkl[None, :]
                keep = (qkl[None, :] + dm_penalty >= kl_cutoff) & (pij[:, None] >= pkl[None, :])
                keep &= ((dm_cond[ish[:, None], ksh[None, :]] > d_cutoff) |
                         (dm_cond[jsh[:, None], ksh[None, :]] > d_cutoff) |
                         (dm_cond[ish[:, None], lsh[None, :]] > d_cutoff) |
                         (dm_cond[jsh[:, None], lsh[None, :]] > d_cutoff))
                ntasks = keep.sum(axis=1)
                ntasks = ntasks[ntasks > 0]
                ijprim = smol.bas_nprim(loc[i]) * smol.bas_nprim(loc[j])
                if ijkl in MULTI:
                    kind, nsq, gs = 'multi', 64, 4
                elif ijkl in STRIDE1:
                    kind, nsq, gs = 'stride1', args.nsq1, 1
                else:
                    kind = 'generic'
                    nsq, gs = generic_nsq(li, lj, lk, ll, ijprim)
                passes = -(-ntasks // nsq)
                last = ntasks - (passes - 1) * nsq
                # warps hold 32 consecutive sq_id of one gout_id (t_id = nsq*gout_id + sq_id)
                warps_per_pass = max(nsq * gs // 32, 1)
                idle_last = gs * ((nsq - last) // 32) if nsq >= 32 else 0 * last
                key = f'({symb[i]}{symb[j]}|{symb[k]}{symb[l]})'
                r = rows.setdefault(key, dict(kind=kind, nsq=nsq, gs=gs, bra=0, tasks=0, slots=0, warps=0,
                                              idle=0, quartets=0))
                r['bra'] += ntasks.size
                r['tasks'] += int(ntasks.sum())
                r['slots'] += int((passes * nsq).sum())
                r['warps'] += int((passes * warps_per_pass).sum())
                r['idle'] += int(idle_last.sum())
                r['quartets'] += int(ntasks.sum()) * ijprim * smol.bas_nprim(loc[k]) * smol.bas_nprim(loc[l])
                r.setdefault('ntasks', []).append(ntasks)
                if args.skip:
                    nb = kept_prim_pairs(pij, i, j)
                    nk = kept_prim_pairs(pkl, k, l)
                    nt_all = keep.sum(axis=1)
                    klprim = smol.bas_nprim(loc[k]) * smol.bas_nprim(loc[l])
                    ideal = float((nb * (keep * nk[None, :]).sum(axis=1)).sum())
                    if kind == 'stride1':
                        now = ideal
                    else:
                        slots_ij = -(-nt_all // nsq) * nsq
                        now = float((slots_ij * nb).sum()) * klprim
                    r['work_now'] = r.get('work_now', 0.) + now
                    r['work_ideal'] = r.get('work_ideal', 0.) + ideal

print(f'{args.mol} {args.basis}: nbas {nbas}, nao {smol.nao}, log cutoff {log_cutoff:.2f}, dm_penalty {dm_penalty:.2f}')
print(f"{'class':10s} {'kernel':8s} {'nsq x gs':>8s} {'bra pairs':>9s} {'tasks/bra':>9s} {'pad':>6s} "
      f"{'idle warps':>10s} {'prim quartets':>13s}")
tot = {}
for key, r in sorted(rows.items(), key=lambda kv: -kv[1]['quartets']):
    print(f"{key:10s} {r['kind']:8s} {r['nsq']:4d}x{r['gs']:<3d} {r['bra']:9d} {r['tasks'] / max(r['bra'], 1):9.1f} "
          f"{1 - r['tasks'] / max(r['slots'], 1):6.1%} {r['idle'] / max(r['warps'], 1):10.1%} {r['quartets']:13d}")
    t = tot.setdefault(r['kind'], [0, 0, 0, 0])
    t[0] += r['tasks']; t[1] += r['slots']; t[2] += r['idle']; t[3] += r['warps']
for kind, t in tot.items():
    print(f'{kind}: pad {1 - t[0] / t[1]:.1%} of slots, {t[2] / t[3]:.1%} idle warps')

if args.model:
    sizes = [int(x) for x in args.model.split(',')]
    # A block of n quartets (n/32 warps) spends ceil(ntasks/n) passes on a bra pair; with no barriers inside, its idle
    # share of the warp time is the padded share of its slots. n = 32 is one warp per bra pair.
    print(f"stride-1 kernels, idle share of the warp time for blocks of {sizes} quartets:")
    for key, r in sorted(rows.items(), key=lambda kv: -kv[1]['quartets']):
        if r['kind'] != 'stride1':
            continue
        nt = numpy.concatenate(r['ntasks'])
        pads = [1 - nt.sum() / (-(-nt // n) * n).sum() for n in sizes]
        print(f"{key:10s} " + " ".join(f"{x:6.1%}" for x in pads))
    nt_all = [numpy.concatenate(r['ntasks']) for r in rows.values() if r['kind'] == 'stride1']
    w_all = [r['quartets'] / max(r['tasks'], 1) for r in rows.values() if r['kind'] == 'stride1']
    for n in sizes:
        # weight each class's slots by its primitive quartets per task (work per slot)
        used = sum(nt.sum() * w for nt, w in zip(nt_all, w_all))
        slots = sum((-(-nt // n) * n).sum() * w for nt, w in zip(nt_all, w_all))
        print(f"  {n:4d} quartets per block: {1 - used / slots:.1%} idle, weighted by primitive quartets per task")

if args.skip:
    print('primitive quartets computed now, and the share a per-quartet skip would remove:')
    tot = {}
    for key, r in sorted(rows.items(), key=lambda kv: -kv[1]['work_now']):
        if r['kind'] == 'stride1':
            continue
        print(f"{key:10s} {r['kind']:8s} {r['work_now']:14.0f} {1 - r['work_ideal'] / max(r['work_now'], 1):7.1%}")
        t = tot.setdefault(r['kind'], [0., 0.])
        t[0] += r['work_now']; t[1] += r['work_ideal']
    for kind, t in tot.items():
        print(f'{kind}: {1 - t[1] / t[0]:.1%} of the primitive quartets avoidable')
