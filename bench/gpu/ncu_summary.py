"""Summarize an Nsight Compute CSV (--csv --page raw --print-units base) from ncu_parts.py: per angular-momentum class
(integral kernels, paired in launch order with the --labels file) or per kernel name (all others): launches, GPU
time, FP64 flops (2*DFMA + DADD + DMUL), FP64-pipe share of warp instructions, FP64 pipe utilization, SIMT
efficiency (active threads per warp instruction / 32), occupancy, global atomics/reductions and DRAM traffic.
Instruction and atomic counts follow from the algorithm and carry over between GPUs; times, utilizations and DRAM
bytes (cache sizes) are specific to this card.
Usage: ncu_summary.py NCU.csv [--labels LABELS.json] [--kernel-prefix rys_k_] [--top 25]"""
import argparse
import csv
import json
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--labels', default=None)
ap.add_argument('--kernel-prefix', default=None, help='integral kernels, paired with the labels in launch order')
ap.add_argument('--top', type=int, default=25)
args = ap.parse_args()

rows = [r for r in csv.reader(open(args.csv)) if r and not r[0].startswith('==')]
hdr = rows[0]
col = {h: k for k, h in enumerate(hdr)}


def val(r, m):
    s = r[col[m]].replace(',', '')
    try:
        return float(s)
    except ValueError:
        return 0.


labels = json.load(open(args.labels)) if args.labels else []
groups = defaultdict(lambda: defaultdict(float))
n_int = 0
for r in rows[2:]:
    name = re.sub(r'\(.*', '', r[col['Kernel Name']]).replace('void ', '')
    if args.kernel_prefix and name.startswith(args.kernel_prefix):
        key = labels[n_int]['class'] if n_int < len(labels) else name
        n_int += 1
    else:
        key = name[:40]
    g = groups[key]
    t = val(r, 'gpu__time_duration.sum') * 1e-9
    g['launches'] += 1
    g['t'] += t
    g['flop64'] += (2 * val(r, 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum')
                    + val(r, 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum')
                    + val(r, 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum'))
    g['flop32'] += (2 * val(r, 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum')
                    + val(r, 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum')
                    + val(r, 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'))
    g['inst'] += val(r, 'smsp__inst_executed.sum')
    g['inst64'] += val(r, 'smsp__inst_executed_pipe_fp64.sum')
    g['atom'] += val(r, 'smsp__sass_inst_executed_op_global_atom.sum') + val(r, 'smsp__sass_inst_executed_op_global_red.sum')
    g['dram'] += val(r, 'dram__bytes_read.sum') + val(r, 'dram__bytes_write.sum')
    # time-weighted averages
    g['fp64util_t'] += val(r, 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active') * t
    g['simt_t'] += val(r, 'smsp__thread_inst_executed_per_inst_executed.ratio') / 32 * t
    g['occ_t'] += val(r, 'sm__warps_active.avg.pct_of_peak_sustained_active') * t
    g['regs'] = max(g['regs'], val(r, 'launch__registers_per_thread'))
if args.labels and n_int != len(labels):
    print(f'WARNING: {n_int} integral kernels but {len(labels)} labels; classes may be misassigned')

tot = {k: sum(g[k] for g in groups.values()) for k in ('t', 'flop64', 'inst', 'atom', 'dram')}
print(f"total: {sum(g['launches'] for g in groups.values()):.0f} launches, {tot['t']:.3f} s GPU, "
      f"{tot['flop64'] / 1e9:.1f} GFLOP FP64 ({tot['flop64'] / tot['t'] / 1e9:.1f} GFLOP/s), "
      f"{tot['inst'] / 1e9:.2f} G warp instructions, {tot['atom'] / 1e6:.2f} M atomic instructions, "
      f"{tot['dram'] / 1e9:.2f} GB DRAM")
print(f"{'class/kernel':40s} {'n':>4s} {'time s':>7s} {'%t':>5s} {'GFLOP64':>8s} {'%fl':>5s} {'fp64 %inst':>10s} "
      f"{'fp64 util':>9s} {'simt':>5s} {'occ':>5s} {'regs':>4s} {'M atom':>7s} {'MB dram':>8s}")
for key, g in sorted(groups.items(), key=lambda kv: -kv[1]['t'])[:args.top]:
    t = g['t'] or 1e-30
    print(f"{key:40s} {g['launches']:4.0f} {g['t']:7.3f} {100 * g['t'] / tot['t']:5.1f} {g['flop64'] / 1e9:8.2f} "
          f"{100 * g['flop64'] / max(tot['flop64'], 1):5.1f} {100 * g['inst64'] / max(g['inst'], 1):10.1f} "
          f"{g['fp64util_t'] / t:9.1f} {g['simt_t'] / t:5.2f} {g['occ_t'] / t:5.1f} {g['regs']:4.0f} "
          f"{g['atom'] / 1e6:7.2f} {g['dram'] / 1e6:8.1f}")
