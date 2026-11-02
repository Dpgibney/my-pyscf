"""Summarize an Nsight Compute CSV (--csv --page raw --print-units base) from ncu_parts.py: per angular-momentum class
(integral kernels, paired in launch order with the --labels file) or per kernel name (all others): launches, GPU
time, FP64 flops (2*DFMA + DADD + DMUL), FP64-pipe share of warp instructions, FP64 pipe utilization, SIMT
efficiency (active threads per warp instruction / 32), occupancy, global atomics/reductions and DRAM traffic.
Instruction and atomic counts follow from the algorithm and carry over between GPUs; times, utilizations and DRAM
bytes (cache sizes) are specific to this card.
With --ref REF.csv (and --ref-labels) it also prints, per class, the time, FP64 flops and warp instructions of
REF next to those of NCU.csv.
With --launches it instead prints one row per integral-kernel launch: class, kernel, block and grid size, time, FP64
pipe busy share, SIMT efficiency, occupancy, FMA share of the FP64 instructions and, when the CSV has them, the
largest warp-stall reasons. A grid of fewer blocks than the card has SMs leaves SMs idle (the FP64 busy share counts
active SMs only).
With --gemms it instead prints the BLAS GEMM launches grouped by kernel and grid size: launches, FP64 flops per
launch, time and flop rate. The flops count every DFMA a kernel executes, the tiles' padding included, so
flops per launch that match 2·M_pad·N_pad·K for the kernel's tile (e.g. 128x64) show how much of a small GEMM
is padding.
With --l2 it instead prints the L2 and stall table of a CSV with those metrics (scratch/ncu_cphf2.sh's first pass):
the L2 atomic unit's input busy share, L2 tag and crossbar throughput, DRAM throughput, L2 requests per atomic warp
instruction and sectors per request, the L2 hit rates of atomics and reads, the L1 hit rate of global loads, and the
largest warp-stall reasons (stalled cycles per active warp cycle); time-weighted per class.
A CSV may be gzipped (NCU.csv.gz).
Usage: ncu_summary.py NCU.csv [--labels LABELS.json] [--kernel-prefix rys_k_] [--top 25]
                      [--ref REF.csv --ref-labels REF_LABELS.json] [--l2] [--launches] [--gemms]"""
import argparse
import csv
import gzip
import json
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--labels', default=None)
ap.add_argument('--kernel-prefix', default=None, help='integral kernels, paired with the labels in launch order')
ap.add_argument('--top', type=int, default=25)
ap.add_argument('--ref', default=None)
ap.add_argument('--ref-labels', default=None)
ap.add_argument('--l2', action='store_true')
ap.add_argument('--launches', action='store_true')
ap.add_argument('--gemms', action='store_true')
args = ap.parse_args()


def read_rows(path):
    '''The CSV's rows without ncu's ==PROF== lines'''
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith('==')]


def summarize(path, labels_path):
    '''Per class (integral kernels) or kernel name (others): sums and time-weighted averages of the metrics'''
    rows = read_rows(path)
    col = {h: k for k, h in enumerate(rows[0])}

    def val(r, m):
        s = r[col[m]].replace(',', '')
        try:
            return float(s)
        except ValueError:
            return 0.

    labels = json.load(open(labels_path)) if labels_path else []
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
    if labels_path and n_int != len(labels):
        print(f'WARNING: {n_int} integral kernels but {len(labels)} labels; classes may be misassigned')
    return groups


L2_SUMS = ('lts__t_requests_op_atom.sum', 'lts__t_sectors_op_atom.sum', 'lts__t_sectors_op_atom_lookup_hit.sum',
           'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_read_lookup_hit.sum',
           'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum',
           'smsp__sass_inst_executed_op_global_atom.sum')
L2_PCTS = (('atom unit', 'lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed'),
           ('L2 tag', 'lts__t_sectors.avg.pct_of_peak_sustained_elapsed'),
           ('xbar>L2', 'lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed'),
           ('DRAM', 'dram__throughput.avg.pct_of_peak_sustained_elapsed'))


def l2_table(path, labels_path):
    '''The L2 and stall table (--l2), per class as in summarize'''
    rows = read_rows(path)
    col = {h: k for k, h in enumerate(rows[0])}
    stalls = [h for h in rows[0] if h.startswith('smsp__warp_issue_stalled_') and h.endswith('per_warp_active.pct')]

    def val(r, m):
        try:
            return float(r[col[m]].replace(',', ''))
        except (KeyError, ValueError):
            return 0.
    labels = json.load(open(labels_path)) if labels_path else []
    groups = defaultdict(lambda: defaultdict(float))
    n_int = 0
    for r in rows[2:]:
        name = re.sub(r'\(.*', '', r[col['Kernel Name']]).replace('void ', '')
        if not (args.kernel_prefix and name.startswith(args.kernel_prefix)):
            continue
        key = labels[n_int]['class'] if n_int < len(labels) else name
        n_int += 1
        t = val(r, 'gpu__time_duration.sum') * 1e-9
        for g in (groups[key], groups['all']):
            g['t'] += t
            for m in L2_SUMS:
                g[m] += val(r, m)
            for _, m in L2_PCTS:
                g[m] += val(r, m) * t
            for m in stalls:
                g[m] += val(r, m) * t
    tot = groups['all']
    top = sorted(stalls, key=lambda m: -tot[m])[:5]
    short = [m.replace('smsp__warp_issue_stalled_', '').replace('_per_warp_active.pct', '')[:10] for m in top]
    print(f"{'class/kernel':12s} {'time s':>7s} " + ' '.join(f'{c + " %":>10s}' for c, _ in L2_PCTS) +
          f" {'req/atom':>8s} {'sec/req':>7s} {'L2 atom hit':>11s} {'L2 rd hit':>9s} {'L1 ld hit':>9s} |" +
          ' '.join(f'{c:>10s}' for c in short) + '  (stalled % of active warp cycles)')
    order = sorted((k for k in groups if k != 'all'), key=lambda k: -groups[k]['t'])[:args.top] + ['all']
    for k in order:
        g = groups[k]
        t = g['t'] or 1e-30

        def ratio(a, b):
            return g[a] / max(g[b], 1)
        print(f"{k:12s} {g['t']:7.3f} " + ' '.join(f'{g[m] / t:10.1f}' for _, m in L2_PCTS) +
              f" {ratio('lts__t_requests_op_atom.sum', 'smsp__sass_inst_executed_op_global_atom.sum'):8.2f}"
              f" {ratio('lts__t_sectors_op_atom.sum', 'lts__t_requests_op_atom.sum'):7.2f}"
              f" {100 * ratio('lts__t_sectors_op_atom_lookup_hit.sum', 'lts__t_sectors_op_atom.sum'):11.1f}"
              f" {100 * ratio('lts__t_sectors_op_read_lookup_hit.sum', 'lts__t_sectors_op_read.sum'):9.1f}"
              f" {100 * ratio('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'):9.1f}"
              " |" + ' '.join(f'{g[m] / t:10.1f}' for m in top))


def launch_table(path, labels_path):
    '''One row per integral-kernel launch (--launches), the longest first'''
    rows = read_rows(path)
    col = {h: k for k, h in enumerate(rows[0])}
    stalls = [h for h in rows[0] if h.startswith('smsp__warp_issue_stalled_') and h.endswith('per_warp_active.pct')]

    def val(r, m):
        try:
            return float(r[col[m]].replace(',', ''))
        except (KeyError, ValueError):
            return 0.
    labels = json.load(open(labels_path)) if labels_path else []
    out = []
    n_int = 0
    for r in rows[2:]:
        name = re.sub(r'\(.*', '', r[col['Kernel Name']]).replace('void ', '')
        if not (args.kernel_prefix and name.startswith(args.kernel_prefix)):
            continue
        key = labels[n_int]['class'] if n_int < len(labels) else ''
        n_int += 1
        f = [val(r, f'smsp__sass_thread_inst_executed_op_{o}_pred_on.sum') for o in ('dfma', 'dadd', 'dmul')]
        out.append((val(r, 'gpu__time_duration.sum') * 1e-9, key, name, val(r, 'launch__block_size'),
                    val(r, 'launch__grid_size'), val(r, 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active'),
                    val(r, 'smsp__thread_inst_executed_per_inst_executed.ratio') / 32,
                    val(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'), 100 * f[0] / max(sum(f), 1),
                    [(m, val(r, m)) for m in stalls]))
    tot = sum(x[0] for x in out)
    top = sorted(stalls, key=lambda m: -sum(dict(x[9])[m] * x[0] for x in out))[:4]
    short = [m.replace('smsp__warp_issue_stalled_', '').replace('_per_warp_active.pct', '')[:10] for m in top]
    print(f'{len(out)} launches, {tot:.3f} s')
    print(f"{'class':10s} {'kernel':22s} {'block':>5s} {'grid':>6s} {'ms':>8s} {'%t':>5s} {'fp64 %':>6s} {'simt':>5s} "
          f"{'occ':>5s} {'FMA %':>5s}" + (' |' + ' '.join(f'{c:>10s}' for c in short) if top else ''))
    for t, key, name, block, grid, fp64, simt, occ, fma, st in sorted(out, key=lambda x: -x[0])[:args.top]:
        d = dict(st)
        print(f'{key:10s} {name[:22]:22s} {block:5.0f} {grid:6.0f} {t * 1e3:8.2f} {100 * t / tot:5.1f} {fp64:6.1f} '
              f'{simt:5.2f} {occ:5.1f} {fma:5.1f}' + (' |' + ' '.join(f'{d[m]:10.1f}' for m in top) if top else ''))


def gemm_table(path):
    '''The GEMM launches (--gemms) by kernel and grid size'''
    rows = read_rows(path)
    col = {h: k for k, h in enumerate(rows[0])}

    def val(r, m):
        try:
            return float(r[col[m]].replace(',', ''))
        except ValueError:
            return 0.

    groups = defaultdict(list)
    for r in rows[2:]:
        name = re.sub(r'\(.*', '', r[col['Kernel Name']]).replace('void ', '')
        if 'gemm' not in name:
            continue
        flop = (2 * val(r, 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum')
                + val(r, 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum')
                + val(r, 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum'))
        groups[(name[:40], int(val(r, 'launch__grid_size')))].append((flop, val(r, 'gpu__time_duration.sum') * 1e-9))
    print(f"{'kernel':40s} {'grid':>5s} {'n':>4s} {'GFLOP/launch min':>16s} {'mean':>6s} {'max':>6s} {'time s':>7s} "
          f"{'GFLOP/s':>7s}")
    for (name, grid), xs in sorted(groups.items()):
        fl = [x[0] / 1e9 for x in xs]
        t = sum(x[1] for x in xs)
        print(f"{name:40s} {grid:5d} {len(xs):4d} {min(fl):16.3f} {sum(fl) / len(fl):6.3f} {max(fl):6.3f} {t:7.3f} "
              f"{sum(fl) / t:7.1f}")


if args.gemms:
    gemm_table(args.csv)
    raise SystemExit
if args.launches:
    launch_table(args.csv, args.labels)
    raise SystemExit
if args.l2:
    l2_table(args.csv, args.labels)
    raise SystemExit
groups = summarize(args.csv, args.labels)
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

if args.ref:
    ref = summarize(args.ref, args.ref_labels)
    print(f"\nper class, {args.ref} -> {args.csv}")
    print(f"{'class/kernel':40s} {'time s':>15s} {'GFLOP64':>15s} {'G warp inst':>15s}")
    keys = sorted(set(ref) | set(groups), key=lambda k: -ref.get(k, {}).get('t', 0))
    for key in keys[:args.top]:
        a, b = ref.get(key, {}), groups.get(key, {})
        print(f"{key:40s} {a.get('t', 0):6.3f} -> {b.get('t', 0):6.3f} {a.get('flop64', 0) / 1e9:6.2f} -> "
              f"{b.get('flop64', 0) / 1e9:6.2f} {a.get('inst', 0) / 1e9:6.3f} -> {b.get('inst', 0) / 1e9:6.3f}")
    ta, tb = (sum(g['t'] for g in x.values()) for x in (ref, groups))
    print(f"{'total':40s} {ta:6.3f} -> {tb:6.3f}")
