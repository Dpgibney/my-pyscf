"""Per-class comparison of two Nsight Compute K-build profiles (ncu_parts.py, the same class labels): time, FP64
flops (thread level, so padding lanes count), warp instructions, FP64 warp instructions and their share, FP64 pipe
utilization and registers, with the kernel each class ran in A and B. Classes are sorted by A's time. --generic then lists the classes that B still runs in the
generic kernel ({prefix}kernel), with their outputs per thread at 8 threads per quartet.
Usage: ncu_compare.py A.csv A_labels.json B.csv B_labels.json [--prefix rys_k_] [--top 30] [--generic]"""
import argparse
import csv
import json
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('a_csv')
ap.add_argument('a_labels')
ap.add_argument('b_csv')
ap.add_argument('b_labels')
ap.add_argument('--prefix', default='rys_k_')
ap.add_argument('--top', type=int, default=30)
ap.add_argument('--generic', action='store_true')
args = ap.parse_args()


def load(path, labels_path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith('==')]
    col = {h: i for i, h in enumerate(rows[0])}
    labels = json.load(open(labels_path))

    def v(r, m):
        try:
            return float(r[col[m]].replace(',', ''))
        except ValueError:
            return 0.
    agg = defaultdict(lambda: defaultdict(float))
    kern = defaultdict(set)
    n = 0
    for r in rows[2:]:
        name = re.sub(r'\(.*', '', r[col['Kernel Name']]).replace('void ', '')
        if not name.startswith(args.prefix):
            continue
        c = labels[n]['class']
        n += 1
        g = agg[c]
        t = v(r, 'gpu__time_duration.sum') * 1e-9
        g['t'] += t
        kern[c].add(name)
        g['fl'] += (2 * v(r, 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum')
                    + v(r, 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum')
                    + v(r, 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum'))
        g['inst'] += v(r, 'smsp__inst_executed.sum')
        g['i64'] += v(r, 'smsp__inst_executed_pipe_fp64.sum')
        g['pipe'] += v(r, 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active') * t
        g['regs'] = max(g['regs'], v(r, 'launch__registers_per_thread'))
    return agg, kern


a, ka = load(args.a_csv, args.a_labels)
b, kb = load(args.b_csv, args.b_labels)
print(f"{'class':9s} {'kernel A -> B':34s} {'time A -> B':>22s} {'GFLOP A/B':>13s} {'G inst A/B':>12s} "
      f"{'G fp64 A/B':>12s} {'fp64 %inst':>11s} {'pipe %':>11s} {'regs':>8s}")
ta = tb = 0
for c in sorted(a, key=lambda c: -a[c]['t'])[:args.top]:
    x, y = a[c], b.get(c, defaultdict(float))
    ta += x['t']
    tb += y['t']
    kn = f"{'/'.join(sorted(ka[c]))[:15]} -> {'/'.join(sorted(kb.get(c, [])))[:15]}"
    print(f"{c:9s} {kn:34s} {x['t']:7.3f} -> {y['t']:6.3f} ({x['t'] / max(y['t'], 1e-12):4.2f}x) "
          f"{x['fl'] / 1e9:5.1f}/{y['fl'] / 1e9:5.1f} {x['inst'] / 1e9:5.2f}/{y['inst'] / 1e9:5.2f} "
          f"{x['i64'] / 1e9:5.3f}/{y['i64'] / 1e9:5.3f} "
          f"{100 * x['i64'] / max(x['inst'], 1):4.1f}/{100 * y['i64'] / max(y['inst'], 1):4.1f} "
          f"{x['pipe'] / max(x['t'], 1e-12):4.1f}/{y['pipe'] / max(y['t'], 1e-12):4.1f} {x['regs']:3.0f}/{y['regs']:3.0f}")
TA = sum(g['t'] for g in a.values())
TB = sum(g['t'] for g in b.values())
print(f'listed classes: {ta:.3f} -> {tb:.3f} s; all integral kernels: {TA:.3f} -> {TB:.3f} s ({TA / TB:.3f}x)')
if args.generic:
    generic = {c: g['t'] for c, g in b.items() if kb[c] == {args.prefix + 'kernel'}}
    print(f'still generic in B: {sum(generic.values()):.3f} of {TB:.3f} s ({100 * sum(generic.values()) / TB:.0f}%)')
    for c, t in sorted(generic.items(), key=lambda kv: -kv[1]):
        nout = 1
        for ch in c:
            if ch in 'spdfg':
                l = 'spdfg'.index(ch)
                nout *= (l + 1) * (l + 2) // 2
        print(f'  {c}  {t:6.3f} s  {nout:5d} outputs ({-(-nout // 8)} per thread at 8 threads per quartet)')
