"""Split a kernel's instructions into regions of its source (node1): reads Nsight Compute's source page of a report
with SourceCounters, exported as
    ncu --import REPORT.ncu-rep --page source --print-source cuda,sass --csv > SRC.csv
(the library built with -lineinfo), and sums, per kernel and region, the warp instructions executed, the FP64 warp
instructions (SASS opcodes D*: DFMA, DADD, DMUL, ...) and the warp-stall samples, which follow where the time goes. Each SASS instruction counts once, under its line in the
kernel's own file (the outermost frame of its inlining stack). For the generic kernels it also gives the output
phase's instructions per output and root, counting the iterations by the loads of a line that runs once per
iteration (MARKERS).
The regions of the Hessian's second-derivative kernels (rys_contract_jk_ip2.cu) are built in (REGIONS); other files
count as task screening (create_tasks.cu), Rys roots (rys_roots*), the g recursions (build_rys_gxyz.cuh) or other.
With --order LABELS.json (ncu_parts.py --labels of the whole stage) and --classes, as given to ncu_ejk2_class.py,
each launch is named by its class.
Usage: ncu_source_regions.py SRC.csv [--order LABELS.json --classes '(dp|ps),...']"""
import argparse
import csv
import os
from collections import defaultdict

# (first line, last line, region) in rys_contract_jk_ip2.cu
REGIONS = {
    'rys_ejk_ip2_type12_kernel': [(34, 101, 'setup, bra pair'), (102, 110, 'task screening'),
                                  (111, 140, 'setup, bra pair'), (141, 283, 'task, density products'),
                                  (284, 336, 'primitive pairs'), (337, 337, 'Rys roots'),
                                  (338, 339, 'g recursions'), (340, 570, 'outputs'), (571, 624, 'atomics')],
    'rys_ejk_ip2_type3_kernel': [(626, 693, 'setup, bra pair'), (694, 702, 'task screening'),
                                 (703, 732, 'setup, bra pair'), (733, 869, 'task, density products'),
                                 (870, 922, 'primitive pairs'), (923, 923, 'Rys roots'),
                                 (924, 925, 'g recursions'), (926, 1145, 'outputs'), (1146, 1196, 'atomics')],
}
# a line that runs once per output and root (the density product's load): its loads count the iterations
MARKERS = {'rys_ejk_ip2_type12_kernel': 364, 'rys_ejk_ip2_type3_kernel': 950}
ORDER = ['setup, bra pair', 'task screening', 'task, density products', 'primitive pairs', 'Rys roots',
         'g recursions', 'outputs', 'atomics', 'other']


def region(func, path, line):
    name = os.path.basename(path)
    if name == 'rys_contract_jk_ip2.cu':
        for a, b, r in REGIONS.get(func, []):
            if a <= line <= b:
                return r
        return 'other'
    if name == 'create_tasks.cu':
        return 'task screening'
    if name.startswith('rys_roots'):
        return 'Rys roots'
    if name == 'build_rys_gxyz.cuh':
        return 'g recursions'
    return 'other'


def num(x):
    '''A metric cell: "" or "-" for none'''
    return float(x) if x not in ('', '-') else 0.


def parse(path):
    '''The launches of the source page, in order: [(kernel, {region: [instructions, FP64, stall samples]},
    iterations)]. A new launch starts where the kernel name changes (the profiled launches alternate type12 and
    type3, or are unrolled kernels of their own names). Each SASS instruction is listed under every source line of
    its inlining stack; it is counted once, under its line in the kernel's own file (the outermost frame) when it
    has one.'''
    launches = []
    cur_file = cur_func = cur_line = None
    cols = None
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == 'File Path':
                cur_file = row[1]
                continue
            if row[0] == 'Function Name':
                func = row[1].split('(')[0]
                if func != cur_func:
                    cur_func = func
                    launches.append((func, defaultdict(list), {}))
                continue
            if row[0] == 'Line No':
                cols = {}
                for n, c in enumerate(row):   # two "Source" columns (CUDA, SASS): keep the first
                    cols.setdefault(c, n)
                continue
            if cols is None or not launches:
                continue
            if row[0]:          # a CUDA source line; its SASS rows follow
                cur_line = int(row[0])
                continue
            sass = row[3].split()
            if not sass:
                continue
            op = sass[1] if sass[0].startswith('@') else sass[0]
            _, lines, data = launches[-1]
            lines[row[2]].append((cur_file, cur_line))
            data[row[2]] = (op, num(row[cols['Instructions Executed']]),
                            num(row[cols['Warp Stall Sampling (All Samples)']]))
    out = []
    for func, lines, data in launches:
        reg = defaultdict(lambda: [0., 0., 0.])
        iters = 0.
        for addr, where in lines.items():
            own = [w for w in where if os.path.basename(w[0]) == 'rys_contract_jk_ip2.cu']
            file, line = (own or where)[0]
            r = region(func, file, line)
            op, n, st = data[addr]
            if own and line == MARKERS.get(func) and op.startswith('LD'):
                iters += n
            reg[r][0] += n
            reg[r][2] += st
            if op.split('.')[0] in ('DFMA', 'DADD', 'DMUL', 'DSETP', 'DMNMX'):
                reg[r][1] += n
        out.append((func, reg, iters))
    return out


def report(func, reg, iters, label=''):
    ti, tf, ts = (sum(v[k] for v in reg.values()) for k in range(3))
    print(f'{label}{func}: {ti / 1e9:.3f} G warp instructions, {tf / 1e9:.3f} G FP64 ({100 * tf / max(ti, 1):.0f}%), '
          f'{ts:.0f} stall samples')
    if iters:
        print(f'  per output and root (warp level): {reg["outputs"][0] / iters:.1f} instructions, '
              f'{reg["outputs"][1] / iters:.1f} FP64; {iters / 1e6:.2f} M iterations')
    print(f'  {"region":24s} {"inst":>7s} {"FP64":>7s} {"stall":>7s}  FP64 share of the region')
    for r in ORDER:
        if r in reg:
            i, f, st = reg[r]
            print(f'  {r:24s} {100 * i / max(ti, 1):6.1f}% {100 * f / max(tf, 1):6.1f}% {100 * st / max(ts, 1):6.1f}%  '
                  f'{100 * f / max(i, 1):5.1f}%')


def regions_by_launch(path, order):
    '''{(class, kind): {'fp64_out', 'iters', 'stall_out', 'stall'}} for the launches of path, which ran in the
    order given (a list of (class, kind))'''
    launches = parse(path)
    assert len(launches) == len(order), (len(launches), len(order))
    out = {}
    for (func, reg, iters), (c, kind) in zip(launches, order):
        assert ('type12' in func) == (kind == 't12'), (func, kind)
        out[c, kind] = {'fp64_out': reg['outputs'][1], 'iters': iters, 'stall_out': reg['outputs'][2],
                        'stall': sum(v[2] for v in reg.values()), 'func': func}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--order', default=None, help='launch labels (ncu_parts.py --labels) for the class names; the '
                    'profiled launches are the first of each class and kernel among them')
    ap.add_argument('--classes', default=None, help='the classes profiled (ncu_ejk2_class.py --classes)')
    args = ap.parse_args()
    launches = parse(args.csv)
    labels = launch_order(args.order, args.classes) if args.order else [None] * len(launches)
    for (func, reg, iters), lab in zip(launches, labels):
        report(func, reg, iters, f'{lab[0]} {lab[1]}: ' if lab else '')


def launch_order(labels_json, classes):
    '''The order of the profiled launches: the first call of each listed class and kernel in the stage's order'''
    import json
    targets = set(classes.split(','))
    seen = []
    for x in json.load(open(labels_json)):
        c, kind = x['class'].split()
        if c in targets and (c, kind) not in seen:
            seen.append((c, kind))
    return seen


if __name__ == '__main__':
    main()
