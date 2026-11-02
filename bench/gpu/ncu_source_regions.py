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
With --loops TEXT it instead splits each launch into the loops whose header contains TEXT (found by brace matching
in the --kernel-files sources) and the rest, by instruction class (atomic adds, global loads, shared memory, FP64,
other), with the stall reasons and the global sectors requested against the coalesced count: e.g.
--loops 'for (int i_dm' for the density loop of the J+K kernels.
Usage: ncu_source_regions.py SRC.csv [--order LABELS.json --classes '(dp|ps),...']
       ncu_source_regions.py SRC.csv --loops 'for (int i_dm' [--kernel-files rys_contract_jk.cu,unrolled_rys_jk.cu]
                             [--labels 'A;B;...'] [--loop-helpers rys_contract_k.cuh:190-235]"""
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


# per-instruction columns kept besides the stall reasons: global sectors requested, and the ideal (coalesced) count
EXTRA = ('L2 Theoretical Sectors Global', 'L2 Theoretical Sectors Global Ideal', 'Thread Instructions Executed')


def read_launches(path):
    '''The launches of the source page, in order: [(kernel, {address: [(file, line), ...]}, {address: (opcode,
    warp instructions, stall samples)})]. A new launch starts where the kernel name changes (the profiled launches
    alternate kernels, or are unrolled kernels of their own names). Each SASS instruction is listed under every
    source line of its inlining stack. The fourth item of an instruction holds its per-reason stall samples
    (stall_*) and the EXTRA columns.'''
    launches = []
    cur_file = cur_func = cur_line = None
    cols = None
    files = set()   # the source files listed so far for the current launch: one coming again starts the next
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == 'File Path':
                cur_file = row[1]
                continue
            if row[0] == 'Function Name':
                func = row[1].split('(')[0]
                if func != cur_func or cur_file in files:
                    cur_func = func
                    files = set()
                    launches.append((func, defaultdict(list), {}))
                files.add(cur_file)
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
            if not sass or row[2] == '...':   # an elided run of instructions, no metrics
                continue
            op = sass[1] if sass[0].startswith('@') else sass[0]
            _, lines, data = launches[-1]
            lines[row[2]].append((cur_file, cur_line))
            extra = {c: num(row[n]) for c, n in cols.items() if c in EXTRA or
                     (c.startswith('stall_') and not c.endswith('(Not Issued)'))}
            data[row[2]] = (op, num(row[cols['Instructions Executed']]),
                            num(row[cols['Warp Stall Sampling (All Samples)']]), extra)
    return launches


def parse(path):
    '''The launches of the second-derivative kernels: [(kernel, {region: [instructions, FP64, stall samples]},
    iterations)]. Each SASS instruction is counted once, under its line in the kernel's own file (the outermost
    frame) when it has one.'''
    launches = read_launches(path)
    out = []
    for func, lines, data in launches:
        reg = defaultdict(lambda: [0., 0., 0.])
        iters = 0.
        for addr, where in lines.items():
            own = [w for w in where if os.path.basename(w[0]) == 'rys_contract_jk_ip2.cu']
            file, line = (own or where)[0]
            r = region(func, file, line)
            op, n, st = data[addr][:3]
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


FP64 = ('DFMA', 'DADD', 'DMUL', 'DSETP', 'DMNMX')


def opclass(op):
    base = op.split('.')[0]
    if base == 'RED':
        return 'atomic adds (RED)'
    if base in ('ATOM', 'ATOMG'):
        return 'atomics (ATOM)'
    if base in ('LDG', 'LD'):
        return 'global loads'
    if base in ('LDS', 'STS', 'LDSM'):
        return 'shared memory'
    if base in FP64:
        return 'FP64'
    return 'other'


def loop_lines(path, marker):
    '''The line numbers (1-based) of path inside the loops whose header contains marker, by brace matching'''
    text = open(path).read().split('\n')
    inside = set()
    for n, line in enumerate(text):
        if marker not in line:
            continue
        depth, started = 0, False
        for m in range(n, len(text)):
            depth += text[m].count('{') - text[m].count('}')
            started = started or '{' in text[m]
            inside.add(m + 1)
            if started and depth <= 0:
                break
    return inside


def helper_lines(spec):
    '''{file name: line numbers} from 'file:first-last,...': the bodies of helpers that only the loops call'''
    out = defaultdict(set)
    for item in filter(None, spec.split(',')):
        name, rng = item.split(':')
        a, b = map(int, rng.split('-'))
        out[name].update(range(a, b + 1))
    return out


def loop_report(path, marker, kernel_files, labels=None, helpers=None):
    '''Per launch: the instructions and stall samples inside the loops marked by marker (in kernel_files, the
    outermost frame of each instruction) and outside them, by instruction class. An instruction listed only under
    the lines of helpers (helper_lines) counts as inside the loops; one listed only under other files' lines (CUDA
    headers) takes the region of the instruction before it.'''
    helpers = helpers or {}
    cache = {}
    for n, (func, lines, data) in enumerate(read_launches(path)):
        tab = defaultdict(lambda: [0., 0.])
        why = defaultdict(lambda: defaultdict(float))
        sectors = defaultdict(float)
        where_is = {}   # address: True (in the loops), False, or None (listed only under headers)
        for addr, where in lines.items():
            own = [w for w in where if os.path.basename(w[0]) in kernel_files]
            if own:
                file, line = own[0]
                if file not in cache:
                    cache[file] = loop_lines(file, marker)
                where_is[addr] = line in cache[file]
            elif any(ln in helpers.get(os.path.basename(f), ()) for f, ln in where):
                where_is[addr] = True
            else:
                where_is[addr] = None
        # an instruction listed only under header lines (e.g. atomicAdd's) takes the region of the nearest
        # preceding instruction (by address) that has one
        last = False
        for addr in sorted(where_is, key=lambda a: int(a, 16)):
            if where_is[addr] is None:
                where_is[addr] = last
            last = where_is[addr]
        for addr in lines:
            r = 'density loop' if where_is[addr] else 'integrals, rest'
            op, k, st, extra = data[addr]
            tab[r, opclass(op)][0] += k
            tab[r, opclass(op)][1] += st
            for c, v in extra.items():
                why[r][c] += v
                if c in EXTRA:
                    sectors[r, opclass(op), c] += v
            sectors[r, opclass(op), 'warp'] += k
        ti = sum(v[0] for v in tab.values())
        ts = sum(v[1] for v in tab.values())
        lab = f'{labels[n]}: ' if labels else ''
        print(f'{lab}{func}: {ti / 1e9:.3f} G warp instructions, {ts:.0f} stall samples')
        for r in ('density loop', 'integrals, rest'):
            ri = sum(v[0] for (rr, c), v in tab.items() if rr == r)
            rs = sum(v[1] for (rr, c), v in tab.items() if rr == r)
            print(f'  {r:16s} {100 * ri / max(ti, 1):5.1f}% of instructions, {100 * rs / max(ts, 1):5.1f}% of stalls: ' +
                  ', '.join(f'{c} {100 * v[0] / max(ti, 1):.1f}%/{100 * v[1] / max(ts, 1):.1f}%'
                            for (rr, c), v in sorted(tab.items(), key=lambda kv: -kv[1][1]) if rr == r))
            reasons = sorted(((c, v) for c, v in why[r].items() if c.startswith('stall_')), key=lambda cv: -cv[1])
            print(f'  {"":16s} stalls by reason: ' + ', '.join(f'{c[6:]} {100 * v / max(ts, 1):.1f}%'
                                                             for c, v in reasons[:5] if v))
            for c in ('global loads', 'atomics (ATOM)', 'atomic adds (RED)'):
                got, ideal, threads = (sectors[r, c, x] for x in EXTRA)
                if got:
                    print(f'  {"":16s} {c}: {threads / 1e6:.1f} M thread operations '
                          f'({threads / max(sectors[r, c, "warp"], 1):.1f} per warp instruction), {got / 1e6:.1f} M L2 '
                          f'sectors, {got / max(ideal, 1):.2f}x the coalesced count')


# (first line, last line, region) of the generic MD J kernel md_j_1dm_kernel in md_contract_j.cu; iter_Rt_n (lines
# 39-69) is inlined into the R tensor region
J_GENERIC = [(39, 69, 'R tensor'), (254, 322, 'R tensor'), (324, 356, 'contraction'),
             (358, 398, 'reductions, atomics')]
J_ORDER = ['setup, screening', 'Boys function', 'R tensor', 'contraction', 'R tensor, contraction',
           'reductions, atomics']


def unrolled_j_regions(path):
    '''{kernel: [(first line, last line, region)]} of the unrolled J kernels in path: after the boys_fn call up to the
    bra-side reduction ('double *vj_cache = Rp_cache;') the R tensor and the contraction, from there to the next
    kernel the reductions and the atomics'''
    text = open(path).read().split('\n')
    starts = [(n + 1, t.split('void ')[1].split('(')[0]) for n, t in enumerate(text) if t.startswith('void md_j_')]
    out = {}
    for k, (a, name) in enumerate(starts):
        b = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(text)
        boys = next(n + 1 for n in range(a, b) if 'boys_fn(' in text[n])
        red = next(n + 1 for n in range(boys, b) if 'double *vj_cache = Rp_cache;' in text[n])
        out[name] = [(boys + 1, red - 2, 'R tensor, contraction'), (red - 1, b, 'reductions, atomics')]
    return out


def j_report(path, src_dir, labels=None):
    '''Per launch of the J engine: warp instructions, FP64 instructions (D* opcodes), stall samples and active lanes
    per region. An instruction is in the Boys function if any frame of its inlining stack is in gamma_inc.cu or
    boys.cu; otherwise its line in the kernel's own file (md_contract_j.cu, unrolled_md_j*.cu) decides.'''
    unrolled = {}
    for f in ('unrolled_md_j.cu', 'unrolled_md_j_4dm.cu'):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            unrolled.update(unrolled_j_regions(p))
    for n, (func, lines, data) in enumerate(read_launches(path)):
        tab = defaultdict(lambda: [0., 0., 0., 0.])   # warp inst, FP64, stall samples, thread inst
        ops = defaultdict(lambda: defaultdict(float))
        for addr, where in lines.items():
            names = [os.path.basename(f) for f, _ in where]
            if 'gamma_inc.cu' in names or 'boys.cu' in names:
                r = 'Boys function'
            else:
                r = 'setup, screening'
                spans = J_GENERIC if func.startswith('md_j_1dm') or func.startswith('md_j_4dm') else \
                    unrolled.get(func, [])
                own = [ln for f, ln in where if os.path.basename(f) in
                       ('md_contract_j.cu', 'unrolled_md_j.cu', 'unrolled_md_j_4dm.cu')]
                for ln in own:
                    for a, b, rr in spans:
                        if a <= ln <= b:
                            r = rr
                    if r != 'setup, screening':
                        break
            op, k, st, extra = data[addr]
            t = tab[r]
            t[0] += k
            t[2] += st
            t[3] += extra.get('Thread Instructions Executed', 0.)
            if op.split('.')[0] in FP64:
                t[1] += k
            ops[r][op.split('.')[0]] += k
        ti, tf, ts, tt = (sum(v[c] for v in tab.values()) for c in range(4))
        lab = f'{labels[n]}: ' if labels else ''
        print(f'{lab}{func}: {ti / 1e9:.3f} G warp instructions, {tf / 1e9:.3f} G FP64 ({100 * tf / max(ti, 1):.0f}%), '
              f'{ts:.0f} stall samples, {tt / max(32 * ti, 1):.2f} of the lanes active')
        print(f'  {"region":22s} {"inst":>6s} {"FP64":>6s} {"stall":>6s} {"lanes":>6s}  top opcodes')
        for r in J_ORDER:
            if r in tab:
                i, f, st, th = tab[r]
                top = sorted(ops[r].items(), key=lambda kv: -kv[1])[:6]
                print(f'  {r:22s} {100 * i / max(ti, 1):5.1f}% {100 * f / max(tf, 1):5.1f}% {100 * st / max(ts, 1):5.1f}% '
                      f'{th / max(32 * i, 1):6.2f}  ' + ' '.join(f'{o} {100 * v / max(i, 1):.0f}%' for o, v in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--j', default=None, metavar='SRC_DIR',
                    help='J-engine kernels (ncu_j_class.py): regions by the gvhf-md sources in SRC_DIR')
    ap.add_argument('--order', default=None, help='launch labels (ncu_parts.py --labels) for the class names; the '
                    'profiled launches are the first of each class and kernel among them')
    ap.add_argument('--classes', default=None, help='the classes profiled (ncu_ejk2_class.py --classes)')
    ap.add_argument('--loops', default=None, help='split by the loops whose header contains this text '
                    '(e.g. "for (int i_dm"), in --kernel-files')
    ap.add_argument('--kernel-files', default='rys_contract_jk.cu,unrolled_rys_jk.cu')
    ap.add_argument('--labels', default=None, help='with --loops: one label per launch, semicolon-separated')
    ap.add_argument('--loop-helpers', default='rys_contract_k.cuh:190-235',
                    help='with --loops: file:first-last,... of helpers called only inside the loops (default: '
                    'load_dm and dot_dm of the generic J/K kernels)')
    args = ap.parse_args()
    if args.j:
        j_report(args.csv, args.j, args.labels.split(';') if args.labels else None)
        return
    if args.loops:
        loop_report(args.csv, args.loops, set(args.kernel_files.split(',')),
                    args.labels.split(';') if args.labels else None, helper_lines(args.loop_helpers))
        return
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
