'''Side by side, the phase tables of hess_mem.py runs in one log (sections opened by "== LABEL ..." lines, as
scratch/hess-mem-ab/run.sh writes them): calls, time and peak RSS of each phase per run.

Usage: hess_mem_compare.py RUN.log'''
import re
import sys

runs = {}
order = []
label = None
in_table = False
for line in open(sys.argv[1]):
    m = re.match(r'== (\S+)', line)
    if m:
        label = m.group(1)
        runs[label] = {'phases': {}, 'total': None}
        in_table = False
        continue
    if label is None:
        continue
    m = re.match(r'Hessian done, (\d+) s in all; .*peak RSS ([\d.]+) GB \(ru_maxrss\)', line)
    if m:
        runs[label]['total'] = (float(m.group(1)), float(m.group(2)))
    if line.startswith('phase '):
        in_table = True
        continue
    if in_table:
        m = re.match(r'(.+?)\s+(\d+)\s+([\d.]+)s\s+([\d.]+) GB', line)
        if not m:
            in_table = False
            continue
        name = m.group(1).strip()
        runs[label]['phases'][name] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
        if name not in order:
            order.append(name)

labels = list(runs)
print(f'{"phase":36s}' + ''.join(f'{lab + " time":>13s}{lab + " peak":>12s}' for lab in labels))
for name in order:
    row = f'{name:36s}'
    for lab in labels:
        p = runs[lab]['phases'].get(name)
        row += f'{p[1]:12.1f}s{p[2]:9.2f} GB' if p else f'{"":13s}{"":12s}'
    print(row)
row = f'{"whole run (ru_maxrss)":36s}'
for lab in labels:
    t = runs[lab]['total']
    row += f'{t[0]:12.0f}s{t[1]:9.2f} GB' if t else f'{"":13s}{"":12s}'
print(row)
