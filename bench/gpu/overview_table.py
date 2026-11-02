"""The overview table from overview.py's JSON lines: for each workload and task, unmodified GPU4PySCF (label base)
against the current tree (label current), the speedup, the largest difference of the results and the SCF cycles, as
Markdown. A workload split over several runs (e.g. scf,grad,hess and scf,tddft) is merged, the first run's SCF kept.
Rows with only one of the two trees are left out and listed at the end.
Usage: overview_table.py overview.jsonl"""
import json
import sys

TASKS = [('scf', 'SCF'), ('grad', 'gradient'), ('hess', 'Hessian'), ('tddft', 'TDDFT, 5 states')]
UNITS = {'scf': 'Eh', 'grad': 'Eh/bohr', 'hess': 'Eh/bohr²', 'tddft': 'Eh'}
DIFF_KEY = {'scf': 'e', 'grad': 'grad', 'hess': 'hess', 'tddft': 'tddft'}

runs = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    key = (r['mol'], r['basis'], r['xc'], r.get('df', False))
    prev = runs.setdefault(key, {}).get(r['label'])
    if prev is None:
        runs[key][r['label']] = r
    else:  # another run of the same workload: add its tasks, keep the first run's SCF
        for field in ('times', 'sm_mhz', 'max_diff', 'max_ref'):
            for k, v in r.get(field, {}).items():
                prev.setdefault(field, {}).setdefault(k, v)

print('| workload | task | v1.9.0 | current | speedup | largest difference |')
print('|---|---|---|---|---|---|')
missing = []
for key, pair in runs.items():
    mol, basis, xc, df = key
    if 'base' not in pair or 'current' not in pair:
        missing.append(key)
        continue
    b, c = pair['base'], pair['current']
    name = f"{mol} {basis} {xc.upper().replace('_', '-')}{' (DF)' if df else ''}, {b['nao']} AOs"
    first = True
    for task, label in TASKS:
        if task not in b['times'] or task not in c['times']:
            continue
        tb, tc = b['times'][task], c['times'][task]
        if task == 'scf':
            cyc = (f"{b['cycles']} cycles" if b['cycles'] == c['cycles']
                   else f"{b['cycles']} / {c['cycles']} cycles")
            label = f'SCF ({cyc})'
        k = DIFF_KEY[task]
        d = c.get('max_diff', {}).get(k)
        m = c.get('max_ref', {}).get(k)
        diff = f'{d:.1e} {UNITS[task]}' + (f' (values up to {m:.2g})' if task in ('grad', 'hess') else '')
        print(f"| {name if first else ''} | {label} | {tb:.1f} s | {tc:.1f} s | {tb / tc:.2f}× | {diff} |")
        first = False
for key in missing:
    print(f'(only one tree ran: {key})')
