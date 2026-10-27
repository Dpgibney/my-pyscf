"""Summarize a results JSON as markdown: per case, median wall time per config, speedup vs a base
config, the largest differences in computed values, and the cost breakdown by category.

Usage: compare.py RESULTS.json [--base LABEL]
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

SKIP_KEYS = ('cycles', 'converged')


def max_delta(vals, ref):
    '''Largest absolute difference over the numeric values both dicts share.'''
    worst = 0.
    for k, v in vals.items():
        if k in SKIP_KEYS or k not in ref:
            continue
        a, b = np.asarray(v, dtype=float), np.asarray(ref[k], dtype=float)
        if a.shape == b.shape:
            worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--base', default=None)
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text())
    meta = data['meta']
    labels = list(meta['configs'])
    base = args.base or labels[0]
    runs = {}
    for r in data['runs']:
        runs.setdefault((r['case'], r['config']), []).append(r)

    print(f'## {meta["tag"]} ({meta["time"]}), {meta["threads"]} threads, {meta["reps"]} reps, '
          f'benchmark_ready={meta["machine_start"]["benchmark_ready"]}')
    for label, c in meta['configs'].items():
        print(f'- **{label}**: tree={c["tree"]} env={c["env"] or "-"} [{c["variant"]}]')
    print(f'- shas: {meta["shas"]}')

    for case in meta['cases']:
        if (case, base) not in runs:
            continue
        print(f'\n### {case}')
        if case.startswith('micro_'):
            keys = list(runs[(case, base)][0]['values'])
            print('| metric | ' + ' | '.join(labels) + ' |\n|---|' + '---|' * len(labels))
            for k in keys:
                cells = [f'{statistics.median(r["values"][k] for r in runs[(case, l)]):.3f}'
                         if (case, l) in runs else '-' for l in labels]
                print(f'| {k} | ' + ' | '.join(cells) + ' |')
            continue
        t_base = statistics.median(r['wall'] for r in runs[(case, base)])
        ref_vals = runs[(case, base)][0]['values']
        cats = {}
        for label in labels:
            for cat in runs.get((case, label), [{}])[0].get('categories', {}):
                cats[cat] = 1
        base_cats = {cat: statistics.median(r['categories'].get(cat, 0.) for r in runs[(case, base)])
                     for cat in cats}
        shown = sorted(base_cats, key=lambda c: -base_cats[c])[:6]
        print('| config | wall (s) | min–max | speedup | cycles | max Δvalue | '
              + ' | '.join(shown) + ' |')
        print('|---' * (6 + len(shown)) + '|')
        for label in labels:
            rs = runs.get((case, label))
            if not rs:
                continue
            walls = [r['wall'] for r in rs]
            t = statistics.median(walls)
            cyc = rs[0]['values'].get('cycles', '-')
            delta = max(max_delta(r['values'], ref_vals) for r in rs)
            catcells = [f'{statistics.median(r["categories"].get(c, 0.) for r in rs):.2f}' for c in shown]
            print(f'| {label} | {t:.2f} | {min(walls):.2f}–{max(walls):.2f} | {t_base / t:.2f}x | {cyc} | '
                  f'{delta:.1e} | ' + ' | '.join(catcells) + ' |')


if __name__ == '__main__':
    main()
