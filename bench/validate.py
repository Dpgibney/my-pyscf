"""Correctness gate: run cases with tight convergence in a candidate config and compare against the
unmodified reference build (results cached per reference SHA/variant/env).

Usage: validate.py CAND [--ref SPEC] [--suite quick|full] [--cases a,b] [--threads N]
  e.g. validate.py new=pyscf
Exit status 1 if any value exceeds its tolerance.
"""
import argparse
import json
import sys

import numpy as np

from common import RESULTS, Config, repo_shas, run_worker
from cases import CASES

ABS_TOL = {'e_tot': 1e-9, 'grad': 1e-8, 'hessian': 1e-7, 'excitations': 1e-9}
REL_TOL = 1e-11   # every other numeric value (e.g. grid checksums)
REF_DEFAULT = 'ref=pyscf-base'


def reference(name, cfg, threads):
    sha = repo_shas().get(cfg.tree, cfg.tree)
    env = '-'.join(f'{k}={v}' for k, v in sorted(cfg.env.items())) or 'default'
    variant = cfg.variant().replace(' ', '_').replace('=', '-')
    path = RESULTS / 'ref' / f'{name}__{cfg.tree}@{sha}__{variant}__{env}__t{threads}.json'
    if path.exists():
        return json.loads(path.read_text())
    rec = run_worker(name, cfg, threads, tight=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def compare(cand, ref):
    rows, ok = [], True
    for key, rv in ref.items():
        cv = cand.get(key)
        if key == 'cycles':
            rows.append((key, f'{cv} vs {rv}', '', 'warn' if cv != rv else 'ok'))
            continue
        if key == 'converged':
            good = bool(np.all(cv))
            ok &= good
            rows.append((key, str(cv), '', 'ok' if good else 'FAIL'))
            continue
        a, b = np.asarray(cv, dtype=float), np.asarray(rv, dtype=float)
        if a.shape != b.shape:
            ok = False
            rows.append((key, f'shape {a.shape} vs {b.shape}', '', 'FAIL'))
            continue
        diff = float(np.max(np.abs(a - b))) if a.size else 0.
        tol = ABS_TOL.get(key, REL_TOL * max(1., float(np.max(np.abs(b))) if b.size else 1.))
        good = diff <= tol
        ok &= good
        rows.append((key, f'{diff:.2e}', f'{tol:.0e}', 'ok' if good else 'FAIL'))
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cand')
    ap.add_argument('--ref', default=REF_DEFAULT)
    ap.add_argument('--suite', default='quick')
    ap.add_argument('--cases', default=None)
    ap.add_argument('--threads', type=int, default=10)
    args = ap.parse_args()
    cand, ref = Config.parse(args.cand), Config.parse(args.ref)
    names = args.cases.split(',') if args.cases else \
        [n for n, (_, s) in CASES.items() if args.suite in s and not n.startswith('micro_')]
    all_ok = True
    print(f'candidate {cand.label}: {cand.tree} {cand.env} [{cand.variant()}]')
    print(f'reference {ref.label}: {ref.tree} {ref.env} [{ref.variant()}]')
    for name in names:
        r = reference(name, ref, args.threads)
        c = run_worker(name, cand, args.threads, tight=True)
        ok, rows = compare(c['values'], r['values'])
        all_ok &= ok
        print(f'{"PASS" if ok else "FAIL"}  {name}')
        for key, diff, tol, status in rows:
            if status != 'ok' or key in ABS_TOL:
                print(f'      {key:22s} max|d|={diff:>10s} tol={tol:>6s} {status}')
    print('ALL PASS' if all_ok else 'SOME FAILED')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
