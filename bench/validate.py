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
    '''The cached reference record, and for post-SCF cases the reference's SCF solution (see
    cases.converged_ks), which the candidate starts from.'''
    sha = repo_shas().get(cfg.tree, cfg.tree)
    env = '-'.join(f'{k}={v}' for k, v in sorted(cfg.env.items())) or 'default'
    variant = cfg.variant().replace(' ', '_').replace('=', '-')
    path = RESULTS / 'ref' / f'{name}__{cfg.tree}@{sha}__{variant}__{env}__t{threads}.json'
    scf_path = path.with_suffix('.scf.npz') if getattr(CASES[name][0], 'post_scf', False) else None
    if path.exists() and (scf_path is None or scf_path.exists()):
        return json.loads(path.read_text()), scf_path
    run_cfg = cfg if scf_path is None else Config(cfg.label, cfg.tree, {**cfg.env, 'BENCH_SCF_OUT': str(scf_path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = run_worker(name, run_cfg, threads, tight=True)
    path.write_text(json.dumps(rec))
    return rec, scf_path


def compare(cand, ref):
    rows, ok = [], True
    for key, rv in ref.items():
        cv = cand.get(key)
        if key == 'cycles':
            rows.append((key, f'{cv} vs {rv}', '', 'warn' if cv != rv else 'ok'))
            continue
        if key == 'converged':
            # the candidate must converge wherever the reference did
            good = bool(np.all(np.asarray(cv) | ~np.asarray(rv, dtype=bool)))
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
        r, scf_path = reference(name, ref, args.threads)
        cand_cfg = cand if scf_path is None else \
            Config(cand.label, cand.tree, {**cand.env, 'BENCH_SCF_IN': str(scf_path)})
        c = run_worker(name, cand_cfg, args.threads, tight=True)
        ok, rows = compare(c['values'], r['values'])
        # an unconverged reference checks nothing: fix the case's settings and delete the cached file
        ref_conv = bool(np.all(r['values'].get('converged', True)))
        ok &= ref_conv
        all_ok &= ok
        print(f'{"PASS" if ok else "FAIL"}  {name}' + ('' if ref_conv else '  (reference did not converge)'))
        for key, diff, tol, status in rows:
            if status != 'ok' or key in ABS_TOL:
                print(f'      {key:22s} max|d|={diff:>10s} tol={tol:>6s} {status}')
    print('ALL PASS' if all_ok else 'SOME FAILED')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
