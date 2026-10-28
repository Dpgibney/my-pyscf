"""Time benchmark cases for one or more configurations and save a results JSON.

Usage: run_bench.py [--suite quick|full|micro] [--cases a,b] [--reps N] [--threads N] [--tag NAME] CONFIG...
  CONFIG = LABEL=TREE[:VAR=VAL,...] with TREE a PySCF tree in the workspace or 'wheel', e.g.
  run_bench.py --tag it0 A=wheel B=pyscf-base:OPENBLAS_CORETYPE=Barcelona

Each measurement is a fresh worker process (warm-up excluded from timing). Configurations are
interleaved within each repetition, in alternating order (ABBA...), so thermal and frequency drift
hits all of them equally.
"""
import argparse
import json
import time

from common import RESULTS, Config, machine_state, repo_shas, run_worker
from cases import CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--suite', default='quick')
    ap.add_argument('--cases', default=None, help='comma-separated case names (overrides --suite)')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--tag', default='bench')
    args = ap.parse_args()

    configs = [Config.parse(s) for s in args.configs]
    names = args.cases.split(',') if args.cases else [n for n, (_, s) in CASES.items() if args.suite in s]
    machine = machine_state()
    if not machine['benchmark_ready']:
        print('WARNING: machine is not benchmark-ready (see scripts/check_power.sh); '
              'timings are recorded but flagged.')
    out = {'meta': {'tag': args.tag, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'reps': args.reps,
                    'threads': args.threads, 'cases': names, 'machine_start': machine,
                    'shas': repo_shas(),
                    'configs': {c.label: {'tree': c.tree, 'env': c.env, 'variant': c.variant()}
                                for c in configs}},
           'runs': []}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f'{time.strftime("%Y%m%d-%H%M%S")}_{args.tag}.json'
    for name in names:
        for rep in range(args.reps):
            # ABBA: alternate the order so no config always runs right after another
            # (with boost clocks, a run that follows another sees a hotter, slower chip)
            for cfg in (configs if rep % 2 == 0 else configs[::-1]):
                rec = run_worker(name, cfg, args.threads)
                rec.update(config=cfg.label, rep=rep)
                out['runs'].append(rec)
                cats = sorted(rec['categories'].items(), key=lambda kv: -kv[1])[:3]
                print(f'{name:28s} {cfg.label:>6s} rep{rep} {rec["wall"]:8.2f}s  core={rec["blas_core"]}  '
                      + '  '.join(f'{k}={v:.2f}' for k, v in cats), flush=True)
                out['meta']['machine_end'] = machine_state()
                path.write_text(json.dumps(out, indent=1))
    print(f'saved {path}')


if __name__ == '__main__':
    main()
