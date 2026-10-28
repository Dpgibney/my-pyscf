"""Run prune_check.py in the wheel and pyscf-base environments and compare the Hessians."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from common import SCRATCH, Config, worker_env

SCRIPT = Path(__file__).resolve().parent / 'prune_check.py'
RUNS = [('wheel default', 'A=wheel', []),
        ('master, cutoff 1e-7', 'B=pyscf-base', ['1e-7']),
        ('master default', 'B=pyscf-base', [])]

res = {}
for label, spec, args in RUNS:
    cfg = Config.parse(spec)
    p = subprocess.run([cfg.python(), str(SCRIPT)] + args, env=worker_env(cfg, 10), cwd=SCRATCH,
                       capture_output=True, text=True, check=True)
    rec = json.loads(next(l for l in p.stdout.splitlines() if l.startswith('@@'))[2:])
    res[label] = rec
    print(f"{label:22s} pyscf {rec['version']} cutoff={rec['cutoff']} ngrids={rec['ngrids']} "
          f"e_tot={rec['e_tot']:.12f}", flush=True)

ref = res['wheel default']
for label in ('master, cutoff 1e-7', 'master default'):
    r = res[label]
    dh = np.max(np.abs(np.asarray(r['hessian']) - np.asarray(ref['hessian'])))
    print(f"{label:22s} vs wheel: |dE|={abs(r['e_tot'] - ref['e_tot']):.1e}  max|dH|={dh:.1e}")
