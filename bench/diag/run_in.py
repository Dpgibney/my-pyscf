"""Run a python script in a bench configuration's pinned environment (cwd = workspace scratch/).
Usage: run_in.py CONFIG [THREADS] script.py [args...]
  e.g. bench/diag/run_in.py G=pyscf bench/diag/switch_sweep.py caffeine B3LYP
"""
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import SCRATCH, Config, worker_env  # noqa: E402

cfg = Config.parse(sys.argv[1])
args = sys.argv[2:]
threads = int(args.pop(0)) if args[0].isdigit() else 10
args[0] = os.path.abspath(args[0])   # the script runs from scratch/
sys.exit(subprocess.run([cfg.python()] + args, env=worker_env(cfg, threads), cwd=SCRATCH).returncode)
