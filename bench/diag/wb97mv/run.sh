#!/bin/bash
# run.sh [THREADS] [MAXMEM_MB] script.py args...   -- workspace PySCF (perf tree), workspace venv
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
S=${SCRATCH:-/tmp}
T=${1}; M=${2}; shift 2
ulimit -v ${VMEM_KB:-18000000}   # hard cap on the address space: a runaway allocation fails instead of swapping the machine
exec env -u LD_LIBRARY_PATH OMP_NUM_THREADS=$T PYSCF_MAX_MEMORY=$M OPENBLAS_THREAD_TIMEOUT=20 \
  PYTHONPATH=$WS/pyscf:$WS/bench PYSCF_CONFIG_FILE=$WS/bench/pyscf_conf.py PYTHONNOUSERSITE=1 \
  $WS/.venv/bin/python -W ignore -u "$@"
