# Source this on node1 to run GPU4PySCF from the venv that node1_gpu_venv.sh made:
#     source ~/Chemistry/my-pyscf/scripts/node1-gpu-env.sh
#     python my_script.py
# It activates .venv-gpu and puts the CUDA 12 libraries of NVIDIA's pip packages, and cuTENSOR's, on
# LD_LIBRARY_PATH: node1 has no CUDA toolkit, and CuPy 13.4.1 does not look inside those packages itself.
# With the argument `perf` (source ... node1-gpu-env.sh perf) it also puts the gpu4pyscf perf tree that
# node1_gpu4pyscf_tree.sh copies to $WS/gpu4pyscf first on PYTHONPATH; without it the unmodified wheel runs.
# Defaults (override by exporting them first):
#   OMP_NUM_THREADS=8       one thread per physical core of the i7-11700K, for the CPU parts
#   OPENBLAS_THREAD_TIMEOUT=20  as in pyscf-env.sh: numpy's and scipy's OpenBLAS threads stop spinning
#                           soon after a call instead of competing with PySCF's OpenMP threads
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv-gpu/bin/activate"
_sp="$_ws/.venv-gpu/lib/python3.12/site-packages"
_libs="$(printf '%s:' "$_sp"/nvidia/*/lib "$_sp"/cutensor/lib)"
export LD_LIBRARY_PATH="${_libs%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-20}"
if [ "${1:-}" = perf ]; then
    export PYTHONPATH="$_ws/gpu4pyscf${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _ws _sp _libs
