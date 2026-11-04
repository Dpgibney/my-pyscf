# Source this to run GPU4PySCF from the venv and the trees of scripts/gpu_setup.sh:
#     source scripts/gpu-env.sh perf      the modified tree (gpu4pyscf/)
#     source scripts/gpu-env.sh base      the unmodified v1.9.0 tree (gpu4pyscf-base/)
#     source scripts/gpu-env.sh           the v1.9.0 wheel installed in the venv
# It activates .venv-gpu and puts the CUDA 12 libraries of NVIDIA's pip packages, and cuTENSOR's, on
# LD_LIBRARY_PATH (CuPy 13.4.1 does not look inside those packages itself).
# Defaults (export them first to override):
#   OMP_NUM_THREADS          the number of physical cores, for the CPU parts
#   OPENBLAS_THREAD_TIMEOUT  20: numpy's and scipy's OpenBLAS threads stop spinning soon after a call instead of
#                            competing with PySCF's OpenMP threads
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv-gpu/bin/activate"
_sp="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
_libs="$(printf '%s:' "$_sp"/nvidia/*/lib "$_sp"/cutensor/lib)"
export LD_LIBRARY_PATH="${_libs%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)}"
export OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-20}"
export PYSCF_CONFIG_FILE="$_ws/bench/pyscf_conf.py"
case "${1:-}" in
    perf) export PYTHONPATH="$_ws/gpu4pyscf${PYTHONPATH:+:$PYTHONPATH}" ;;
    base) export PYTHONPATH="$_ws/gpu4pyscf-base${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
unset _ws _sp _libs
