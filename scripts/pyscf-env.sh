# Source this to run Python with the optimized PySCF build from this workspace (scripts/cpu_setup.sh builds it):
#     source <workspace>/scripts/pyscf-env.sh
#     python my_script.py
# It activates the workspace venv (numpy/scipy/h5py pinned; add packages with `python -m pip install <pkg>`, or
# `uv pip install --python "$VIRTUAL_ENV/bin/python" <pkg>` in a venv made by uv, which has no pip: a bare `pip`
# there would be another Python's) and puts the perf PySCF tree on PYTHONPATH. Defaults (override by exporting
# them first):
#   PYSCF_MAX_MEMORY=20000  (or 2/3 of the RAM if that is less) two-electron integrals of systems up to ~370 basis
#                           functions stay in memory (exact; 1.7-2x faster SCF than the 4000 MB default for
#                           caffeine/def2-SVP). Larger systems run direct SCF, which keeps the
#                           integrals it can in up to 3/4 of the memory left (semi-direct, exact;
#                           mf.semi_direct = False switches it off). DFT also keeps the radial
#                           parts of the AO values on the grids when they fit in 1/4 of the memory
#                           left (0.3-1.4 GB for 25-60 atoms; exact). Each job may thus use most of
#                           this budget: lower it when running several jobs at once.
#   OMP_NUM_THREADS=<physical cores>  one thread per core; with all hardware threads (the OpenMP default,
#                           20 on the laptop's 10 cores) every workload measured was 20-33% slower
#   OPENBLAS_THREAD_TIMEOUT=20  numpy's and scipy's own OpenBLAS thread pools stop spinning
#                           2^20 cycles (~0.5 ms) after a call instead of 2^28 (~134 ms). The
#                           spinning threads otherwise share the cores with PySCF's OpenMP
#                           threads: SCFs 1.2-1.9x faster, identical results.
# Don't put this tree on the PYTHONPATH of another Python that has PySCF installed (e.g. anaconda's) instead:
# PySCF's namespace plugin lookup would mix in modules from that installation.
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv/bin/activate"
export PYTHONPATH="$_ws/pyscf${PYTHONPATH:+:$PYTHONPATH}"
_cores=$(lscpu -p=Core,Socket 2> /dev/null | grep -v '^#' | sort -u | wc -l)
[ "$_cores" -gt 0 ] || _cores=$(nproc)
# MemTotal is in KiB, PYSCF_MAX_MEMORY in MB (10^6 bytes)
_mem=$(awk '/^MemTotal:/ {m = int($2 * 1.024e-3 / 1.5); print (m < 20000 ? m : 20000)}' /proc/meminfo)
export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-${_mem:-4000}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_cores}"
export OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-20}"
unset _ws _cores _mem
