# Source this to run Python with the optimized PySCF build from this workspace (scripts/cpu_setup.sh builds it):
#     source <workspace>/scripts/pyscf-env.sh
#     python my_script.py
# It activates the workspace venv (numpy/scipy/h5py pinned; add packages with `python -m pip install <pkg>`, or
# `uv pip install --python "$VIRTUAL_ENV/bin/python" <pkg>` in a venv made by uv, which has no pip: a bare `pip`
# there would be another Python's) and puts the perf PySCF tree on PYTHONPATH. The defaults below are worked out
# each time this is sourced, from the cores and memory this process may use (scripts/resources.py), so in a Slurm
# job they follow that job's request (-c, --mem). Export your own values first to override them.
#   OMP_NUM_THREADS         one thread per physical core of the CPUs this process may run on (a Slurm job's
#                           allocation, taskset), at most $SLURM_CPUS_PER_TASK. With two threads per core (all
#                           20 hardware threads of the laptop's 10 cores) every workload measured was 20-33% slower.
#   PYSCF_MAX_MEMORY        2/3 of the memory this process may use (the RAM, a cgroup limit such as a Slurm job's,
#                           or Slurm's --mem/--mem-per-cpu), at most 20000 MB outside Slurm jobs. At 20000 the
#                           two-electron integrals of systems up to ~370 basis functions stay in memory (exact;
#                           1.7-2x faster SCF than the 4000 MB default for caffeine/def2-SVP). Larger systems run
#                           direct SCF, which keeps the integrals it can in up to 3/4 of the memory left
#                           (semi-direct, exact; mf.semi_direct = False switches it off). DFT also keeps the
#                           radial parts of the AO values on the grids when they fit in 1/4 of the memory left
#                           (0.3-1.4 GB for 25-60 atoms; exact). Each job may thus use most of this budget: lower
#                           it when running several jobs at once outside a scheduler.
#   OPENBLAS_THREAD_TIMEOUT=20  numpy's and scipy's own OpenBLAS thread pools stop spinning
#                           2^20 cycles (~0.5 ms) after a call instead of 2^28 (~134 ms). The
#                           spinning threads otherwise share the cores with PySCF's OpenMP
#                           threads: SCFs 1.2-1.9x faster, identical results.
# In a Slurm job, run python directly in the batch script, or with `srun -c $SLURM_CPUS_PER_TASK`: since Slurm
# 22.05 srun does not take -c from sbatch, and its step would get one CPU.
# Don't put this tree on the PYTHONPATH of another Python that has PySCF installed (e.g. anaconda's) instead:
# PySCF's namespace plugin lookup would mix in modules from that installation.
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv/bin/activate"
export PYTHONPATH="$_ws/pyscf${PYTHONPATH:+:$PYTHONPATH}"
read -r _cores _mem < <(python -I "$_ws/scripts/resources.py")
_mem=$(( ${_mem:-6000} * 2 / 3 ))
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$_mem" -gt 20000 ]; then
    _mem=20000
fi
export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-$_mem}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${_cores:-$(nproc)}}"
export OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-20}"
unset _ws _cores _mem
