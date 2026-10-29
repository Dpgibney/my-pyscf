# Source this to run Python with the optimized PySCF build from this workspace:
#     source /home/danny/Chemistry/my-pyscf/scripts/pyscf-env.sh
#     python my_script.py
# It activates the workspace venv (numpy/scipy/h5py pinned; the anaconda base env is untouched,
# add packages with `uv pip install --python "$VIRTUAL_ENV/bin/python" <pkg>`) and puts the perf
# PySCF tree on PYTHONPATH. Defaults (override by exporting them first):
#   PYSCF_MAX_MEMORY=20000  two-electron integrals of systems up to ~370 basis functions stay in
#                           memory (exact; 1.7-2x faster SCF than the 4000 MB default for
#                           caffeine/def2-SVP). Larger systems run direct SCF, which keeps the
#                           integrals it can in up to 3/4 of the memory left (semi-direct, exact;
#                           mf.semi_direct = False switches it off). DFT also keeps the radial
#                           parts of the AO values on the grids when they fit in 1/4 of the memory
#                           left (0.3-1.4 GB for 25-60 atoms; exact). Each job may thus use most of
#                           this budget: lower it when running several jobs at once.
#   OMP_NUM_THREADS=10      one thread per physical core; with all 20 hardware threads (the
#                           OpenMP default) every workload measured was 20-33% slower
# Don't put this tree on anaconda's PYTHONPATH instead: PySCF's namespace plugin lookup would mix
# in modules from the pyscf 2.12.1 installed there.
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv/bin/activate"
export PYTHONPATH="$_ws/pyscf${PYTHONPATH:+:$PYTHONPATH}"
export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-20000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
unset _ws
