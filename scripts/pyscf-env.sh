# Source this to run Python with the optimized PySCF build from this workspace:
#     source /home/danny/Chemistry/my-pyscf/scripts/pyscf-env.sh
#     python my_script.py
# It activates the workspace venv (numpy/scipy/h5py pinned; the anaconda base env is untouched,
# add packages with `uv pip install --python "$VIRTUAL_ENV/bin/python" <pkg>`), puts the perf
# PySCF tree on PYTHONPATH, and raises PYSCF_MAX_MEMORY so two-electron integrals of systems up to
# ~370 basis functions stay in memory (exact; ~1.7-2x faster SCF than the 4000 MB default for
# caffeine/def2-SVP). Don't put this tree on anaconda's PYTHONPATH instead: PySCF's namespace
# plugin lookup would mix in modules from the pyscf 2.12.1 installed there.
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ws/.venv/bin/activate"
export PYTHONPATH="$_ws/pyscf${PYTHONPATH:+:$PYTHONPATH}"
export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-20000}"
unset _ws
