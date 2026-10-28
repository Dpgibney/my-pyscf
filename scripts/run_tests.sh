#!/usr/bin/env bash
# Usage: run_tests.sh <pyscf-tree> [test dirs relative to the tree's pyscf/ package...]
#   e.g. run_tests.sh pyscf-base            (default: dft scf df grad hessian tdscf)
#        run_tests.sh pyscf dft tdscf
# Mirrors upstream .github/workflows/run_tests.sh (xdist, one OpenMP thread per worker, pytest.ini's
# "not _high_cost and not _skip" selection) but runs from scratch/ so chkfiles and temporary files stay
# out of the clone. xcfun is not built, so its tests are skipped.
set -uo pipefail
. "$(dirname "$0")/env.sh"
tree=$1; shift
dirs=("$@")
(( ${#dirs[@]} )) || dirs=(dft scf df grad hessian tdscf)
paths=()
for d in "${dirs[@]}"; do paths+=("$WS/$tree/pyscf/$d"); done

work="$WS/scratch/tests-$tree"
rm -rf "$work"
mkdir -p "$work/tmp"
printf 'scf_hf_SCF_mute_chkfile = True\nTMPDIR = "%s"\n' "$work/tmp" > "$work/pyscf_conf.py"
cd "$work"
ulimit -s 20000
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_THREAD_TIMEOUT=4 OMP_WAIT_POLICY=passive \
    PYTHONPATH="$WS/$tree" PYSCF_CONFIG_FILE="$work/pyscf_conf.py" PYTHONNOUSERSITE=1 \
    "$WS/.venv/bin/python" -m pytest -c "$WS/$tree/pytest.ini" --rootdir "$WS/$tree" \
    -n "${PYTEST_JOBS:-10}" --dist loadfile --timeout=900 -q -p no:cacheprovider -rfE \
    --ignore="$WS/$tree/pyscf/dft/test/test_xcfun.py" "${paths[@]}"
