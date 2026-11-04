#!/bin/bash
# The GPU comparison on this machine (after scripts/gpu_setup.sh): the modified tree's unit tests, then
# bench/gpu/overview.py for v1.9.0 (label base) and the modified tree (label current) on the workloads of the
# node1 overview (NOTES.md, "GPU4PySCF overview"), each run timed once after a warm-up, and the table of
# bench/gpu/overview_table.py. Everything goes to bench/results/<host>/: machine.txt (GPU, driver, packages, the
# trees' commits), tests.txt, overview.jsonl, overview_table.md, the saved results (*.npy) and log.txt.
#     bash scripts/gpu_bench.sh [--quick] [--no-tests]
#   --quick      caffeine def2-SVP B3LYP only
#   --no-tests   skip the unit tests
set -uo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUICK=0
TESTS=1
for a in "$@"; do
    case "$a" in
        --quick) QUICK=1 ;;
        --no-tests) TESTS=0 ;;
        *) echo "unknown argument $a" >&2; exit 1 ;;
    esac
done
for T in gpu4pyscf gpu4pyscf-base; do
    [ -d "$WS/$T/gpu4pyscf" ] || { echo "no $WS/$T: run scripts/gpu_setup.sh first" >&2; exit 1; }
done
source "$WS/scripts/gpu-env.sh" ""   # the venv only; each run below sets PYTHONPATH to its tree
R="$WS/bench/results/$(hostname -s)"
mkdir -p "$R"
OUT="$R/overview.jsonl"
{
    date
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv
    nvcc --version | tail -n 2
    lscpu | grep -E "Model name|^CPU\(s\)|Socket"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    for T in gpu4pyscf gpu4pyscf-base; do
        echo "$T: $(git -C "$WS/$T" describe --tags --always) $(git -C "$WS/$T" rev-parse HEAD)"
    done
    python -m pip freeze
} > "$R/machine.txt" 2>&1
echo "results in $R"

if [ $TESTS = 1 ]; then
    echo "== unit tests of the modified tree"
    (
        cd "$R"
        for f in scf/tests/test_scf_jk.py scf/tests/test_jk_one_pass.py dft/tests/test_rks.py dft/tests/test_uks.py \
                 grad/tests/test_rks_grad.py grad/tests/test_uks_grad.py grad/tests/test_tdrks_grad.py \
                 hessian/tests/test_rks_hessian.py hessian/tests/test_uks_hessian.py tdscf/tests/test_tdrks.py; do
            PYTHONPATH="$WS/gpu4pyscf" timeout 3600 python -m pytest -q -p no:cacheprovider \
                "$WS/gpu4pyscf/gpu4pyscf/$f" 2>&1 | tail -n 1 | sed "s|^|$f: |"
        done
    ) | tee "$R/tests.txt"
fi

run() {  # mol basis xc tasks [--df]: v1.9.0, then the modified tree compared with it
    local tag=$1_$2_$3${5:+_df}_${4//,/-}
    echo "== $1 $2 $3 ${5:-} ($4)"
    PYTHONPATH="$WS/gpu4pyscf-base" python -W ignore "$WS/bench/gpu/overview.py" --label base --mol $1 --basis $2 \
        --xc $3 --tasks $4 ${5:-} --save "$R/${tag}_base" --out "$OUT" 2>&1 | tail -n 1 | cut -c1-300
    PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$WS/bench/gpu/overview.py" --label current --mol $1 --basis $2 \
        --xc $3 --tasks $4 ${5:-} --save "$R/${tag}_cur" --compare "$R/${tag}_base" --out "$OUT" 2>&1 \
        | tail -n 1 | cut -c1-300
}
{
    run caffeine def2-svp b3lyp scf,grad,hess
    run caffeine def2-svp b3lyp scf,tddft
    if [ $QUICK = 0 ]; then
        run caffeine def2-svp pbe scf,grad,hess
        run caffeine def2-tzvp pbe scf,grad
        run caffeine def2-svp r2scan scf,grad,hess
        run caffeine def2-tzvp b3lyp scf,grad
        run caffeine def2-svp b3lyp scf,grad,hess --df
        run caffeine def2-svp b97m_v scf,grad
        run caffeine def2-svp wb97m_v scf,grad
        run benzene def2-svp b97m_v scf,grad,hess
    fi
} 2>&1 | tee "$R/log.txt"
python "$WS/bench/gpu/overview_table.py" "$OUT" > "$R/overview_table.md"
cat "$R/overview_table.md"
echo "== done: $R (commit and push it, or copy it back)"
