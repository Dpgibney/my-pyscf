#!/bin/bash
# The GPU comparison on this machine (after scripts/gpu_setup.sh): the modified tree's unit tests, then
# bench/gpu/overview.py for v1.9.0 (label base) and the modified tree (label current) on the workloads of the
# node1 overview (NOTES.md, "GPU4PySCF overview"), each run timed once after a warm-up, and the table of
# bench/gpu/overview_table.py. Everything goes to bench/results/<host>/: machine.txt (GPU, driver, packages, the
# trees' commits), tests.txt, overview.jsonl, overview_table.md, the saved results (*.npy) and log.txt.
# After the overview comes the production-like job of bench/gpu/prod_parts.py (caffeine, wB97M-V, XC grid (99,590),
# VV10 grid (50,194), SMD water; every part of the SCF and of the gradient timed) for v1.9.0, the modified tree, and
# the modified tree's two-level SCF, on def2-SVP, def2-TZVP and def2-TZVPD: prod_parts.jsonl and prod_log.txt.
# From part 37 on the two-level SCF iterates with a seminumerical K (gpu4pyscf.sgx); it is run three ways: with the
# model's own K in its cheap model (part 36's), with the seminumerical K (the default), and with a fitted J as well.
#     bash scripts/gpu_bench.sh [--quick] [--no-tests] [--large]
#   --quick      caffeine def2-SVP B3LYP only, and the production-like job on def2-SVP only
#   --no-tests   skip the unit tests
#   --large      also NAD+ (70 atoms, 1916 functions on def2-TZVPD, about 15,000 surface points) with the modified
#                tree, two-level and plain: the production size, for a card with 24 GB or more
set -uo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUICK=0
TESTS=1
LARGE=0
for a in "$@"; do
    case "$a" in
        --quick) QUICK=1 ;;
        --no-tests) TESTS=0 ;;
        --large) LARGE=1 ;;
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
        for f in scf/tests/test_scf_jk.py scf/tests/test_jk_one_pass.py scf/tests/test_two_level.py \
                 sgx/tests/test_sgx.py \
                 dft/tests/test_rks.py dft/tests/test_uks.py \
                 grad/tests/test_rks_grad.py grad/tests/test_uks_grad.py grad/tests/test_scf_reuse.py \
                 grad/tests/test_tdrks_grad.py \
                 solvent/tests/test_pcm_grad.py solvent/tests/test_smd_grad.py solvent/tests/test_pcm_tdscf_grad.py \
                 hessian/tests/test_rks_hessian.py hessian/tests/test_uks_hessian.py \
                 solvent/tests/test_pcm_hessian.py solvent/tests/test_smd_hessian.py tdscf/tests/test_tdrks.py; do
            [ -f "$WS/gpu4pyscf/gpu4pyscf/$f" ] || { echo "$f: not in this tree (an older pin)"; continue; }
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

prod() {  # basis: v1.9.0, the modified tree compared with it, and the modified tree's two-level SCF
    local P="$WS/bench/gpu/prod_parts.py" J="$R/prod_parts.jsonl"
    echo "== production-like job, caffeine $1: v1.9.0"
    PYTHONPATH="$WS/gpu4pyscf-base" python -W ignore "$P" --label base --basis $1 --save "$R/prod_$1_base" --out "$J"
    echo "== production-like job, caffeine $1: the modified tree"
    PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$P" --label current --basis $1 --compare "$R/prod_$1_base" --out "$J"
    [ $TWO_LEVEL = 1 ] || return 0
    if [ $SGX = 0 ]; then
        echo "== production-like job, caffeine $1: the modified tree, two-level SCF"
        PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$P" --label two_level --basis $1 --set two_level=True \
            --compare "$R/prod_$1_base" --out "$J"
        return 0
    fi
    echo "== production-like job, caffeine $1: two-level SCF, the model's K in the cheap model"
    PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$P" --label two_level_exact_k --basis $1 --tasks scf \
        --set two_level=True --set two_level_sgx_level=None --compare "$R/prod_$1_base" --out "$J"
    echo "== production-like job, caffeine $1: two-level SCF, seminumerical K in the cheap model"
    PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$P" --label two_level --basis $1 --set two_level=True \
        --compare "$R/prod_$1_base" --out "$J"
    echo "== production-like job, caffeine $1: two-level SCF, seminumerical K and fitted J in the cheap model"
    PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$P" --label two_level_etb --basis $1 --tasks scf \
        --set two_level=True --set two_level_auxbasis=aug_etb --compare "$R/prod_$1_base" --out "$J"
}
# mf.two_level exists from part 36 on (a tree at an older pin would silently run the plain SCF), the seminumerical
# K of its cheap model from part 37 on
TWO_LEVEL=0
[ -f "$WS/gpu4pyscf/gpu4pyscf/scf/two_level.py" ] && TWO_LEVEL=1
SGX=0
[ -f "$WS/gpu4pyscf/gpu4pyscf/sgx/sgx.py" ] && SGX=1
{
    prod def2-svp
    if [ $QUICK = 0 ]; then
        prod def2-tzvp
        prod def2-tzvpd
    fi
    if [ $LARGE = 1 ] && [ $TWO_LEVEL = 1 ]; then
        NAD="$WS/bench/diag/wb97mv/nad+_mmff.xyz"
        echo "== production size, NAD+ def2-TZVPD: the modified tree, two-level SCF"
        PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$WS/bench/gpu/prod_parts.py" --label nad_two_level --mol "$NAD" \
            --charge 1 --basis def2-tzvpd --conv-tol 1e-9 --set two_level=True --save "$R/prod_nad" \
            --out "$R/prod_parts.jsonl"
        if [ $SGX = 1 ]; then
            echo "== production size, NAD+ def2-TZVPD: two-level SCF with a fitted J in the cheap model too"
            PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$WS/bench/gpu/prod_parts.py" --label nad_two_level_etb \
                --mol "$NAD" --charge 1 --basis def2-tzvpd --conv-tol 1e-9 --tasks scf --set two_level=True \
                --set two_level_auxbasis=aug_etb --compare "$R/prod_nad" --out "$R/prod_parts.jsonl"
        fi
        echo "== production size, NAD+ def2-TZVPD: the modified tree, plain SCF"
        PYTHONPATH="$WS/gpu4pyscf" python -W ignore "$WS/bench/gpu/prod_parts.py" --label nad_plain --mol "$NAD" \
            --charge 1 --basis def2-tzvpd --conv-tol 1e-9 --compare "$R/prod_nad" --out "$R/prod_parts.jsonl"
    fi
} 2>&1 | tee "$R/prod_log.txt"
echo "== done: $R (commit and push it, or copy it back)"
