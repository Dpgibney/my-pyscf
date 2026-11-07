#!/usr/bin/env bash
# PySCF's RIJCOSX on node2 (6 threads, PySCF's default max_memory) against exact and RIJK, beside the ORCA runs
# of scripts/node2_orca_rx.sh: caffeine and n-C20H42/def2-TZVP B3LYP. pyscf-rx = the rijcosx branch (git archive,
# built by scripts/build_pyscf.sh). RIJCOSX before (SGX's default screening, the int3c2e RI-J: BENCH_RIJ_ENGINE=0)
# and after (the J-engine and sgx_tol_pair 1e-9). bench/diag/rijcosx.py samples memory and disk.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 PYSCF_CONFIG_FILE=$PWD/bench/pyscf_conf.py
unset PYSCF_MAX_MEMORY
R=scratch/node2_rijcosx
mkdir -p $R
run() { PYTHONPATH=pyscf-rx python -W ignore bench/diag/rijcosx.py --basis def2-tzvp --ref $R/ref.json --json $R/runs.jsonl "$@" 2>&1 | grep -v "Warning\|plugins\|PYSCF_EXT\|^/home\|K build:"; }
for m in caffeine c20h42; do
    run --mol $m --method exact --tag node2
    run --mol $m --method rijk --tag node2
    BENCH_RIJ_ENGINE=0 run --mol $m --method rijcosx --tag "node2 before"
    run --mol $m --method rijcosx --pair-tol 1e-9 --tag "node2 after"
done
echo NODE2 RIJCOSX DONE
