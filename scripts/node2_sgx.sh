#!/usr/bin/env bash
# PySCF's seminumerical exchange (pyscf.sgx) against exact J/K on node2 (6 threads, 12 GB): B3LYP SCFs of caffeine
# (def2-SVP, def2-TZVP) and n-C20H42 (def2-SVP, def2-TZVP); exact in the modified (perf) and original (base) trees,
# and SGX in both trees, as sgx_fit's default (SGX J and K) and with pjs (DF-J, SGX K, P-junction screening).
# The original tree's exact runs come last in each group, as the slowest.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
M=PYSCF_MAX_MEMORY=12000
C="perf=pyscf:$M perf_pjs=pyscf:$M,BENCH_SGX=pjs perf_sgx=pyscf:$M,BENCH_SGX=sgx base_pjs=pyscf-base:$M,BENCH_SGX=pjs base_sgx=pyscf-base:$M,BENCH_SGX=sgx base=pyscf-base:$M"
{
echo "== caffeine def2-SVP, def2-TZVP and n-C20H42 def2-SVP B3LYP ($(date +%H:%M))"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp,scf_caffeine_tzvp_b3lyp,scf_c20h42_svp_b3lyp \
    --reps 1 --threads 6 --tag node2-sgx $C) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo "== n-C20H42 def2-TZVP B3LYP ($(date +%H:%M))"
(cd bench && python -W ignore run_bench.py --cases scf_c20h42_tzvp_b3lyp --reps 1 --threads 6 --tag node2-sgx-tzvp $C) \
    2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 SGX DONE
} > scratch/node2_sgx.log 2>&1
