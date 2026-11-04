#!/usr/bin/env bash
# node2: wB97M-V caffeine/def2-SVP SCF and gradient in the original and the modified PySCF, each with the
# default NLC (VV10) grid and with SG-1's (50, 194) pruned grid (bench/cases.py BENCH_NLC_GRID); 1 rep, 6 threads.
cd ~/Chemistry/my-pyscf
{
echo "== wB97M-V, default and SG-1 NLC grids ($(date +%H:%M))"
(cd bench && ../.venv/bin/python run_bench.py --cases scf_caffeine_svp_wb97mv,grad_caffeine_svp_wb97mv --reps 1 \
    --threads 6 --tag node2-wb97mv base=pyscf-base:PYSCF_MAX_MEMORY=9000 perf=pyscf:PYSCF_MAX_MEMORY=9000 \
    base_sg1=pyscf-base:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1 perf_sg1=pyscf:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1) \
    2>&1 | grep -v "RuntimeWarning"
echo NODE2 WB97MV DONE
} > scratch/node2_wb97mv.log 2>&1
