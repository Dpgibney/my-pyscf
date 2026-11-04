#!/usr/bin/env bash
# The user skipped the original tree's default-grid wB97M-V gradient: once node2_wb97mv.sh has recorded its 4 SCFs,
# stop its first gradient worker (the original tree, default grid), let it end, and run the other three gradients.
cd ~/Chemistry/my-pyscf
f=$(ls -t bench/results/*node2-wb97mv.json | head -1)
nscf() { .venv/bin/python -c 'import json, sys; print(sum(r["case"] == "scf_caffeine_svp_wb97mv" for r in json.load(open(sys.argv[1]))["runs"]))' "$f" 2>/dev/null; }
until [ "$(nscf)" = 4 ]; do sleep 20; done
until p=$(pgrep -f "[w]orker.py grad_caffeine_svp_wb97mv"); do sleep 2; done
kill $p
while pgrep -f "[n]ode2_wb97mv.sh" > /dev/null; do sleep 5; done
{
echo "== wB97M-V gradients without the original tree's default grid ($(date +%H:%M)); stopped worker $p"
(cd bench && ../.venv/bin/python run_bench.py --cases grad_caffeine_svp_wb97mv --reps 1 --threads 6 \
    --tag node2-wb97mv-grad perf=pyscf:PYSCF_MAX_MEMORY=9000 \
    base_sg1=pyscf-base:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1 perf_sg1=pyscf:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1) \
    2>&1 | grep -v "RuntimeWarning"
echo NODE2 WB97MV GRAD DONE
} > scratch/node2_wb97mv_grad.log 2>&1
