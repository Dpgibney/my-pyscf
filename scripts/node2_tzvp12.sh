#!/usr/bin/env bash
# After node2_run.sh: the modified tree's caffeine/def2-TZVP B3LYP SCF with PYSCF_MAX_MEMORY=12000 (a larger
# semi-direct cache than the benchmark's 9000), 2 reps, 6 threads.
cd ~/Chemistry/my-pyscf
until grep -q "NODE2 DONE" scratch/node2_run.log || ! pgrep -f "[n]ode2_run.sh" > /dev/null; do sleep 30; done
{
echo "== caffeine def2-TZVP B3LYP SCF, modified tree, PYSCF_MAX_MEMORY=12000 ($(date +%H:%M))"
free -g | head -2
(cd bench && ../.venv/bin/python run_bench.py --cases scf_caffeine_tzvp_b3lyp --reps 2 --threads 6 \
    --tag node2-tzvp12 perf12=pyscf:PYSCF_MAX_MEMORY=12000) 2>&1 | grep -v "RuntimeWarning"
f=$(ls -t bench/results/*node2-tzvp12.json | head -1)
.venv/bin/python -c "
import json, sys
for r in json.load(open(sys.argv[1]))['runs']:
    print(f\"  {r['case']} {r['config']} rep{r['rep']}: {r['wall']:.1f} s, peak RSS {r['max_rss_mb'] / 1024:.1f} GB, \"
          f\"cycles {r['values'].get('cycles')}, E {r['values'].get('e_tot'):.10f}\")
" $f
echo NODE2 TZVP12 DONE
} > scratch/node2_tzvp12.log 2>&1
