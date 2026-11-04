#!/usr/bin/env bash
# B3LYP with PCM (C-PCM, water) on n-C26H54/def2-TZVP (80 atoms, 1130 AOs), the original tree (pyscf-base) and the
# modified one (pyscf), 6 threads, PySCF's default max_memory: bench/diag/pcm_memory.py samples the RSS by phase and
# stops a run at 13 GB (node2 has 15 GB); /usr/bin/time -v for the kernel's peak RSS.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
for tree in pyscf-base pyscf; do
    echo "== $tree $(date +%H:%M)"
    PYTHONPATH=$tree OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
        > $L/pcm_$tree.log 2>&1
    echo "   exit $?"
    grep -E "atoms|\] RSS|WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |End|Maximum resident|Elapsed|cycle=" $L/pcm_$tree.log | tail -n 45
done
echo NODE2 PCM DONE $(date +%H:%M)
