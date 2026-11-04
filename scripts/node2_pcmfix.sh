#!/usr/bin/env bash
# After node2_pcm.sh: the same B3LYP/PCM run of n-C26H54/def2-TZVP in pyscf-pcmfix, a copy of the modified tree
# (pyscf) whose pcm.py makes get_D_S's surface matrices a block of rows at a time (bitwise the same S and D).
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
while pgrep -f "[n]ode2_pcm.sh" > /dev/null; do sleep 30; done
echo "== pyscf-pcmfix $(date +%H:%M)"
rm -rf pyscf-pcmfix && cp -a pyscf pyscf-pcmfix
cp scratch/pcm_blocked.py pyscf-pcmfix/pyscf/solvent/pcm.py
PYTHONPATH=pyscf-pcmfix OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
    timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
    > $L/pcm_pyscf-pcmfix.log 2>&1
echo "   exit $?"
grep -E "atoms|\] RSS|WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |End|Maximum resident|Elapsed|cycle=" $L/pcm_pyscf-pcmfix.log | tail -n 45
echo NODE2 PCMFIX DONE $(date +%H:%M)
