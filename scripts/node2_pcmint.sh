#!/usr/bin/env bash
# B3LYP/C-PCM on n-C26H54/def2-TZVP, as scripts/node2_pcm.sh, with the PCM code of pyscf branch pcm-mem: the surface
# integrals through int1e_grids, K factorized once, no D or dense R for C-PCM. pyscf-pcmint and pyscf-base-pcmint are
# copies of the modified (pyscf) and original (pyscf-base) trees with the branch's pcm.py, grad/pcm.py and
# hessian/pcm.py (pyscf/solvent is the same in both trees before the branch). bench/diag/pcm_memory.py now also times
# each phase.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
for t in pyscf pyscf-base; do
    rm -rf $t-pcmint && cp -a $t $t-pcmint
    cp scratch/pcmint/pcm.py $t-pcmint/pyscf/solvent/pcm.py
    cp scratch/pcmint/grad_pcm.py $t-pcmint/pyscf/solvent/grad/pcm.py
    cp scratch/pcmint/hessian_pcm.py $t-pcmint/pyscf/solvent/hessian/pcm.py
done
for tree in pyscf-pcmint pyscf-base-pcmint; do
    echo "== $tree $(date +%H:%M)"
    PYTHONPATH=$tree OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
        > $L/pcm_$tree.log 2>&1
    echo "   exit $?"
    grep -E "atoms|WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |Maximum resident|Elapsed|semi-direct" $L/pcm_$tree.log | tail -n 40
done
echo NODE2 PCMINT DONE $(date +%H:%M)
