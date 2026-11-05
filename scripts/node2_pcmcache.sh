#!/usr/bin/env bash
# B3LYP/C-PCM on n-C26H54/def2-TZVP (as scripts/node2_pcmint.sh) in the modified tree with pyscf branch pcm-cache's
# _attach_solvent.py: the solvent part of get_veff first, so PCM's matrices are there when the J/K (It9) and XC
# (It11) caches are sized at their first calls. pyscf-pcmcache is a copy of pyscf (perf with pcm-mem) with that file.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
rm -rf pyscf-pcmcache && cp -a pyscf pyscf-pcmcache
cp scratch/pcmcache/_attach_solvent.py pyscf-pcmcache/pyscf/solvent/_attach_solvent.py
for tree in pyscf-pcmcache; do
    echo "== $tree $(date +%H:%M)"
    PYTHONPATH=$tree OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
        > $L/pcm_$tree.log 2>&1
    echo "   exit $?"
    grep -E "atoms|WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |Maximum resident|Elapsed|semi-direct" $L/pcm_$tree.log | tail -n 40
done
echo NODE2 PCMCACHE DONE $(date +%H:%M)
