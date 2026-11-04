#!/usr/bin/env bash
# PCM's _get_v split into the int3c2e integrals and the einsum (bench/diag/pcm_int3c_time.py), in the original
# (pyscf-base) and the modified (pyscf) tree, twice each, alternating: which part makes the modified tree's PCM
# integral steps ~12% slower at the same block size?
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
for rep in 1 2; do
    for tree in pyscf-base pyscf; do
        echo "== $tree (rep $rep) $(date +%H:%M:%S)"
        PYTHONPATH=$tree OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 timeout 900 python -W ignore bench/diag/pcm_int3c_time.py 20 123
    done
done
echo NODE2 PCMINT3C DONE $(date +%H:%M)
