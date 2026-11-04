#!/usr/bin/env bash
# PCM's integral steps (_get_v, _get_vmat) at the block sizes of the PCM SCFs, in the original tree (pyscf-base):
# bench/diag/pcm_blk_time.py, run before in the modified tree (pyscf). Does the modified tree's slower PCM SCF
# (2704 s against 2494 s) come from these steps?
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
echo "== pyscf-base $(date +%H:%M)"
PYTHONPATH=pyscf-base OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 timeout 1800 python -W ignore bench/diag/pcm_blk_time.py
echo "   exit $?"
echo NODE2 PCMBLK DONE $(date +%H:%M)
