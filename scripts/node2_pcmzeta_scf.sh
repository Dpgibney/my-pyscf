#!/usr/bin/env bash
# The B3LYP/C-PCM SCF of n-C26H54/def2-TZVP (as scripts/node2_pcmint.sh) in the modified tree (pyscf: perf with the
# new PCM code and the solvent part of get_veff first), linked through LD_LIBRARY_PATH to qcint-sx-zeta: qcint
# sgx-exp (It25/It26) plus SIMD roots for Gaussian charges. What perf would give with sgx-exp merged.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
echo "== pyscf with qcint-sx-zeta $(date +%H:%M)"
LD_LIBRARY_PATH=$PWD/install/qcint-sx-zeta/lib PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 \
    /usr/bin/time -v timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
    > $L/pcm_pyscf-sxzeta.log 2>&1
echo "   exit $?"
grep -E "atoms|WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |Maximum resident|Elapsed|semi-direct" $L/pcm_pyscf-sxzeta.log | tail -n 40
echo NODE2 SXZETA DONE $(date +%H:%M)
