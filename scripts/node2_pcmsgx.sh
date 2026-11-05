#!/usr/bin/env bash
# PySCF's seminumerical exchange on the B3LYP/C-PCM system of the ORCA comparison (n-C26H54/def2-TZVP), in the pyscf
# tree (perf with the new PCM code, qcint perf with It25/It26), 6 threads, the default max_memory: sgx.sgx_fit with
# pjs (density-fitted J and SGX K, P-junction screening: the RIJCOSX-like setup), then the default mode (SGX for J
# and K). The density-fitted J tensor goes to disk (PYSCF_TMPDIR; node2's /tmp is tmpfs).
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
export PYSCF_TMPDIR=$PWD/scratch/tmp
mkdir -p $PYSCF_TMPDIR
L=scratch/node2_logs
O=scratch/node2_pcmsgx.log
{
for mode in pjs sgx; do
    echo "== $mode $(date +%H:%M)"
    PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 --sgx $mode \
        > $L/pcm_pyscf-sgx-$mode.log 2>&1
    echo "   exit $?"
    grep -E "WATCHDOG|^E |^SGX|^  *[0-9]+\.[0-9]+  |Maximum resident|Elapsed" $L/pcm_pyscf-sgx-$mode.log | tail -n 30
done
echo "NODE2 PCMSGX DONE $(date +%H:%M)"
} > $O 2>&1
