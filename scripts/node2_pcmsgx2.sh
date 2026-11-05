#!/usr/bin/env bash
# PySCF's SGX (pjs) on the B3LYP/C-PCM system (n-C26H54/def2-TZVP) set up as ORCA's RIJCOSX: the J fitting basis
# def2-universal-jfit (ORCA's def2/J has the same 1868 functions) instead of PySCF's predefined def2-tzvp-jkfit, and
# SGX grid levels as ORCA's GRIDX1/2/3 (Lebedev 50/110/194 at most):
#   orca:  levels 0 -> 1 during the SCF, then the energy once on level 2 (ORCA recomputes the exchange on GRIDX3);
#   jfit:  only the J basis changed, level 2 throughout (SGX's default);
#   stage: levels 0 -> 2, self-consistent on level 2 at the end.
# 6 threads, the default max_memory; the J tensor on disk (PYSCF_TMPDIR; /tmp is tmpfs).
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
export PYSCF_TMPDIR=$PWD/scratch/tmp
mkdir -p $PYSCF_TMPDIR
L=scratch/node2_logs
O=scratch/node2_pcmsgx2.log
{
for run in orca jfit stage; do
    case $run in
        orca)  opts="--sgx-levels 0 1 --sgx-final-level 2";;
        jfit)  opts="";;
        stage) opts="--sgx-levels 0 2";;
    esac
    echo "== $run ($opts) $(date +%H:%M)"
    PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
        --sgx pjs --auxbasis def2-universal-jfit $opts > $L/pcm_pyscf-sgx-$run.log 2>&1
    echo "   exit $?"
    grep -E "WATCHDOG|^E |^SGX|^final|^  *[0-9]+\.[0-9]+  |Maximum resident|Elapsed" $L/pcm_pyscf-sgx-$run.log | tail -n 30
done
echo "NODE2 PCMSGX2 DONE $(date +%H:%M)"
} > $O 2>&1
