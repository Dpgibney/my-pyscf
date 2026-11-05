#!/usr/bin/env bash
# PCM's surface integrals in one call of SGXnr_direct_drv_batches (the batches of all exponents spread over the
# threads), on node2: nr_sgx_direct.c and pcm.py into the pyscf tree, its C libraries rebuilt; pcm_int_time.py on
# n-C26H54/def2-TZVP against the saved run of the per-exponent calls (scratch/pcmzeta/c26_pgo.npz, the old qcint);
# then the B3LYP/C-PCM SCF.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
O=scratch/node2_pcmpar.log
{
echo "== start $(date +%H:%M)"
cp scratch/pcmpar/nr_sgx_direct.c pyscf/pyscf/lib/vhf/nr_sgx_direct.c
cp scratch/pcmpar/pcm.py pyscf/pyscf/solvent/pcm.py
cmake --build pyscf/pyscf/lib/build -j 6 > $L/pcmpar_build.log 2>&1 && echo "  rebuilt" || { echo "  build failed"; tail -n 20 $L/pcmpar_build.log; }
nm -D pyscf/pyscf/lib/libcvhf.so | grep -c SGXnr_direct_drv_batches
PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 timeout 3000 python -W ignore bench/diag/pcm_int_time.py --n 26 --basis def2-tzvp --reps 2 --ref scratch/pcmzeta/c26_pgo.npz --int3c
echo "== pyscf SCF $(date +%H:%M)"
PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
    timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 \
    > $L/pcm_pyscf-pcmpar.log 2>&1
echo "   exit $?"
grep -E "WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |Maximum resident|semi-direct" $L/pcm_pyscf-pcmpar.log | tail -n 30
echo "NODE2 PCMPAR DONE $(date +%H:%M)"
} > $O 2>&1
