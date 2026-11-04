#!/usr/bin/env bash
# It24 + FMA + one division per two pairs (node2): the FMA build kept as pyscf-fma, the new nr_numint.c built into
# pyscf; the kernels on fixed inputs (3 reps), then the wB97M-V SCF at the default NLC grid, ABBA, 2 reps.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
{
echo "== keep the FMA build as pyscf-fma, build the shared-division version ($(date +%H:%M))"
rm -rf pyscf-fma && cp -a pyscf pyscf-fma
tar xf scratch/recip2_files.tar
cmake --build pyscf/pyscf/lib/build -j 12 > scratch/node2_logs/recip2_build.log 2>&1 && echo "  build ok" \
    || { echo "  build failed"; tail -n 20 scratch/node2_logs/recip2_build.log; echo NODE2 RECIP2 DONE; exit 1; }
echo "== the kernels on fixed inputs (vv10_check.py, level-1 caffeine grid, 6 threads, best of 3)"
for t in pyscf-fma pyscf pyscf-fma pyscf; do
    PYTHONPATH=$t OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 python -W ignore bench/diag/vv10_check.py \
        scratch/vv10_in.npz scratch/vv10_r2_$t.npz --reps 3 | sed "s|.*pyscf/dft: |$t: |"
done
python -W ignore bench/diag/vv10_check.py --compare scratch/vv10_pyscf-base.npz scratch/vv10_r2_pyscf.npz
echo "== wB97M-V SCF, default NLC grid: the FMA build against shared divisions (ABBA, 2 reps)"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_wb97mv --reps 2 --threads 6 --tag node2-recip2 \
    fma=pyscf-fma:PYSCF_MAX_MEMORY=9000 recip2=pyscf:PYSCF_MAX_MEMORY=9000) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 RECIP2 DONE
} > scratch/node2_recip2.log 2>&1
