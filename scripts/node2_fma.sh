#!/usr/bin/env bash
# It24 + FMA contraction in the VV10 functions (node2): the It24 build kept as pyscf-it24, the new nr_numint.c built
# into pyscf; the kernels on fixed inputs (3 reps), then the wB97M-V SCF at the default NLC grid, ABBA, 2 reps.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
{
echo "== keep It24 as pyscf-it24, build the FMA version ($(date +%H:%M))"
rm -rf pyscf-it24 && cp -a pyscf pyscf-it24
tar xf scratch/fma_files.tar
cmake --build pyscf/pyscf/lib/build -j 12 > scratch/node2_logs/fma_build.log 2>&1 && echo "  build ok" \
    || { echo "  build failed"; tail -n 20 scratch/node2_logs/fma_build.log; echo NODE2 FMA DONE; exit 1; }
objdump -d --no-show-raw-insn pyscf/pyscf/lib/libdft.so | grep -c "vfmadd\|vfmsub\|vfnmadd" | sed 's/^/  FMA instructions in libdft: /'
echo "== the kernels on fixed inputs (vv10_check.py, level-1 caffeine grid, 6 threads, best of 3)"
for t in pyscf-it24 pyscf pyscf-it24 pyscf; do
    PYTHONPATH=$t OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 python -W ignore bench/diag/vv10_check.py \
        scratch/vv10_in.npz scratch/vv10_fma_$t.npz --reps 3 | sed "s|.*pyscf/dft: |$t: |"
done
python -W ignore bench/diag/vv10_check.py --compare scratch/vv10_pyscf-base.npz scratch/vv10_fma_pyscf.npz
echo "== wB97M-V SCF, default NLC grid: It24 against It24 + FMA (ABBA, 2 reps)"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_wb97mv --reps 2 --threads 6 --tag node2-fma \
    it24=pyscf-it24:PYSCF_MAX_MEMORY=9000 fma=pyscf:PYSCF_MAX_MEMORY=9000) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 FMA DONE
} > scratch/node2_fma.log 2>&1
