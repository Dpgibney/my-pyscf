#!/usr/bin/env bash
# It26 on node2. (1) qcint sgx-exp 1f1b739 (the grid coordinates transposed by shuffles instead of gathers) and 1701582
# (+ the omega-guarded int1e_grids optimizer): PGO builds qcint-sx-tr, qcint-sx-opt; gbench against qcint-sx-fin (the
# committed It25) and timings; optcheck (the optimizer against the per-call path, and its omega guard). (2) pyscf-sgx26:
# pyscf-sgx with the AO values on the grid kept across K builds (sgx_jk.py), built on qcint-sx-opt; its K-build profile
# (pyscf-sgxprof26), also with 12 threads; the SGX SCFs against pyscf-sgx (It25), 12 GB, ABBA, 2 reps.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup6
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in qcint/src/cint1e_grids.c scratch/qcint_perf_backup6/
tar xf scratch/sgx6_files.tar
for v in tr opt; do
    echo "== qcint-sx-$v ($(date +%H:%M))"
    cp qcint_$v/* qcint/src/
    bash scripts/build_qcint_pgo.sh sx-$v > $L/qcint_sx_$v.log 2>&1 && echo "  built" || { echo "  build failed"; tail -n 20 $L/qcint_sx_$v.log; }
done
cp scratch/qcint_perf_backup6/* qcint/src/
rm -f qcint/src/rys_roots_chunk.h
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
gcc -O2 -o bench/qcint/gbench bench/qcint/gbench.c -ldl -lm
gcc -O2 -o scratch/optcheck bench/qcint/optcheck.c -ldl -lm
echo "== int1e_grids (gbench) $(date +%H:%M)"
for m in caffeine caffeine-tz c20; do
    for v in sx-tr sx-opt; do
        echo "$m qcint-$v against qcint-sx-fin: $(bench/qcint/gbench install/qcint-$v/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-sx-fin/lib/libcint.so.6 | sed 's/^install.*integrals, //' | head -n 1)"
    done
    for lib in qcint-pgo qcint-sx-fin qcint-sx-tr qcint-sx-opt qcint-pgo qcint-sx-fin qcint-sx-tr qcint-sx-opt; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | sed 's/^all *//')"
    done
done
echo "== optcheck (qcint-sx-opt) $(date +%H:%M)"
for m in caffeine c20; do scratch/optcheck install/qcint-sx-opt/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin; done
echo "== pyscf-sgx26 (pyscf-sgx with the AO values kept) on qcint-sx-opt $(date +%H:%M)"
rm -rf pyscf-sgx26 && cp -a pyscf-sgx pyscf-sgx26
cp scratch/sgx_jk_26.py pyscf-sgx26/pyscf/sgx/sgx_jk.py
bash scripts/build_pyscf.sh pyscf-sgx26 qcint-sx-opt libxc-stock -DNUMINT_X86_64_V3=ON > $L/sgx26_pyscf.log 2>&1 \
    && echo "  $(tail -n 1 $L/sgx26_pyscf.log)" || { echo "  build failed"; tail -n 20 $L/sgx26_pyscf.log; echo NODE2 SGX6 DONE; exit 1; }
rm -rf pyscf-sgxprof26 && cp -a pyscf-sgx26 pyscf-sgxprof26
sed -i '0,/#include <stdlib.h>/s//#define SGX_PROFILE\n#include <stdlib.h>/' pyscf-sgxprof26/pyscf/lib/vhf/nr_sgx_direct.c
bash scripts/build_pyscf.sh pyscf-sgxprof26 qcint-sx-opt libxc-stock -DNUMINT_X86_64_V3=ON > $L/sgx26_prof.log 2>&1 \
    && echo "  $(tail -n 1 $L/sgx26_prof.log)" || { echo "  build failed"; tail -n 20 $L/sgx26_prof.log; }
for c in "caffeine def2-svp --pjs" "caffeine def2-tzvp --pjs" "c20h42 def2-svp --pjs" "caffeine def2-svp"; do
    set -- $c
    PYTHONPATH=pyscf-sgxprof26 OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 PYSCF_MAX_MEMORY=12000 python -W ignore \
        bench/diag/sgx_profile.py --mol $1 --basis $2 ${3:-} --reps 2 2>&1 | grep -v RuntimeWarning
done
echo "== caffeine def2-SVP pjs with 12 threads (SMT) $(date +%H:%M)"
PYTHONPATH=pyscf-sgxprof26 OMP_NUM_THREADS=12 OPENBLAS_THREAD_TIMEOUT=20 PYSCF_MAX_MEMORY=12000 python -W ignore \
    bench/diag/sgx_profile.py --mol caffeine --basis def2-svp --pjs --reps 2 2>&1 | grep -v RuntimeWarning
echo "== SGX SCFs: pyscf-sgx (It25) against pyscf-sgx26 $(date +%H:%M)"
M=PYSCF_MAX_MEMORY=12000
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp,scf_caffeine_tzvp_b3lyp,scf_c20h42_svp_b3lyp \
    --reps 2 --threads 6 --tag node2-sgx6 old=pyscf-sgx:$M,BENCH_SGX=pjs new=pyscf-sgx26:$M,BENCH_SGX=pjs) 2>&1 | grep -v "RuntimeWarning\|WARNING"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp \
    --reps 2 --threads 6 --tag node2-sgx6-default old=pyscf-sgx:$M,BENCH_SGX=sgx new=pyscf-sgx26:$M,BENCH_SGX=sgx) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 SGX6 DONE $(date +%H:%M)
} > scratch/node2_sgx6.log 2>&1
