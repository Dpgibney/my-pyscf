#!/usr/bin/env bash
# It25 step 2 on node2 (after node2_sgx2.sh): qcint int1e_grids with, on top of step 1 (the large-x roots of a SIMD
# chunk at once), the one-root chunks (rys_root1's fits evaluated on a whole chunk, selected by x) and GRID_BLKSIZE
# 112 (a 224-point SGX batch in two blocks instead of three), in two variants:
#   sameops: the operations of CINTrys_roots;
#   fewdiv:  t^2 = rt/x and F1/ww instead of u/(u+1) (two to four divisions fewer per chunk; last-bit changes);
#   root2:   fewdiv and the two-root chunks (rys_root2's fits on a whole chunk, CINTrys_root2_chunk);
#   vexp:    root2 with exp(-x) of the chunks' lanes in one vectorized loop (glibc's libmvec, as qcint's other exp loops).
# PGO builds as qcint-sx-<variant> (node2's qcint sources restored afterwards); gbench against
# qcint-pgo (bitwise comparison, timings); kbench (the SGX K contraction with PySCF's default flags and with
# NUMINT_X86_64_V3's); pyscf-sgx with nr_sgx_direct.c built with the kernel flags, on qcint-sx-vexp; its K-build
# profile (a copy with SGX_PROFILE); the SGX SCFs against the modified tree on qcint-pgo, 12 GB, ABBA, 2 reps.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
while pgrep -f "[n]ode2_sgx2.sh" > /dev/null; do sleep 30; done
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in scratch/qcint_perf_backup/
tar xf scratch/sgx3_files.tar
for v in sameops fewdiv root2 vexp; do
    echo "== qcint-sx-$v ($(date +%H:%M))"
    cp scratch/qcint_variants/g1e_grids.$v.c qcint/src/g1e_grids.c
    # rys_roots.c with CINTrys_largex_tables and CINTrys_root2_chunk (unused by sameops and fewdiv)
    r=root2; [ $v = vexp ] && r=vexp
    cp scratch/qcint_variants/rys_roots.$r.c qcint/src/rys_roots.c
    cp scratch/qcint_variants/rys_roots.$r.h qcint/src/rys_roots.h
    bash scripts/build_qcint_pgo.sh sx-$v > $L/qcint_sx_$v.log 2>&1 && echo "  built" || { echo "  build failed"; tail -n 20 $L/qcint_sx_$v.log; }
done
cp scratch/qcint_perf_backup/* qcint/src/
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
echo "== int1e_grids (gbench, single thread) $(date +%H:%M)"
gcc -O2 -o bench/qcint/gbench bench/qcint/gbench.c -ldl -lm
for m in caffeine caffeine-tz c20; do
    for v in sgxpgo sx-sameops sx-fewdiv sx-root2 sx-vexp; do
        echo "$m qcint-$v against qcint-pgo: $(bench/qcint/gbench install/qcint-$v/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-pgo/lib/libcint.so.6 | sed 's/^install.*integrals, //' | tr '\n' ' ')"
    done
    for lib in qcint-pgo qcint-sgxpgo qcint-sx-sameops qcint-sx-fewdiv qcint-sx-root2 qcint-sx-vexp qcint-pgo qcint-sgxpgo qcint-sx-sameops qcint-sx-fewdiv qcint-sx-root2 qcint-sx-vexp; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | awk '{print $3, "s,", $4, "ns/integral"}')"
    done
    bench/qcint/gbench install/qcint-sx-vexp/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | head -n 14
done
echo "== the SGX K contraction (kbench, single thread)"
gcc -O2 -msse3 bench/qcint/kbench.c -o scratch/kbench && gcc -O3 -march=x86-64-v3 -ffp-contract=off bench/qcint/kbench.c -o scratch/kbench_v3
for r in 1 2; do echo "  -O2 -msse3:  $(scratch/kbench 40)"; echo "  x86-64-v3:   $(scratch/kbench_v3 40)"; done
echo "== pyscf-sgx (nr_sgx_direct.c with the kernel flags) on qcint-sx-vexp $(date +%H:%M)"
bash scripts/build_pyscf.sh pyscf-sgx qcint-sx-vexp libxc-stock -DNUMINT_X86_64_V3=ON > $L/sgx3_pyscf.log 2>&1 \
    && echo "  $(tail -n 1 $L/sgx3_pyscf.log)" || { echo "  build failed"; tail -n 20 $L/sgx3_pyscf.log; echo NODE2 SGX3 DONE; exit 1; }
echo "  nr_sgx_direct.c flags: $(awk '/nr_sgx_direct.c.o:/{f=1} f&&/FLAGS/{print; f=0}' pyscf-sgx/pyscf/lib/build/build.ninja)"
echo "== pyscf-sgxprof (the same with SGX_PROFILE) $(date +%H:%M)"
rm -rf pyscf-sgxprof && cp -a pyscf-sgx pyscf-sgxprof
sed -i '0,/#include <stdlib.h>/s//#define SGX_PROFILE\n#include <stdlib.h>/' pyscf-sgxprof/pyscf/lib/vhf/nr_sgx_direct.c
bash scripts/build_pyscf.sh pyscf-sgxprof qcint-sx-vexp libxc-stock -DNUMINT_X86_64_V3=ON > $L/sgx3_prof.log 2>&1 \
    && echo "  $(tail -n 1 $L/sgx3_prof.log)" || { echo "  build failed"; tail -n 20 $L/sgx3_prof.log; }
for c in "caffeine def2-svp --pjs" "caffeine def2-tzvp --pjs" "c20h42 def2-svp --pjs"; do
    set -- $c
    PYTHONPATH=pyscf-sgxprof OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 PYSCF_MAX_MEMORY=12000 python -W ignore \
        bench/diag/sgx_profile.py --mol $1 --basis $2 ${3:-} --reps 2 2>&1 | grep -v RuntimeWarning
done
echo "== SGX pjs SCFs: modified tree (qcint-pgo) against pyscf-sgx (qcint-sx-vexp) $(date +%H:%M)"
M=PYSCF_MAX_MEMORY=12000
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp,scf_caffeine_tzvp_b3lyp,scf_c20h42_svp_b3lyp \
    --reps 2 --threads 6 --tag node2-sgx3 old=pyscf:$M,BENCH_SGX=pjs new=pyscf-sgx:$M,BENCH_SGX=pjs) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo "== SGX default-mode SCF $(date +%H:%M)"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp \
    --reps 2 --threads 6 --tag node2-sgx3-default old=pyscf:$M,BENCH_SGX=sgx new=pyscf-sgx:$M,BENCH_SGX=sgx) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 SGX3 DONE $(date +%H:%M)
} > scratch/node2_sgx3.log 2>&1
