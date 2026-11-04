#!/usr/bin/env bash
# It25 step 1 on node2: qcint's int1e_grids with the large-x roots of a SIMD chunk at once (qcint sgx-exp), built with
# PGO as install/qcint-sgxpgo (node2's qcint sources restored afterwards); gbench against qcint-pgo; pyscf-sgx rebuilt
# on it (no profiling); the SGX (pjs) SCFs against the modified tree on qcint-pgo, 12 GB, ABBA, 2 reps.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
echo "== qcint-sgxpgo ($(date +%H:%M))"
mkdir -p scratch/qcint_perf_backup && cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h scratch/qcint_perf_backup/
tar xf scratch/sgx2_files.tar
bash scripts/build_qcint_pgo.sh sgxpgo > $L/qcint_sgxpgo.log 2>&1 && echo "  built" || { echo "  build failed"; tail -n 20 $L/qcint_sgxpgo.log; }
cp scratch/qcint_perf_backup/* qcint/src/
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c | cut -c1-8)"
echo "== int1e_grids: qcint-sgxpgo against qcint-pgo (gbench, single thread)"
for m in caffeine caffeine-tz c20; do
    bench/qcint/gbench install/qcint-sgxpgo/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-pgo/lib/libcint.so.6
    for lib in qcint-pgo qcint-sgxpgo qcint-pgo qcint-sgxpgo; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | awk '{print $3, "s,", $4, "ns/integral"}')"
    done
done
echo "== pyscf-sgx on qcint-sgxpgo"
sed -i '/^#define SGX_PROFILE$/d' pyscf-sgx/pyscf/lib/vhf/nr_sgx_direct.c
bash scripts/build_pyscf.sh pyscf-sgx qcint-sgxpgo libxc-stock -DNUMINT_X86_64_V3=ON > $L/sgx2_pyscf.log 2>&1 \
    && echo "  $(tail -n 1 $L/sgx2_pyscf.log)" || { echo "  build failed"; tail -n 20 $L/sgx2_pyscf.log; echo NODE2 SGX2 DONE; exit 1; }
echo "== SGX pjs SCFs: modified tree (qcint-pgo) against pyscf-sgx (qcint-sgxpgo)"
M=PYSCF_MAX_MEMORY=12000
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_b3lyp,scf_caffeine_tzvp_b3lyp,scf_c20h42_svp_b3lyp \
    --reps 2 --threads 6 --tag node2-sgx2 old=pyscf:$M,BENCH_SGX=pjs new=pyscf-sgx:$M,BENCH_SGX=pjs) 2>&1 | grep -v "RuntimeWarning\|WARNING"
echo NODE2 SGX2 DONE
} > scratch/node2_sgx2.log 2>&1
