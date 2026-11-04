#!/usr/bin/env bash
# It25 step 3 on node2 (after node2_sgx3.sh): qcint "uni" - the vector roots of step 2 (vexp) moved into
# CINTrys_roots_chunk (rys_roots.c), which CINTg0_1e_grids and _CINTrys_roots_batch (the ERIs, nuclear
# attraction, 3c1e) both call. PGO build qcint-sx-uni; gbench against qcint-pgo and qcint-sx-vexp; qbench (int2e,
# ip1) against qcint-pgo: bitwise comparison and timings, single thread. Also qcint-sx-uni104: GRID_BLKSIZE 104.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
while pgrep -f "[n]ode2_sgx3.sh" > /dev/null; do sleep 30; done
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup2
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in scratch/qcint_perf_backup2/
tar xf scratch/sgx4_files.tar
tar xf scratch/sgx3_files.tar qcint/src/cint_config.h.in   # GRID_BLKSIZE 112
cp scratch/qcint_variants/g1e_grids.uni.c qcint/src/g1e_grids.c
cp scratch/qcint_variants/rys_roots.uni.c qcint/src/rys_roots.c
cp scratch/qcint_variants/rys_roots.uni.h qcint/src/rys_roots.h
bash scripts/build_qcint_pgo.sh sx-uni > $L/qcint_sx_uni.log 2>&1 && echo "  built qcint-sx-uni" || { echo "  build failed"; tail -n 20 $L/qcint_sx_uni.log; }
# the same with GRID_BLKSIZE 104 (perf's), to see what the block size does alone
sed -i 's/^#define GRID_BLKSIZE    112$/#define GRID_BLKSIZE    104/' qcint/src/cint_config.h.in
bash scripts/build_qcint_pgo.sh sx-uni104 > $L/qcint_sx_uni104.log 2>&1 && echo "  built qcint-sx-uni104" || { echo "  build failed"; tail -n 20 $L/qcint_sx_uni104.log; }
cp scratch/qcint_perf_backup2/* qcint/src/
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
echo "== int1e_grids (gbench) $(date +%H:%M)"
for m in caffeine caffeine-tz c20; do
    for v in sx-vexp pgo; do
        echo "$m qcint-sx-uni against qcint-$v: $(bench/qcint/gbench install/qcint-sx-uni/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-$v/lib/libcint.so.6 | sed 's/^install.*integrals, //' | tr '\n' ' ')"
    done
    for lib in qcint-pgo qcint-sx-vexp qcint-sx-uni qcint-sx-uni104 qcint-pgo qcint-sx-vexp qcint-sx-uni qcint-sx-uni104 qcint-sx-uni qcint-sx-uni104; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | awk '{print $3, "s,", $4, "ns/integral"}')"
    done
done
echo "== ERIs (qbench) $(date +%H:%M)"
for c in "caffeine int2e" "caffeine-tz int2e --thr 1e-9" "benzene ip1" "caffeine ip1"; do
    set -- $c
    echo "$1 $2: $(bench/qcint/qbench install/qcint-sx-uni/lib/libcint.so.6 scratch/qcint/$1.bin $2 ${3:-} ${4:-} --compare install/qcint-pgo/lib/libcint.so.6 2>&1 | tail -n 1)"
    for lib in qcint-pgo qcint-sx-uni qcint-pgo qcint-sx-uni; do
        echo "  $lib: $(bench/qcint/qbench install/$lib/lib/libcint.so.6 scratch/qcint/$1.bin $2 ${3:-} ${4:-} --reps 2 2>&1 | head -1 | grep -o '[0-9.]* s per pass.*')"
    done
done
echo NODE2 SGX4 DONE $(date +%H:%M)
} > scratch/node2_sgx4.log 2>&1
