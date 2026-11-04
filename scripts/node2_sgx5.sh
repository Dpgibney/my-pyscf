#!/usr/bin/env bash
# It25 final check on node2 (after node2_sgx4.sh): the committed qcint sgx-exp (3659e67: the one-root chunk inlined
# from rys_roots_chunk.h, CINTrys_roots_chunk for the ERIs and two roots, GRID_BLKSIZE 112), PGO build
# qcint-sx-fin; gbench against qcint-sx-vexp (expected bitwise) and timings; qbench int2e against qcint-sx-uni
# (expected bitwise) and timings, single thread.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
while pgrep -f "[n]ode2_sgx4.sh" > /dev/null; do sleep 30; done
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup3
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in scratch/qcint_perf_backup3/
tar xf scratch/sgx5_files.tar
cp qcint_fin/* qcint/src/
bash scripts/build_qcint_pgo.sh sx-fin > $L/qcint_sx_fin.log 2>&1 && echo "  built qcint-sx-fin" || { echo "  build failed"; tail -n 20 $L/qcint_sx_fin.log; }
cp scratch/qcint_perf_backup3/* qcint/src/
rm -f qcint/src/rys_roots_chunk.h
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' '), $(cd qcint && git status --short | wc -l) changed files"
for m in caffeine caffeine-tz c20; do
    echo "$m qcint-sx-fin against qcint-sx-vexp: $(bench/qcint/gbench install/qcint-sx-fin/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-sx-vexp/lib/libcint.so.6 | sed 's/^install.*integrals, //' | head -n 1)"
    for lib in qcint-pgo qcint-sx-vexp qcint-sx-fin qcint-pgo qcint-sx-vexp qcint-sx-fin; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | awk '{print $3, "s,", $4, "ns/integral"}')"
    done
done
echo "caffeine int2e qcint-sx-fin against qcint-sx-uni: $(bench/qcint/qbench install/qcint-sx-fin/lib/libcint.so.6 scratch/qcint/caffeine.bin int2e --compare install/qcint-sx-uni/lib/libcint.so.6 2>&1 | tail -n 1)"
for lib in qcint-pgo qcint-sx-fin qcint-pgo qcint-sx-fin; do
    echo "  $lib: $(bench/qcint/qbench install/$lib/lib/libcint.so.6 scratch/qcint/caffeine.bin int2e --reps 2 2>&1 | head -1 | grep -o '[0-9.]* s per pass.*')"
done
echo NODE2 SGX5 DONE $(date +%H:%M)
} > scratch/node2_sgx5.log 2>&1
