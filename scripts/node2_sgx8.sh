#!/usr/bin/env bash
# It26 on node2: the optimizer's pair data made by the loop's own call, in CINTOpt_setij's layout ((i, j) then
# (j, i)): PGO build qcint-sx-opt3; optcheck; gbench against qcint-sx-opt; timings of sx-opt, sx-opt2, sx-opt3.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup8
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in qcint/src/cint1e_grids.c scratch/qcint_perf_backup8/
tar xf scratch/sgx8_files.tar
cp qcint_opt3/* qcint/src/
bash scripts/build_qcint_pgo.sh sx-opt3 > $L/qcint_sx_opt3.log 2>&1 && echo "  built qcint-sx-opt3" || { echo "  build failed"; tail -n 20 $L/qcint_sx_opt3.log; }
cp scratch/qcint_perf_backup8/* qcint/src/
rm -f qcint/src/rys_roots_chunk.h
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
for m in caffeine c20; do scratch/optcheck install/qcint-sx-opt3/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin | head -n 2; done
for m in caffeine caffeine-tz c20; do
    echo "$m qcint-sx-opt3 against qcint-sx-opt: $(bench/qcint/gbench install/qcint-sx-opt3/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-sx-opt/lib/libcint.so.6 | sed 's/^install.*integrals, //' | head -n 1)"
    for lib in qcint-sx-opt qcint-sx-opt2 qcint-sx-opt3 qcint-sx-opt qcint-sx-opt2 qcint-sx-opt3; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | sed 's/^all *//')"
    done
done
echo NODE2 SGX8 DONE $(date +%H:%M)
} > scratch/node2_sgx8.log 2>&1
