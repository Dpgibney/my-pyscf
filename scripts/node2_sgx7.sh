#!/usr/bin/env bash
# It26 on node2, after node2_sgx6.sh: qcint sgx-exp e1314eb (the optimizer's pair data made by the loop's own
# CINTset_pairdata call for every ordered pair), PGO build qcint-sx-opt2; optcheck (against the per-call path; omega
# guard); gbench against qcint-sx-tr and qcint-sx-opt, and timings.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
{
while pgrep -f "[n]ode2_sgx6.sh" > /dev/null; do sleep 30; done
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup7
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in qcint/src/cint1e_grids.c scratch/qcint_perf_backup7/
tar xf scratch/sgx7_files.tar
cp qcint_opt2/* qcint/src/
bash scripts/build_qcint_pgo.sh sx-opt2 > $L/qcint_sx_opt2.log 2>&1 && echo "  built qcint-sx-opt2" || { echo "  build failed"; tail -n 20 $L/qcint_sx_opt2.log; }
cp scratch/qcint_perf_backup7/* qcint/src/
rm -f qcint/src/rys_roots_chunk.h
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
for m in caffeine c20; do scratch/optcheck install/qcint-sx-opt2/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin; done
for m in caffeine caffeine-tz c20; do
    for v in sx-tr sx-opt; do
        echo "$m qcint-sx-opt2 against qcint-$v: $(bench/qcint/gbench install/qcint-sx-opt2/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-$v/lib/libcint.so.6 | sed 's/^install.*integrals, //' | head -n 1)"
    done
    for lib in qcint-sx-tr qcint-sx-opt qcint-sx-opt2 qcint-sx-tr qcint-sx-opt qcint-sx-opt2; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | sed 's/^all *//')"
    done
done
echo NODE2 SGX7 DONE $(date +%H:%M)
} > scratch/node2_sgx7.log 2>&1
