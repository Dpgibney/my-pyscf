#!/usr/bin/env bash
# qcint sgx-exp plus SIMD Rys roots in int1e_grids' Gaussian-charge (and long-range) branch, PCM's integrals, on
# node2: the PGO build qcint-sx-zeta; gbench of SGX's point-charge integrals against qcint-sx-opt3 (sgx-exp before
# the change, which leaves that branch alone); PCM's _get_v/_get_vmat (bench/diag/pcm_int_time.py,
# n-C26H54/def2-TZVP, the pyscf tree with the new PCM code) with qcint-pgo (perf), qcint-sx-opt3 and qcint-sx-zeta.
# Waits for scripts/node2_pcmcache.sh.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
while pgrep -f "[n]ode2_pcmcache.sh" > /dev/null; do sleep 30; done
{
echo "== start $(date +%H:%M)"
mkdir -p scratch/qcint_perf_backup9
cp qcint/src/g1e_grids.c qcint/src/rys_roots.c qcint/src/rys_roots.h qcint/src/cint_config.h.in qcint/src/cint1e_grids.c scratch/qcint_perf_backup9/
cp scratch/zeta_src/* qcint/src/
bash scripts/build_qcint_pgo.sh sx-zeta > $L/qcint_sx_zeta.log 2>&1 && echo "  built qcint-sx-zeta" || { echo "  build failed"; tail -n 20 $L/qcint_sx_zeta.log; }
cp scratch/qcint_perf_backup9/* qcint/src/
rm -f qcint/src/rys_roots_chunk.h
echo "  qcint sources back to perf: $(md5sum qcint/src/g1e_grids.c qcint/src/cint1e_grids.c qcint/src/cint_config.h.in | cut -c1-8 | tr '\n' ' ')"
for m in caffeine c20; do
    echo "$m qcint-sx-zeta against qcint-sx-opt3: $(bench/qcint/gbench install/qcint-sx-zeta/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 --compare install/qcint-sx-opt3/lib/libcint.so.6 | head -n 2 | tr '\n' ' ')"
    for lib in qcint-sx-opt3 qcint-sx-zeta qcint-sx-opt3 qcint-sx-zeta; do
        echo "$m $lib: $(bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40 | grep '^all' | sed 's/^all *//')"
    done
done
T=scratch/pcmzeta
mkdir -p $T
run() { LD_LIBRARY_PATH=$1 PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 timeout 3000 python -W ignore bench/diag/pcm_int_time.py --n 26 --basis def2-tzvp --reps 2 "${@:2}"; }
echo "-- qcint-pgo (perf)"; run "" --save $T/c26_pgo.npz --int3c
echo "-- qcint-sx-opt3 (sgx-exp before the change)"; run $PWD/install/qcint-sx-opt3/lib --ref $T/c26_pgo.npz
echo "-- qcint-sx-zeta"; run $PWD/install/qcint-sx-zeta/lib --ref $T/c26_pgo.npz --int3c
echo NODE2 ZETA DONE $(date +%H:%M)
} > scratch/node2_zeta.log 2>&1
