#!/usr/bin/env bash
# qcint perf with sgx-exp merged (54bef5c) on node2: the merged sources into qcint/src (to stay); the old PGO build kept
# as install/qcint-pgo-a95651f; install/qcint-pgo rebuilt (the trees' deps link it); qbench of the ERIs (caffeine SVP,
# and TZVP at --thr 1e-9) against the old build, differences and single-thread timings; then the B3LYP/C-PCM SCF of
# n-C26H54/def2-TZVP in the pyscf tree, with It9's rule (no ERI cache below 1% of the ERIs) and with 0.1%.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
O=scratch/node2_qcintmerge.log
{
echo "== start $(date +%H:%M)"
cp scratch/qcint_merged/* qcint/src/
echo "  qcint/src: $(md5sum qcint/src/g1e_grids.c qcint/src/cint1e_grids.c qcint/src/rys_roots.c | cut -c1-8 | tr '\n' ' ')"
[ -d install/qcint-pgo-a95651f ] || cp -a install/qcint-pgo install/qcint-pgo-a95651f
if bash scripts/build_qcint_pgo.sh > $L/qcint_pgo_merged.log 2>&1; then echo "  rebuilt qcint-pgo"; else
    echo "  build failed"; tail -n 20 $L/qcint_pgo_merged.log; echo "NODE2 QCINTMERGE DONE"; exit 1; fi
for m in caffeine caffeine-tz; do
    thr=$( [ $m = caffeine-tz ] && echo "--thr 1e-9" )
    echo "$m int2e, new against old: $(bench/qcint/qbench install/qcint-pgo/lib/libcint.so.6 scratch/qcint/$m.bin int2e $thr --compare install/qcint-pgo-a95651f/lib/libcint.so.6 2>&1 | head -n 3 | tr '\n' ' ')"
    for lib in qcint-pgo-a95651f qcint-pgo qcint-pgo-a95651f qcint-pgo; do
        echo "$m $lib: $(bench/qcint/qbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin int2e $thr --reps 2 2>&1 | head -n 1)"
    done
done
} > $O 2>&1
for sd in "" "--sd-min-fraction 0.001"; do
    tag=$( [ -z "$sd" ] && echo merged || echo merged-sd0.001 )
    echo "== pyscf $tag $(date +%H:%M)" >> $O
    PYTHONPATH=pyscf OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 /usr/bin/time -v \
        timeout 10800 python -W ignore bench/diag/pcm_memory.py --n 26 --basis def2-tzvp --limit-gb 13 $sd \
        > $L/pcm_pyscf-$tag.log 2>&1
    echo "   exit $?" >> $O
    grep -E "WATCHDOG|^E |^  *[0-9]+\.[0-9]+  |Maximum resident|semi-direct" $L/pcm_pyscf-$tag.log | tail -n 30 >> $O
done
echo "NODE2 QCINTMERGE DONE $(date +%H:%M)" >> $O
