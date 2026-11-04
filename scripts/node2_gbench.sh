#!/usr/bin/env bash
# int1e_grids by class on node2 (gbench.c, single thread): libcint-stock (the original tree's) against qcint-pgo
# (the modified tree's), on the SGX grids of caffeine def2-SVP, def2-TZVP and n-C20H42 def2-SVP.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
{
gcc -O2 -Wall -o bench/qcint/gbench bench/qcint/gbench.c -ldl -lm && echo "gbench built"
[ -f scratch/qcint/c20.bin ] || PYTHONPATH=pyscf python -W ignore bench/qcint/dump_basis.py c20 scratch/qcint/c20.bin
for m in caffeine caffeine-tz c20; do
    [ -f scratch/qcint/${m}_sgx.bin ] || PYTHONPATH=pyscf OMP_NUM_THREADS=6 python -W ignore bench/qcint/dump_sgx_grid.py $m scratch/qcint/${m}_sgx.bin
done
for m in caffeine caffeine-tz c20; do
    for lib in libcint-stock qcint-pgo; do
        echo "== $m $lib"
        bench/qcint/gbench install/$lib/lib/libcint.so.6 scratch/qcint/$m.bin scratch/qcint/${m}_sgx.bin --batches 40
    done
done
echo NODE2 GBENCH DONE
} > scratch/node2_gbench.log 2>&1
