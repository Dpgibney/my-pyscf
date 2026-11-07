#!/usr/bin/env bash
# Usage: train_qcint_pgo.sh <qcint-variant>      (or --dumps: only write the molecule dumps that are missing)
# PGO training workload for an instrumented qcint build (install/<qcint-variant>): bench/qcint/qbench over the
# integrals PySCF uses most (SCF ERIs, gradient and hessian derivative integrals, Cartesian and
# range-separated ERIs) on def2-SVP/TZVP molecules, and the batched int2e_sph and derivatives through
# bench/qcint/bbench.
# The molecule dumps (scratch/qcint) come from
# bench/qcint/dump_basis.py run with the workspace's pyscf tree, which must be built.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mols="$WS/scratch/qcint"
mkdir -p "$mols"
for m in water benzene caffeine caffeine-tz; do
    [[ -f $mols/$m.bin ]] || "$WS/.venv/bin/python" "$WS/bench/diag/run_in.py" X=pyscf 4 \
        "$WS/bench/qcint/dump_basis.py" "$m" "$mols/$m.bin"
done
[[ $1 == --dumps ]] && exit 0
lib="$INSTALL/$1/lib/libcint.so.6"
q() { "$WS/bench/qcint/qbench" "$lib" "$@" --reps 1 > /dev/null; }
q "$mols/caffeine.bin" int2e
q "$mols/caffeine-tz.bin" int2e --thr 1e-9
q "$mols/benzene.bin" ip1
for mode in ipip1 ipvip1 ip1ip2 cart sr lr; do q "$mols/water.bin" "$mode" --thr 0; done
q "$mols/benzene.bin" ipip1 --thr 1e-8
# the batched int2e_sph (int2e_sph_batch, several quartets of a class in the SIMD lanes), which PySCF's
# direct J/K uses for most quartets: its classes on caffeine, full and long range
if nm -D "$lib" | grep -q " int2e_sph_batch$"; then
    b() { "$WS/bench/qcint/bbench" "$lib" "$@" --reps 1 > /dev/null; }
    b "$mols/caffeine.bin" --sample 2000
    b "$mols/caffeine.bin" --sample 2000 --lr
    b "$mols/caffeine-tz.bin" --sample 500
    if nm -D "$lib" | grep -q " int2e_ip1_sph_batch$"; then
        # the gradients' int2e_ip1
        b "$mols/caffeine.bin" --sample 1000 --ip1
        b "$mols/caffeine-tz.bin" --sample 200 --ip1
    fi
    if nm -D "$lib" | grep -q " int2e_ipip1_sph_batch$"; then
        # the Hessians' int2e_ipip1, ipvip1 and ip1ip2 (caffeine/def2-TZVP has every class up to f)
        for h in ipip1 ipvip1 ip1ip2; do
            b "$mols/caffeine-tz.bin" --sample 50 --int $h
        done
    fi
fi
