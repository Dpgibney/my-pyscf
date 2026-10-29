#!/usr/bin/env bash
# Usage: train_qcint_pgo.sh <qcint-variant>
# PGO training workload for an instrumented qcint build (install/<qcint-variant>): bench/qcint/qbench over the
# integrals PySCF uses most (SCF ERIs, gradient and hessian derivative integrals, Cartesian and
# range-separated ERIs) on def2-SVP/TZVP molecules. The molecule dumps come from bench/qcint/dump_basis.py.
set -euo pipefail
. "$(dirname "$0")/env.sh"
lib="$INSTALL/$1/lib/libcint.so.6"
mols="$WS/scratch/qcint"
for m in water benzene caffeine caffeine-tz; do
    [[ -f $mols/$m.bin ]] || "$WS/.venv/bin/python" "$WS/bench/diag/run_in.py" X=pyscf 4 \
        "$WS/bench/qcint/dump_basis.py" "$m" "$mols/$m.bin"
done
q() { "$WS/bench/qcint/qbench" "$lib" "$@" --reps 1 > /dev/null; }
q "$mols/caffeine.bin" int2e
q "$mols/caffeine-tz.bin" int2e --thr 1e-9
q "$mols/benzene.bin" ip1
for mode in ipip1 ipvip1 ip1ip2 cart sr lr; do q "$mols/water.bin" "$mode" --thr 0; done
q "$mols/benzene.bin" ipip1 --thr 1e-8
