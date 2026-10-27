#!/usr/bin/env bash
# Build OpenBLAS ($WS/OpenBLAS) into $INSTALL/openblas.
# DYNAMIC_ARCH with an explicit kernel list lets OPENBLAS_CORETYPE=Barcelona|Haswell|Zen|SkylakeX|Cooperlake
# switch kernels at runtime without rebuilding. USE_OPENMP=1 so dgemm calls made inside PySCF's OpenMP
# regions run single-threaded instead of oversubscribing. 32-bit integers (PySCF's fblas.h uses int).
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd "$WS/OpenBLAS"
make -j"$JOBS" DYNAMIC_ARCH=1 DYNAMIC_LIST="BARCELONA HASWELL ZEN SKYLAKEX COOPERLAKE" \
     USE_OPENMP=1 NUM_THREADS=64 NO_STATIC=1 CC="$CC" FC="$FC"
make PREFIX="$INSTALL/openblas" NO_STATIC=1 install
