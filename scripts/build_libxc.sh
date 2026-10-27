#!/usr/bin/env bash
# Usage: build_libxc.sh <variant> [extra cmake args...]
# MAXORDER=3 matches PySCF's upstream build (-DDISABLE_KXC=0 -DDISABLE_LXC=1). Installs to $INSTALL/libxc-<variant>.
# libxc compiles as ISO C99 (FP contraction off), so -march changes speed but not results.
set -euo pipefail
. "$(dirname "$0")/env.sh"
variant=$1; shift
name="libxc-$variant"
cmake -S "$WS/libxc" -B "$BUILD/$name" -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DENABLE_FORTRAN=OFF -DMAXORDER=3 -DBUILD_TESTING=ON \
      -DCMAKE_INSTALL_PREFIX="$INSTALL/$name" -DCMAKE_INSTALL_LIBDIR=lib "$@"
cmake --build "$BUILD/$name" -j "${XC_JOBS:-8}"
cmake --install "$BUILD/$name"
