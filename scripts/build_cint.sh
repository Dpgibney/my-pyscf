#!/usr/bin/env bash
# Usage: build_cint.sh <libcint|qcint> <variant> [extra cmake args...]
# Configures with the options PySCF's ExternalProject passes (pyscf/lib/CMakeLists.txt) and installs
# to $INSTALL/<src>-<variant>, e.g. build_cint.sh libcint stock; build_cint.sh libcint native -DBUILD_MARCH_NATIVE=ON
set -euo pipefail
. "$(dirname "$0")/env.sh"
src=$1 variant=$2; shift 2
name="$src-$variant"
cmake -S "$WS/$src" -B "$BUILD/$name" -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DWITH_RANGE_COULOMB=1 -DWITH_F12=1 -DWITH_4C1E=1 -DWITH_FORTRAN=0 -DWITH_CINT2_INTERFACE=0 \
      -DMIN_EXPCUTOFF=20 -DKEEP_GOING=1 \
      -DCMAKE_INSTALL_PREFIX="$INSTALL/$name" -DCMAKE_INSTALL_LIBDIR=lib "$@"
cmake --build "$BUILD/$name" -j "$JOBS"
cmake --install "$BUILD/$name"
