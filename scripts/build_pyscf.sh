#!/usr/bin/env bash
# Usage: build_pyscf.sh <pyscf-tree> <cint-install> <libxc-install> [extra cmake args...]
#   e.g. build_pyscf.sh pyscf-base libcint-stock libxc-stock
#        build_pyscf.sh pyscf libcint-stock libxc-stock -DBUILD_MARCH_NATIVE=ON
# In-place build (the .so files land in <tree>/pyscf/lib, as upstream CI does). xcfun is disabled; its
# imports are guarded. SMD's CDS term needs libsolvent (Fortran), which upstream builds only with ENABLE_SMD.
# Fails if any PySCF library resolves libcint/libxc/BLAS outside deps/.
set -euo pipefail
. "$(dirname "$0")/env.sh"
tree=$1 cint=$2 xc=$3; shift 3
"$WS/scripts/link_deps.sh" "$tree" "$cint" "$xc"
lib="$WS/$tree/pyscf/lib"
rm -rf "$lib/build"
cmake -S "$lib" -B "$lib/build" -GNinja -DBUILD_LIBCINT=OFF -DBUILD_LIBXC=OFF -DENABLE_XCFUN=OFF -DENABLE_SMD=ON \
      -DBLAS_LIBRARIES="$lib/deps/lib/libopenblas.so" "$@"
cmake --build "$lib/build" -j "$JOBS"
bad=0
for so in "$lib"/lib*.so; do
    while read -r dep path; do
        case $dep in libcint*|libxc.*|libopenblas*)
            [[ $path == "$lib/deps/lib/"* ]] || { echo "BAD: $(basename "$so") -> $dep => $path" >&2; bad=1; } ;;
        esac
    done < <(ldd "$so" | awk '$2 == "=>" {print $1, $3}')
done
[[ $bad == 0 ]] && echo "OK: all PySCF libs in $tree resolve cint/xc/BLAS from deps ($(tr '\n' ' ' < "$lib/deps/VARIANT"))"
exit $bad
