#!/usr/bin/env bash
# Usage: link_deps.sh <pyscf-tree> <cint-install> <libxc-install>
#   e.g. link_deps.sh pyscf-base libcint-stock libxc-stock
# PySCF hard-codes include/link dirs and RUNPATH to pyscf/lib/deps, so we symlink our builds there.
# The cint choice is build-time only: qcint's CINTEnvVars layout differs from libcint's, and PySCF's C
# code reads those structs, so a tree must be rebuilt (not LD_LIBRARY_PATH-swapped) to change it.
set -euo pipefail
. "$(dirname "$0")/env.sh"
tree=$1 cint=$2 xc=$3
deps="$WS/$tree/pyscf/lib/deps"
for d in "$INSTALL/$cint" "$INSTALL/$xc" "$INSTALL/openblas"; do
    [[ -d $d ]] || { echo "missing install: $d" >&2; exit 1; }
done
rm -rf "$deps"
mkdir -p "$deps/lib" "$deps/include"
ln -s "$INSTALL/$cint"/lib/libcint.so* "$deps/lib/"
ln -s "$INSTALL/$xc"/lib/libxc.so* "$deps/lib/"
ln -s "$INSTALL/openblas"/lib/libopenblas*.so* "$deps/lib/"
ln -s "$INSTALL/$cint"/include/*.h "$INSTALL/$xc"/include/*.h "$deps/include/"
printf 'cint=%s\nxc=%s\nblas=openblas\n' "$cint" "$xc" > "$deps/VARIANT"
