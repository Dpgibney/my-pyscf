#!/usr/bin/env bash
# Usage: build_qcint_pgo.sh [variant]      (default variant: pgo -> install/qcint-pgo)
# Profile-guided build of qcint (the perf branch as checked out): an instrumented build, the training workload
# of train_qcint_pgo.sh, then the build with the profiles, in the same build directory so that GCC finds them.
# -fprofile-partial-training keeps code the training does not reach (e.g. 3-center or spinor integrals)
# optimized as without a profile. LTO on top gained nothing (It8).
set -euo pipefail
. "$(dirname "$0")/env.sh"
variant=${1:-pgo}
# the molecule dumps first, so that the instrumented library sees the training alone
"$WS/scripts/train_qcint_pgo.sh" --dumps
rm -rf "$BUILD/qcint-$variant"
"$WS/scripts/build_cint.sh" qcint "$variant" "-DCMAKE_C_FLAGS=-fprofile-generate -fprofile-update=prefer-atomic"
"$WS/scripts/train_qcint_pgo.sh" "qcint-$variant"
"$WS/scripts/build_cint.sh" qcint "$variant" "-DCMAKE_C_FLAGS=-fprofile-use -fprofile-partial-training -Wno-missing-profile"
if objdump -d "$INSTALL/qcint-$variant/lib/libcint.so.6" | grep -q "__gcov"; then
    echo "instrumentation left in the final build" >&2
    exit 1
fi
echo "OK: $INSTALL/qcint-$variant (profile-guided)"
