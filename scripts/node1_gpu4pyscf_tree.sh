#!/bin/bash
# Copy the gpu4pyscf perf tree (the working tree of $WS/gpu4pyscf, tracked and new files) to node1 and link the
# compiled libraries of node1's gpu4pyscf wheel into it, so `source scripts/node1-gpu-env.sh perf` imports it.
# The wheel's binaries only fit the tree while it differs from the wheel's version in Python files alone, which
# this checks first. Run it on the laptop: bash scripts/node1_gpu4pyscf_tree.sh
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=v1.9.0   # the version of gpu4pyscf-cuda12x in requirements-gpu.txt
cd "$WS/gpu4pyscf"
nonpy="$( (git diff --name-only "$BASE"; git ls-files --others --exclude-standard) | grep -v '\.py$' || true)"
if [ -n "$nonpy" ]; then
    echo "gpu4pyscf differs from $BASE in non-Python files; the $BASE wheel's binaries would not match:" >&2
    echo "$nonpy" >&2
    exit 1
fi
git ls-files -z --cached --others --exclude-standard | tar --null -T - -cf - |
ssh -o BatchMode=yes node1 'set -euo pipefail
    WS=~/Chemistry/my-pyscf; T=$WS/gpu4pyscf
    rm -rf "$T" && mkdir -p "$T" && tar xf - -C "$T"
    SP=$WS/.venv-gpu/lib/python3.12/site-packages/gpu4pyscf/lib
    ln -sf "$SP"/*.so "$T/gpu4pyscf/lib/"
    mkdir -p "$T/gpu4pyscf/lib/deps" && ln -sfn "$SP/deps/lib" "$T/gpu4pyscf/lib/deps/lib"
    echo "perf tree on node1: $T ($(ls "$T"/gpu4pyscf/lib/*.so | wc -l) libraries linked from the wheel)"'
