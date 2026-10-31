#!/bin/bash
# Put a gpu4pyscf tree from this workspace on node1 and give it compiled libraries. Run it on the laptop:
#     bash scripts/node1_gpu4pyscf_tree.sh [TREE] [--wheel-libs]
# TREE is gpu4pyscf (the perf branch, default) or gpu4pyscf-base (the unmodified v1.9.0 worktree).
# The tree is synced with rsync (tracked and new files, mtimes kept, so rebuilds are incremental). By default the
# CUDA libraries are then built from source for the GTX 1650 Super (sm_75, ~/opt/cuda-12.9.1 from
# node1_cuda_toolkit.sh) in ~/Chemistry/my-pyscf/build/TREE; libxc is linked from the wheel. With --wheel-libs the
# v1.9.0 wheel's compiled libraries are linked instead, which only fits trees that differ from v1.9.0 in Python
# files alone (checked). `source scripts/node1-gpu-env.sh perf|base` then imports the tree.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE=gpu4pyscf
MODE=build
for a in "$@"; do
    case "$a" in
        --wheel-libs) MODE=wheel ;;
        gpu4pyscf|gpu4pyscf-base) TREE="$a" ;;
        *) echo "unknown argument $a" >&2; exit 1 ;;
    esac
done
BASE=v1.9.0   # the version of gpu4pyscf-cuda12x in requirements-gpu.txt
cd "$WS/$TREE"
if [ $MODE = wheel ]; then
    nonpy="$( (git diff --name-only "$BASE"; git ls-files --others --exclude-standard) | grep -v '\.py$' || true)"
    if [ -n "$nonpy" ]; then
        echo "$TREE differs from $BASE in non-Python files; the $BASE wheel's binaries would not match:" >&2
        echo "$nonpy" >&2
        exit 1
    fi
fi
git ls-files -z --cached --others --exclude-standard |
    rsync -a --from0 --files-from=- -e 'ssh -o BatchMode=yes' ./ "node1:Chemistry/my-pyscf/$TREE/"
ssh -o BatchMode=yes node1 "TREE=$TREE MODE=$MODE bash -s" <<'EOF'
set -euo pipefail
WS=~/Chemistry/my-pyscf; T=$WS/$TREE; LIB=$T/gpu4pyscf/lib
SP=$WS/.venv-gpu/lib/python3.12/site-packages/gpu4pyscf/lib
if [ $MODE = wheel ]; then
    find "$LIB" -maxdepth 1 -name '*.so' ! -type l -delete
    ln -sf "$SP"/*.so "$LIB/"
else
    # never let the build write through a link into the wheel
    find "$LIB" -maxdepth 1 -name '*.so' -type l -delete
    C=$HOME/opt/cuda-12.9.1; export PATH=$C/bin:$PATH
    B=$WS/build/$TREE
    [ -f "$B/CMakeCache.txt" ] || cmake -S "$LIB" -B "$B" -DCUDA_ARCHITECTURES=75-real -DBUILD_LIBXC=OFF \
        -DCMAKE_CUDA_COMPILER="$C/bin/nvcc" > "$B.configure.log" 2>&1
    cmake --build "$B" -j8 > "$B.build.log" 2>&1 || { tail -30 "$B.build.log"; exit 1; }
fi
mkdir -p "$LIB/deps" && ln -sfn "$SP/deps/lib" "$LIB/deps/lib"
echo "$TREE on node1: $(find "$LIB" -maxdepth 1 -name '*.so' ! -type l | wc -l) libraries built from source," \
     "$(find "$LIB" -maxdepth 1 -name '*.so' -type l | wc -l) linked from the wheel"
EOF
