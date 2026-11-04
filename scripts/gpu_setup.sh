#!/bin/bash
# Set up the GPU benchmarks on a machine with an NVIDIA GPU (compute capability 7.0 or newer) and a CUDA 12 toolkit:
# a venv with the pinned packages (scripts/requirements-gpu.txt), and the two GPU4PySCF trees of scripts/pins.txt,
# both built from source with the same toolchain: gpu4pyscf (the modified tree) and gpu4pyscf-base (v1.9.0).
#     git clone https://github.com/Dpgibney/my-pyscf && cd my-pyscf && bash scripts/gpu_setup.sh
# Needs nvcc from a CUDA 12 toolkit (e.g. `module load cuda/12.x`), cmake 3.18 or newer, gcc, git, and python3
# (3.10-3.12) with venv. On clusters whose compute nodes have no internet, run it on a login node.
# Options:
#   --arch 80[,90]   CUDA architectures to build for (default: the first GPU's, from nvidia-smi)
#   --jobs N         parallel build jobs (default: nproc)
# Rerunning is incremental: the venv, the clones and the builds are reused, and the clones are moved to the refs
# of pins.txt. Then:
#   bash scripts/gpu_bench.sh            unit tests, and the comparison against v1.9.0 (bench/results/<host>/)
#   source scripts/gpu-env.sh perf       (or base) to run your own scripts against one tree
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH=""
JOBS=$(nproc)
while [ $# -gt 0 ]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
done
for t in nvcc cmake git gcc python3 nvidia-smi; do
    command -v $t > /dev/null || { echo "$t not found (nvcc: load or install a CUDA 12 toolkit)" >&2; exit 1; }
done
if [ -z "$ARCH" ]; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')
fi
CUDA_ARCHS=$(echo "$ARCH" | tr ',' '\n' | sed 's/$/-real/' | paste -sd ';' -)
echo "== $(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -n 1)"
echo "   building for $CUDA_ARCHS with nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), $JOBS jobs"

echo "== venv: .venv-gpu (scripts/requirements-gpu.txt)"
VENV="$WS/.venv-gpu"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$WS/scripts/requirements-gpu.txt"
SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# GPU4PySCF loads cuSOLVER through an unversioned libcusolver.so, which NVIDIA's pip package does not ship
for f in "$SP"/nvidia/cusolver/lib/libcusolver.so.[0-9]*; do
    if [ -e "$f" ]; then
        ln -sf "$(basename "$f")" "$SP/nvidia/cusolver/lib/libcusolver.so"
    fi
done
# the libxc that the trees load at run time (BUILD_LIBXC=OFF below), from the gpu4pyscf wheel
WHEEL_DEPS="$SP/gpu4pyscf/lib/deps/lib"
[ -d "$WHEEL_DEPS" ] || { echo "no $WHEEL_DEPS: the gpu4pyscf-cuda12x wheel did not install" >&2; exit 1; }

for TREE in gpu4pyscf-base gpu4pyscf; do
    URL=""
    REF=""
    read -r URL REF < <(awk -v t=$TREE '$1 == t {print $2, $3}' "$WS/scripts/pins.txt") || true
    [ -n "$REF" ] || { echo "no $TREE line in scripts/pins.txt" >&2; exit 1; }
    T="$WS/$TREE"
    echo "== $TREE: $URL $REF"
    if [ ! -d "$T/.git" ]; then
        git clone -q "$URL" "$T"
    fi
    git -C "$T" fetch -q --tags origin
    git -C "$T" -c advice.detachedHead=false checkout -q "$REF"
    LIB="$T/gpu4pyscf/lib"
    B="$WS/build/$TREE-sm$(echo "$ARCH" | tr ',' '_')"
    mkdir -p "$WS/build"
    if [ ! -f "$B/CMakeCache.txt" ]; then
        cmake -S "$LIB" -B "$B" -DCUDA_ARCHITECTURES="$CUDA_ARCHS" -DBUILD_LIBXC=OFF \
            -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" > "$B.configure.log" 2>&1 \
            || { tail -n 30 "$B.configure.log"; exit 1; }
    fi
    t0=$SECONDS
    cmake --build "$B" -j "$JOBS" > "$B.build.log" 2>&1 || { tail -n 30 "$B.build.log"; exit 1; }
    mkdir -p "$LIB/deps"
    ln -sfn "$WHEEL_DEPS" "$LIB/deps/lib"
    echo "   $(find "$LIB" -maxdepth 1 -name '*.so' ! -type l | wc -l) libraries built ($((SECONDS - t0)) s;" \
         "logs in build/)"
done
echo "== ready. Next: bash scripts/gpu_bench.sh  (or: source scripts/gpu-env.sh perf|base)"
