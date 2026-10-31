#!/bin/bash
# Install a CUDA 12.9.1 toolchain without root on node1, from NVIDIA's redistributable archives (verified against the
# sha256 sums in NVIDIA's manifest): nvcc with the runtime and headers (all that GPU4PySCF's build needs; CuPy brings
# the CUDA libraries at run time), cuobjdump and nvdisasm (inspect binaries), compute-sanitizer, CUPTI, and Nsight
# Compute in $PREFIX/nsight-compute. 12.9 matches the venv's CUDA 12.9 pip libraries and CuPy 13.4.1; the driver
# (595.99) supports up to CUDA 13.2. Run it on node1:  ssh node1 'bash ~/Chemistry/my-pyscf/scripts/node1_cuda_toolkit.sh'
set -euo pipefail
VERSION=12.9.1
PREFIX="${PREFIX:-$HOME/opt/cuda-$VERSION}"
BASE=https://developer.download.nvidia.com/compute/cuda/redist
COMPONENTS="cuda_nvcc cuda_cudart cuda_cccl cuda_cuobjdump cuda_nvdisasm cuda_sanitizer_api cuda_cupti cuda_profiler_api nsight_compute"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$BASE/redistrib_$VERSION.json" -o "$tmp/manifest.json"
mkdir -p "$PREFIX"
for c in $COMPONENTS; do
    read -r rel sha ver < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; a=d['linux-x86_64']; print(a['relative_path'], a['sha256'], d['version'])" "$tmp/manifest.json" "$c")
    curl -fsSL "$BASE/$rel" -o "$tmp/$c.tar.xz"
    echo "$sha  $tmp/$c.tar.xz" | sha256sum -c --quiet
    dest="$PREFIX"
    [ "$c" = nsight_compute ] && dest="$PREFIX/nsight-compute" && mkdir -p "$dest"
    tar xf "$tmp/$c.tar.xz" -C "$dest" --strip-components=1
    rm "$tmp/$c.tar.xz"
    echo "$c $ver" | tee -a "$PREFIX/VERSIONS.txt"
done
# the archives install libraries in lib/; nvcc's default search path is lib64/
[ -e "$PREFIX/lib64" ] || ln -s lib "$PREFIX/lib64"
"$PREFIX/bin/nvcc" --version | tail -2
du -sh "$PREFIX"
