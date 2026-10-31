#!/bin/bash
# Create the GPU4PySCF venv on node1 (the desktop with the GTX 1650 Super). Run it there, from the copy of the
# workspace at the same path:  ssh node1 'bash ~/Chemistry/my-pyscf/scripts/node1_gpu_venv.sh'
# Rerunning reuses the venv and brings it to the pins in requirements-gpu.txt.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$WS/.venv-gpu"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-cache-dir -r "$WS/scripts/requirements-gpu.txt"
# GPU4PySCF loads cuSOLVER with ctypes.util.find_library('cusolver'), falling back to
# numpy.ctypeslib.load_library('libcusolver', <site-packages>/nvidia/cusolver/lib). Both need an unversioned
# libcusolver.so, which NVIDIA's pip package does not ship (only libcusolver.so.11).
sp="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for f in "$sp"/nvidia/cusolver/lib/libcusolver.so.[0-9]*; do
    ln -sf "$(basename "$f")" "$sp/nvidia/cusolver/lib/libcusolver.so"
done
mkdir -p "$WS/bench/results/node1"
"$VENV/bin/python" -m pip freeze > "$WS/bench/results/node1/pip-freeze.txt"
echo "venv ready: $VENV"
