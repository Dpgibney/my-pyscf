#!/bin/bash
# Set up the optimized CPU PySCF of this workspace on a Linux x86-64 machine, built as on the laptop and node2: a
# venv (.venv) with the pinned numpy/scipy/h5py, the trees of scripts/pins.txt, and OpenBLAS (DYNAMIC_ARCH), libxc,
# qcint (-march=native, profile-guided) and PySCF against them (its numint kernels with -march=x86-64-v3 when the
# CPU has AVX2 and FMA). Then a water B3LYP energy checks the build.
#     git clone https://github.com/Dpgibney/my-pyscf && cd my-pyscf && bash scripts/cpu_setup.sh
# Needs gcc, g++, gfortran, make, git, and python3 with venv (3.12 or newer gets the pinned numpy 2.5.4, scipy
# 1.18.1 and h5py 3.16.0, older ones the latest); cmake 3.22 or newer comes from pip if missing. 22 min on node2's
# 6 cores. qcint is built for this CPU: on a cluster, build on a node of the type that will run it.
# Options:
#   --jobs N      parallel build jobs (default: nproc)
#   --with-base   also build the unmodified PySCF (pyscf-base: master 44313dfb4 with the stock libcint) and compare
#                 the XC, J and K kernels of both trees on caffeine (bench/kernels.py)
# Rerunning is incremental: a step is skipped while its trees stay at the same commits. The clones are moved to the
# refs of pins.txt (one with uncommitted changes stops the script instead). Then:
#   source scripts/pyscf-env.sh && python my_script.py
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS=$(nproc)
BASE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --with-base) BASE=1; shift ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
done
[ "$(uname -sm)" = "Linux x86_64" ] || { echo "Linux x86-64 only" >&2; exit 1; }
for t in gcc g++ gfortran make git python3 objdump lscpu; do
    command -v $t > /dev/null || { echo "$t not found" >&2; exit 1; }
done
flags=" $(grep -m 1 '^flags' /proc/cpuinfo | cut -d: -f2) "
V3=-DNUMINT_X86_64_V3=ON
for f in avx avx2 bmi1 bmi2 f16c fma abm movbe xsave; do    # x86-64-v3
    [[ $flags == *" $f "* ]] || V3=""
done
CORES=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
[ "$CORES" -gt 0 ] || CORES=$(nproc)
echo "== $(lscpu | sed -n 's/^Model name: *//p' | head -n 1): $CORES cores, $(nproc) threads," \
     "$(awk '/^MemTotal:/ {printf "%.0f GB", $2 * 1.024e-6}' /proc/meminfo); x86-64-v3 kernels" \
     "$([ -n "$V3" ] && echo on || echo off); $JOBS build jobs"
LOG="$WS/build/cpu_setup"
mkdir -p "$LOG"

echo "== venv: .venv"
VENV="$WS/.venv"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
if "$VENV/bin/python" -m pip --version > /dev/null 2>&1; then
    pip_install() { "$VENV/bin/python" -m pip install -q "$@"; }
    pip_install --upgrade pip
else    # a venv made by uv has no pip
    pip_install() { uv pip install -q --python "$VENV/bin/python" "$@"; }
fi
PINNED="numpy==2.5.4 scipy==1.18.1 h5py==3.16.0"
if ! pip_install $PINNED ninja pytest > "$LOG/pip.log" 2>&1; then
    echo "   $PINNED are not available for $("$VENV/bin/python" --version): installing the latest"
    pip_install numpy scipy h5py ninja pytest
fi
export PATH="$VENV/bin:$PATH" JOBS XC_JOBS="$JOBS"
if ! command -v cmake > /dev/null || ! printf '3.22\n%s\n' "$(cmake --version | awk 'NR == 1 {print $3}')" | sort -V -C
then
    pip_install cmake
fi
"$VENV/bin/python" -c 'import sys, numpy, scipy, h5py; print("   python", sys.version.split()[0], "numpy",
    numpy.__version__, "scipy", scipy.__version__, "h5py", h5py.__version__)'

fetch() {  # TREE: put the clone $WS/TREE at TREE's ref in scripts/pins.txt (a new clone is shallow)
    local tree=$1 url="" ref="" t="$WS/$1" spec want
    local depth=(--depth 1)
    read -r url ref < <(awk -v t="$tree" '$1 == t {print $2, $3}' "$WS/scripts/pins.txt") || true
    [ -n "$ref" ] || { echo "no $tree line in scripts/pins.txt" >&2; exit 1; }
    if [ ! -d "$t/.git" ]; then
        git -c init.defaultBranch=main init -q "$t"
        git -C "$t" remote add origin "$url"
    fi
    if ! git -C "$t" rev-parse -q --verify "$ref^{commit}" > /dev/null; then
        spec="refs/tags/$ref:refs/tags/$ref"
        [[ $ref =~ ^[0-9a-f]{40}$ ]] && spec=$ref
        if git -C "$t" rev-parse -q --verify HEAD > /dev/null \
                && [ "$(git -C "$t" rev-parse --is-shallow-repository)" = false ]; then
            depth=()    # a full clone stays full
        fi
        git -C "$t" fetch -q ${depth[@]+"${depth[@]}"} origin "$spec" || git -C "$t" fetch -q --tags origin
    fi
    want=$(git -C "$t" rev-parse "$ref^{commit}")
    if [ "$(git -C "$t" rev-parse -q --verify HEAD || true)" != "$want" ]; then
        if [ -n "$(git -C "$t" status --porcelain --untracked-files=no)" ]; then
            echo "$t has uncommitted changes: commit or stash them before moving it to $ref" >&2
            exit 1
        fi
        git -C "$t" -c advice.detachedHead=false checkout -q "$want"
    fi
    echo "   $tree: $ref"
}
echo "== sources (scripts/pins.txt)"
TREES="pyscf qcint libxc OpenBLAS"
[ $BASE = 1 ] && TREES="$TREES pyscf-base libcint"
for tree in $TREES; do
    fetch "$tree"
done

step() {  # NAME KEY OUTPUT COMMAND...: run COMMAND (log: build/cpu_setup/NAME.log), unless OUTPUT exists and the
          # last run of NAME finished with the same KEY
    local name=$1 key=$2 out=$3 t0=$SECONDS
    shift 3
    if [ -e "$out" ] && [ "$(cat "$LOG/$name.ok" 2> /dev/null)" = "$key" ]; then
        echo "   $name: up to date"
        return
    fi
    rm -f "$LOG/$name.ok"
    if "$@" > "$LOG/$name.log" 2>&1; then
        echo "$key" > "$LOG/$name.ok"
        echo "   $name: built ($((SECONDS - t0)) s)"
    else
        echo "   $name FAILED after $((SECONDS - t0)) s; the end of build/cpu_setup/$name.log:" >&2
        tail -n 25 "$LOG/$name.log" >&2
        exit 1
    fi
}
rev() { git -C "$WS/$1" rev-parse HEAD; }
qcint_plain() {  # a first, plain qcint in install/qcint-pgo: PySCF builds against it, and the PGO training
                 # needs a PySCF tree for its molecule dumps
    rm -f "$LOG/qcint-pgo.ok" && rm -rf "$WS/build/qcint-pgo" && bash "$WS/scripts/build_cint.sh" qcint pgo
}
echo "== builds (logs in build/cpu_setup/)"
S="$WS/scripts"
step openblas "$(rev OpenBLAS)" "$WS/install/openblas/lib/libopenblas.so" bash "$S/build_openblas.sh"
step libxc "$(rev libxc)" "$WS/install/libxc-stock/lib/libxc.so" bash "$S/build_libxc.sh" stock
step qcint "$(rev qcint)" "$WS/install/qcint-pgo/lib/libcint.so.6" qcint_plain
step pyscf "$(rev pyscf) $(rev qcint) $(rev libxc) $(rev OpenBLAS) $V3" "$WS/pyscf/pyscf/lib/libcvhf.so" \
    bash "$S/build_pyscf.sh" pyscf qcint-pgo libxc-stock $V3
step qbench "$(sha1sum < "$WS/bench/qcint/qbench.c")" "$WS/bench/qcint/qbench" \
    gcc -O2 -o "$WS/bench/qcint/qbench" "$WS/bench/qcint/qbench.c" -ldl -lm
step qcint-pgo "$(rev qcint)" "$WS/install/qcint-pgo/lib/libcint.so.6" bash "$S/build_qcint_pgo.sh"
if [ $BASE = 1 ]; then
    step libcint "$(rev libcint)" "$WS/install/libcint-stock/lib/libcint.so.6" bash "$S/build_cint.sh" libcint stock
    step pyscf-base "$(rev pyscf-base) $(rev libcint) $(rev libxc) $(rev OpenBLAS)" \
        "$WS/pyscf-base/pyscf/lib/libcvhf.so" bash "$S/build_pyscf.sh" pyscf-base libcint-stock libxc-stock
fi

echo "== check: water/def2-SVP B3LYP with the new tree"
WS="$WS" bash -c 'source "$WS/scripts/pyscf-env.sh" && cd "$WS/build/cpu_setup" && python -' <<'EOF'
import os
import sys
import time

import pyscf
from pyscf import dft, gto

ws = os.path.realpath(os.environ['WS'])
t0 = time.time()
mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
e = mf.kernel()
ref = -76.35814183099166    # the same trees on the laptop, 7 cycles
print(f'   E = {e:.10f} Eh, {mf.cycles} cycles, {time.time() - t0:.1f} s ({e - ref:+.1e} from the laptop)')
print(f'   pyscf {pyscf.__version__}: {os.path.relpath(os.path.dirname(pyscf.__file__), ws)}')
loaded = sorted({line.split()[-1] for line in open('/proc/self/maps') if f'{ws}/install/' in line})
for p in loaded:
    print('  ', os.path.relpath(p, ws))
cint = [p for p in loaded if os.path.basename(p).startswith('libcint.')]
ok = (mf.converged and abs(e - ref) < 1e-8 and os.path.realpath(pyscf.__file__).startswith(f'{ws}/pyscf/')
      and len(cint) == 1 and cint[0].startswith(f'{ws}/install/qcint-pgo/'))
print('   ok' if ok else '   FAILED')
sys.exit(0 if ok else 1)
EOF
if [ $BASE = 1 ]; then
    echo "== check: the XC, J and K kernels of the new tree against the unmodified one (bench/kernels.py)"
    if (cd "$WS/bench" && "$VENV/bin/python" kernels.py new=pyscf --threads "$CORES") > "$LOG/kernels.log" 2>&1; then
        echo "   $(grep -c '^  ok' "$LOG/kernels.log") arrays agree (build/cpu_setup/kernels.log)"
    else
        grep -v '^  ok' "$LOG/kernels.log" | tail -n 15
        echo "   kernel check FAILED (build/cpu_setup/kernels.log)" >&2
        exit 1
    fi
fi
echo "== ready ($((SECONDS / 60)) min). To run your scripts with it:"
echo "   source $WS/scripts/pyscf-env.sh && python my_script.py"
echo "   ($(WS="$WS" bash -c 'source "$WS/scripts/pyscf-env.sh" && echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
    "PYSCF_MAX_MEMORY=$PYSCF_MAX_MEMORY"'); export your own values before sourcing to change them)"
