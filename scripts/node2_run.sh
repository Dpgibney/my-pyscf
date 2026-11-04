#!/usr/bin/env bash
# node2 (Ryzen 5 4500: 6 cores, AVX2, 15 GB): build the original (pyscf-base 44313dfb4 + stock libcint) and the
# modified (pyscf perf 802f7e744 + qcint a95651f with PGO) PySCF trees as on the laptop, check them against
# each other (bench/kernels.py), then time both (bench/run_bench.py, ABBA, 2 reps, 6 threads, 9 GB each).
set -uo pipefail
cd ~/Chemistry/my-pyscf
export JOBS=12 XC_JOBS=12
L=scratch/node2_logs
mkdir -p $L
{
echo "== venv ($(python3 --version))"
python3 -m venv .venv || { echo "venv failed"; echo NODE2 DONE; exit 1; }
.venv/bin/pip install -q numpy==2.5.4 scipy==1.18.1 h5py==3.16.0 ninja pytest > $L/pip.log 2>&1 \
  || { echo "pinned versions unavailable, installing the latest"; .venv/bin/pip install -q numpy scipy h5py ninja pytest >> $L/pip.log 2>&1; }
.venv/bin/python -c "import numpy, scipy, h5py; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'h5py', h5py.__version__)" || { echo "python packages failed"; tail -5 $L/pip.log; echo NODE2 DONE; exit 1; }
export PATH=$PWD/.venv/bin:$PATH
step() {  # name, command...: run one build step (once: a finished step leaves $L/name.ok), stop if it fails
    local name=$1; shift
    local t0=$SECONDS
    [ -f $L/$name.ok ] && { echo "  $name already built"; return; }
    if "$@" > $L/$name.log 2>&1; then touch $L/$name.ok; echo "  $name ok ($((SECONDS - t0)) s)"; else
        echo "  $name FAILED ($((SECONDS - t0)) s):"; tail -n 15 $L/$name.log; echo NODE2 DONE; exit 1; fi
}
echo "== builds"
step openblas bash scripts/build_openblas.sh
step libcint-stock bash scripts/build_cint.sh libcint stock
step libxc-stock bash scripts/build_libxc.sh stock
step qbench gcc -O2 -o bench/qcint/qbench bench/qcint/qbench.c -ldl -lm
step qcint-pgo bash scripts/build_qcint_pgo.sh
step pyscf-base bash scripts/build_pyscf.sh pyscf-base libcint-stock libxc-stock
step pyscf bash scripts/build_pyscf.sh pyscf qcint-pgo libxc-stock -DNUMINT_X86_64_V3=ON
tail -n 1 $L/pyscf-base.log; tail -n 1 $L/pyscf.log
grep -m1 "avx512f works" $L/qcint-pgo.log
echo "== fixed-input kernel tests, modified against original"
(cd bench && ../.venv/bin/python kernels.py new=pyscf --threads 6) > $L/kernels.log 2>&1; rc=$?
grep -c "^  ok" $L/kernels.log | sed 's/^/  ok: /'; grep -v "^  ok" $L/kernels.log | tail -n 5
[ $rc = 0 ] || { echo "kernel tests failed (rc $rc)"; echo NODE2 DONE; exit 1; }
echo "== benchmark: original (base) against modified (perf)"
CASES=scf_caffeine_svp_pbe,scf_caffeine_svp_b3lyp,scf_caffeine_svp_r2scan,scf_caffeine+_svp_b3lyp_uks,scf_caffeine_tzvp_dfpbe,grad_benzene_svp_b3lyp,grad_caffeine_svp_b3lyp,tda_benzene_svp_b3lyp,tddft_benzene_svp_b3lyp,tda_caffeine_svp_b3lyp,hess_benzene_svp_b3lyp,scf_caffeine_tzvp_b3lyp,hess_caffeine_svp_b3lyp
(cd bench && ../.venv/bin/python run_bench.py --cases $CASES --reps 2 --threads 6 --tag node2-base-perf \
    base=pyscf-base:PYSCF_MAX_MEMORY=9000 perf=pyscf:PYSCF_MAX_MEMORY=9000) > $L/bench.log 2>&1
echo "  bench rc $?"; tail -n 3 $L/bench.log
echo NODE2 DONE
} > scratch/node2_run.log 2>&1
