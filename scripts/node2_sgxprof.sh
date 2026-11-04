#!/usr/bin/env bash
# The SGX K build's profile on node2 (6 threads): pyscf-sgx = the modified tree with the sgx-exp branch's instrumented
# nr_sgx_direct.c (SGX_PROFILE defined in that copy only), built in its own directory; bench/diag/sgx_profile.py.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
{
echo "== pyscf-sgx: the modified tree with the instrumented SGX driver ($(date +%H:%M))"
if [ ! -d pyscf-sgx ]; then cp -a pyscf pyscf-sgx; fi
tar xf scratch/sgx_files.tar
sed -i '0,/#include <stdlib.h>/s//#define SGX_PROFILE\n#include <stdlib.h>/' pyscf-sgx/pyscf/lib/vhf/nr_sgx_direct.c
bash scripts/build_pyscf.sh pyscf-sgx qcint-pgo libxc-stock -DNUMINT_X86_64_V3=ON > scratch/node2_logs/sgx_build.log 2>&1 \
    && echo "  build ok: $(tail -n 1 scratch/node2_logs/sgx_build.log)" \
    || { echo "  build failed"; tail -n 20 scratch/node2_logs/sgx_build.log; echo NODE2 SGXPROF DONE; exit 1; }
for c in "caffeine def2-svp --pjs" "caffeine def2-svp" "caffeine def2-tzvp --pjs" "c20h42 def2-svp --pjs"; do
    set -- $c
    PYTHONPATH=pyscf-sgx OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 PYSCF_MAX_MEMORY=12000 python -W ignore \
        bench/diag/sgx_profile.py --mol $1 --basis $2 ${3:-} --reps 2 2>&1 | grep -v RuntimeWarning
done
echo NODE2 SGXPROF DONE
} > scratch/node2_sgxprof.log 2>&1
