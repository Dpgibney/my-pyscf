#!/usr/bin/env bash
# bench/diag/pcm_int1e_proto.py on n-C26H54/def2-TZVP (9998 surface points): PCM's surface integrals through the SGX
# driver (int1e_grids with Gaussian charges, s2, contracted on the fly) against the int3c2e path, and one Cholesky
# factorization against _get_vind's two solves; original (pyscf-base) and modified (pyscf) tree, 6 threads.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
for tree in pyscf-base pyscf; do
    echo "== $tree $(date +%H:%M:%S)"
    PYTHONPATH=$tree OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 timeout 1800 python -W ignore bench/diag/pcm_int1e_proto.py --n 26 --basis def2-tzvp
    echo "   exit $?"
done
echo NODE2 PCMPROTO DONE $(date +%H:%M)
