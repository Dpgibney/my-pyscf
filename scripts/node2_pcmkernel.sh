#!/usr/bin/env bash
# The GPU-style PCM kernel (qcint branch pcm-kernel: int1e_grids_dm/charge_contract_cart; pyscf branch pcm-kernel:
# SGXnr_grids_contract_drv and pcm._int1e_grids_contract) on node2: a PGO build qcint-sx-contract of qcint perf plus the
# kernel (qcint/src back to perf after it); the driver and pcm.py into the pyscf tree (the kernel path is taken only
# with a library that has the kernels, so qcint-pgo keeps the SGX path), its C libraries rebuilt; then
# contract_vs_sgx.py on n-C26H54/def2-TZVP, both paths in one process, with LD_LIBRARY_PATH to qcint-sx-contract.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH JOBS=12
L=scratch/node2_logs
O=scratch/node2_pcmkernel.log
{
echo "== start $(date +%H:%M)"
cp qcint/CMakeLists.txt scratch/pcmkernel/CMakeLists.txt.perf
cp scratch/pcmkernel/cint1e_grids_contract.c qcint/src/
sed -i 's|  src/cint1e_grids.c src/g1e_grids.c$|  src/cint1e_grids.c src/g1e_grids.c src/cint1e_grids_contract.c|' qcint/CMakeLists.txt
grep -c cint1e_grids_contract qcint/CMakeLists.txt
bash scripts/build_qcint_pgo.sh sx-contract > $L/qcint_sx_contract.log 2>&1 && echo "  built qcint-sx-contract" || { echo "  build failed"; tail -n 20 $L/qcint_sx_contract.log; }
cp scratch/pcmkernel/CMakeLists.txt.perf qcint/CMakeLists.txt
rm -f qcint/src/cint1e_grids_contract.c
nm -D install/qcint-sx-contract/lib/libcint.so.6 | grep -c contract_cart
cp scratch/pcmkernel/nr_sgx_direct.c pyscf/pyscf/lib/vhf/nr_sgx_direct.c
cp scratch/pcmkernel/pcm.py pyscf/pyscf/solvent/pcm.py
cmake --build pyscf/pyscf/lib/build -j 6 > $L/pcmkernel_build.log 2>&1 && echo "  pyscf rebuilt" || { echo "  pyscf build failed"; tail -n 20 $L/pcmkernel_build.log; }
for nt in 6 1; do
    LD_LIBRARY_PATH=$PWD/install/qcint-sx-contract/lib PYTHONPATH=pyscf OMP_NUM_THREADS=$nt OPENBLAS_THREAD_TIMEOUT=20 \
        timeout 3000 python -W ignore scratch/pcmkernel/contract_vs_sgx.py (bench/diag/pcm_contract_vs_sgx.py) 26 def2-tzvp sph
done
echo "NODE2 PCMKERNEL DONE $(date +%H:%M)"
} > $O 2>&1
