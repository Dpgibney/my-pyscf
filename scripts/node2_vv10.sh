#!/usr/bin/env bash
# It24 on node2: the modified tree's new VV10 kernels (one division per pair, vector lanes, each pair once on the
# SCF's grid). The previous build stays as pyscf-it23 for a control run.
cd ~/Chemistry/my-pyscf
export PATH=$PWD/.venv/bin:$PATH
L=scratch/node2_logs
while pgrep -f "[n]ode2_wb97mv_grad.sh" > /dev/null; do sleep 20; done
{
echo "== keep the previous modified tree as pyscf-it23, install the new VV10 files, rebuild ($(date +%H:%M))"
rm -rf pyscf-it23 && cp -a pyscf pyscf-it23
tar xf scratch/vv10_files.tar
cmake --build pyscf/pyscf/lib/build -j 12 > $L/vv10_build.log 2>&1 && echo "  build ok" \
    || { echo "  build failed"; tail -n 20 $L/vv10_build.log; echo NODE2 VV10 DONE; exit 1; }
echo "== the kernels on fixed inputs (vv10_check.py: caffeine/def2-SVP, minao density, level-1 NLC grid, 6 threads)"
PYTHONPATH=pyscf OMP_NUM_THREADS=6 python -W ignore bench/diag/vv10_check.py --make scratch/vv10_in.npz --level 1
for t in pyscf-base pyscf-it23 pyscf; do
    PYTHONPATH=$t OMP_NUM_THREADS=6 OPENBLAS_THREAD_TIMEOUT=20 python -W ignore bench/diag/vv10_check.py \
        scratch/vv10_in.npz scratch/vv10_$t.npz --reps 2
done
python -W ignore bench/diag/vv10_check.py --compare scratch/vv10_pyscf-base.npz scratch/vv10_pyscf.npz
echo "== fixed-input kernel tests against pyscf-base (VERSION 7, with the VV10 entries)"
(cd bench && python -W ignore kernels.py new=pyscf --threads 6) > $L/kernels_vv10.log 2>&1; rc=$?
grep -c "^  ok" $L/kernels_vv10.log | sed 's/^/  ok: /'; grep -v "^  ok\|RuntimeWarning" $L/kernels_vv10.log | tail -n 4
[ $rc = 0 ] || { echo "kernel tests failed (rc $rc)"; echo NODE2 VV10 DONE; exit 1; }
echo "== wB97M-V: the new modified tree at the default and SG-1 NLC grids; the previous one (pyscf-it23) at SG-1"
(cd bench && python -W ignore run_bench.py --cases scf_caffeine_svp_wb97mv,grad_caffeine_svp_wb97mv --reps 1 \
    --threads 6 --tag node2-vv10 new=pyscf:PYSCF_MAX_MEMORY=9000 new_sg1=pyscf:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1 \
    old_sg1=pyscf-it23:PYSCF_MAX_MEMORY=9000,BENCH_NLC_GRID=sg1) 2>&1 | grep -v "RuntimeWarning"
echo NODE2 VV10 DONE
} > scratch/node2_vv10.log 2>&1
