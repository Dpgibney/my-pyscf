#!/usr/bin/env bash
# ORCA 6.1.1 on the laptop (10 MPI ranks, the system OpenMPI 5.0) for bench/orca/nad/nad_<job>.inp, each job in $JOBS
# (default: rijcosx exact) one after the other, beside bench/diag/rss_watch.py (PSS of the ORCA
# processes, killed at $LIMIT_GB). Outputs in scratch/orca/nad_<job>/ (not in git: ORCA's EULA 3c/3d); a summary
# per job on stdout.
set -u
WS=~/Chemistry/my-pyscf
O=~/Chemistry/Orca/orca_6_1_1/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
export PATH=$O:$PATH
export LD_LIBRARY_PATH=$O/lib
export OMP_NUM_THREADS=1
export OMPI_MCA_osc=^ucx    # the UCX one-sided component fails its query here (harmless warnings otherwise)
for job in ${JOBS:-rijcosx exact}; do
    d=$WS/scratch/orca/nad_$job
    rm -rf "$d" && mkdir -p "$d" && cp $WS/bench/orca/nad/nad_$job.inp $WS/bench/orca/nad/nad.xyz "$d"/
    cd "$d"
    echo "== $job $(date +%H:%M:%S)"
    t0=$(date +%s.%N)
    $O/orca nad_$job.inp > nad_$job.out 2> nad_$job.stderr &
    pid=$!
    python3 $WS/bench/diag/rss_watch.py --pid $pid --exe-prefix $O --limit-gb ${LIMIT_GB:-20} > rss.log 2>&1 &
    wpid=$!
    wait $pid
    rc=$?
    wait $wpid
    t1=$(date +%s.%N)
    echo "   exit $rc, wall $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s"
    grep WATCHDOG rss.log
    tail -n 1 rss.log
    grep -E "Number of basis functions  |Cavity surface points|Total number of grid points|SCF CONVERGED AFTER|FINAL SINGLE POINT|Total SCF time|Norm of the Cartesian|SCF iterations   |SCF Gradient evaluation|TOTAL RUN TIME" nad_$job.out
done
echo LAPTOP ORCA DONE $(date +%H:%M)
