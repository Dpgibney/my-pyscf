#!/usr/bin/env bash
# ORCA 6.1.1 with 6 MPI ranks (OpenMPI 4.1.8 from scripts/node2_openmpi.sh) on the B3LYP/C-PCM system of
# scripts/node2_pcm.sh: bench/orca/c26/c26_<job>.inp for each job in $JOBS (default: exact rijcosx default), one after
# the other. Each runs beside bench/diag/rss_watch.py, which sums the PSS of all ORCA and OpenMPI processes and kills
# them at 13.5 GB (node2 has 15 GB). Outputs in scratch/orca/c26_<job>/.
set -u
WS=~/Chemistry/my-pyscf
O=~/Chemistry/orca/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
M=~/Chemistry/openmpi-4.1.8
export PATH=$M/bin:$O:$PATH
export LD_LIBRARY_PATH=$M/lib:$O/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
for job in ${JOBS:-exact rijcosx default}; do
    d=$WS/scratch/orca/c26_$job
    rm -rf "$d" && mkdir -p "$d" && cp $WS/bench/orca/c26/c26_$job.inp $WS/bench/orca/c26/c26h54.xyz "$d"/
    cd "$d"
    echo "== $job $(date +%H:%M:%S)"
    t0=$(date +%s.%N)
    $O/orca c26_$job.inp > c26_$job.out 2> c26_$job.stderr &
    pid=$!
    python3 $WS/bench/diag/rss_watch.py --pid $pid --exe-prefix $O --exe-prefix $M --limit-gb 13.5 > rss.log 2>&1 &
    wpid=$!
    wait $pid
    rc=$?
    wait $wpid
    t1=$(date +%s.%N)
    echo "   exit $rc, wall $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s"
    grep WATCHDOG rss.log
    tail -n 1 rss.log
    grep -E "Cavity surface points|Total number of grid points|SCF CONVERGED AFTER|FINAL SINGLE POINT|Total SCF time|TOTAL RUN TIME" c26_$job.out
done
echo NODE2 ORCA DONE $(date +%H:%M)
