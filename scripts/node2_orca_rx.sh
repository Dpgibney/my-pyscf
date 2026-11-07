#!/usr/bin/env bash
# ORCA 6.1.1 with 6 MPI ranks (OpenMPI 4.1.8 from scripts/node2_openmpi.sh): the inputs bench/orca/rx/<job>.inp for
# each job in $JOBS (default: caffeine and n-C20H42/def2-TZVP B3LYP, RIJCOSX and exact), one after
# the other. Each runs beside bench/diag/rss_watch.py, which sums the PSS of all ORCA and OpenMPI processes and kills
# them at 13.5 GB (node2 has 15 GB). Outputs in scratch/orca/rx_<job>/.
set -u
WS=~/Chemistry/my-pyscf
O=~/Chemistry/orca/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
M=~/Chemistry/openmpi-4.1.8
export PATH=$M/bin:$O:$PATH
export LD_LIBRARY_PATH=$M/lib:$O/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
for job in ${JOBS:-caffeine_rijcosx caffeine_exact c20h42_rijcosx c20h42_exact}; do
    d=$WS/scratch/orca/rx_$job
    rm -rf "$d" && mkdir -p "$d" && cp $WS/bench/orca/rx/$job.inp $WS/bench/orca/rx/${job%%_*}.xyz "$d"/
    cd "$d"
    echo "== $job $(date +%H:%M:%S)"
    t0=$(date +%s.%N)
    $O/orca $job.inp > $job.out 2> $job.stderr &
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
    grep -E "Total number of grid points|SCF CONVERGED AFTER|FINAL SINGLE POINT|Total SCF time|TOTAL RUN TIME|Split-RI-J  |Chain of spheres X|XC integration  |Recomputing exchange|Coulomb formation|Exchange formation|Fock matrix formation  " $job.out
done
echo NODE2 ORCA RX DONE $(date +%H:%M)
