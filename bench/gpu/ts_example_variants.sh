#!/bin/bash
# ts_example_variants.sh DIR: examples/wb97mv_ts_gpu.py in DIR/<molecule>_<variant>/ as it is, with the start Hessian
# in def2-SVP, with SMD water, and with both, on HCN <-> HNC and (the three atoms and spin replaced) on H + H2 as a
# doublet (UKS). Prints the steps of each search, the frequencies at its end, the largest gradient component there
# and the wall time. Run it in the GPU environment (source scripts/gpu-env.sh perf); VARIANTS selects variants
# (cheapk, not in the default list: the example's optional line, the seminumerical K in the cheap model of the
# two-level CPHF, switched on).
# The CPU side: bench/diag/wb97mv/uks_ts_audit/ts_example_variants.sh (default, svp; VARIANTS="smd svp_smd").
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "$here/../.." && pwd)"
mkdir -p "$1" && cd "$1" || exit 1
h3() { python -c "
import sys, re
s = sys.stdin.read()
s, n = re.subn(r'    C  0.000  0.000  0.000\n    N  1.190  0.000  0.000\n    H  0.486  1.042  0.000\n', '    H  0.000  0.000 -0.880\n    H  0.000  0.000  0.000\n    H  0.000  0.000  0.980\n', s)
assert n == 1
s, n = re.subn(r'    spin=0, ', '    spin=1, ', s)
assert n == 1
sys.stdout.write(s)"; }
on() { sed -e "s/^$1 = .*/$1 = $2/"; }
smd() { sed -e "s/^    # mf = mf.SMD(); mf.with_solvent.solvent = 'water'/    mf = mf.SMD(); mf.with_solvent.solvent = 'water'/"; }
for mol in hcn h3; do
  for v in ${VARIANTS:-default svp smd svp_smd}; do
    d=${mol}_$v; rm -rf $d; mkdir $d; cd $d
    { if [ $mol = h3 ]; then h3; else cat; fi; } < $W/examples/wb97mv_ts_gpu.py | case $v in
      default) cat ;; svp) on START_HESSIAN_BASIS "'def2-svp'" ;; smd) smd ;;
      svp_smd) on START_HESSIAN_BASIS "'def2-svp'" | smd ;;
      cheapk) sed -e "s/^    # hobj.cphf_two_level_parts = /    hobj.cphf_two_level_parts = /" ;;
    esac > run.py
    [ $v != cheapk ] || grep -q "^    hobj.cphf_two_level_parts = ('nlc', 'k')" run.py || echo "cheapk: line not switched on"
    t0=$(date +%s)
    python -u run.py > run.out 2>&1
    printf "%-12s rc=%d %2d steps  %-34s %-10s %s  [%d s]\n" "$d" $? "$(grep -c '^Step' run.out)" \
        "$(grep 'harmonic freq' run.out | sed 's/harmonic frequencies (cm^-1): //')" \
        "$(grep 'largest gradient' run.out | sed 's/largest gradient component //; s/ Eh.Bohr//')" \
        "$(grep '^[0-9]* imaginary mode' run.out)" $(( $(date +%s) - t0 ))
    grep -q 'harmonic freq' run.out || tail -n 12 run.out | cut -c1-200 | sed 's/^/    /'
    cd ..
  done
done
