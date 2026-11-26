#!/bin/bash
# ts_example_variants.sh DIR GEOMETRIC_PATH: examples/wb97mv_ts.py in DIR/ts_<variant>/ as it is, with each of its
# switches, with all three, and with SMD water too, on HCN <-> HNC and (the three atoms and spin replaced) on H + H2 as
# a doublet. GEOMETRIC_PATH: a directory with the geometric package (not in the workspace's venv). Prints the steps of
# each search, the frequencies at its end and the largest component of the exact gradient there.
# VARIANTS="smd svp_smd" (SMD with exact J and K, and with the start Hessian in def2-SVP too) are the CPU side of
# bench/gpu/ts_example_variants.sh; the default list is the six of the first sentence. EXAMPLE: the script to run
# (default examples/wb97mv_ts.py). The free energy G is printed where the example has it; bench/diag/scf_work.py DIR/*
# counts the SCF work of each run from its log.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "$here/../../../.." && pwd)"
EXAMPLE=${EXAMPLE:-$W/examples/wb97mv_ts.py}
PY=$W/.venv/bin/python
mkdir -p "$1" && cd "$1" || exit 1
GM=$2
h3() { $PY -c "
import sys, re
s = sys.stdin.read()
s, n = re.subn(r'    C  0.000  0.000  0.000\n    N  1.190  0.000  0.000\n    H  0.486  1.042  0.000\n', '    H  0.000  0.000 -0.880\n    H  0.000  0.000  0.000\n    H  0.000  0.000  0.980\n', s)
assert n == 1
s, n = re.subn(r'    spin=0, ', '    spin=1, ', s)
assert n == 1
sys.stdout.write(s)"; }
on() { sed -e "s/^$1 = .*/$1 = $2/"; }
allon() { on RIJCOSX True | on SGX_RESPONSE True | on START_HESSIAN_BASIS "'def2-svp'"; }
smd() { sed -e "s/^    # mf = mf.SMD(); mf.with_solvent.solvent = 'water'/    mf = mf.SMD(); mf.with_solvent.solvent = 'water'/"; }
for mol in hcn h3; do
  for v in ${VARIANTS:-default sgx svp rijcosx allon allon_smd}; do
    d=ts_${mol}_$v; rm -rf $d; mkdir $d; cd $d
    { if [ $mol = h3 ]; then h3; else cat; fi; } < $EXAMPLE | case $v in
      default) cat ;; sgx) on SGX_RESPONSE True ;; svp) on START_HESSIAN_BASIS "'def2-svp'" ;;
      rijcosx) on RIJCOSX True ;; allon) allon ;; allon_smd) allon | smd ;;
      smd) smd ;; svp_smd) on START_HESSIAN_BASIS "'def2-svp'" | smd ;;
    esac > run.py
    t0=$(date +%s)
    PYTHONPATH=$GM:$W/pyscf $PY run.py > run.out 2>&1
    printf "%-16s %2d steps  %-44s %-20s %s [%d s]\n" "${mol}_$v" "$(grep -c '^Step' run.out)" \
        "$(grep 'harmonic freq' run.out | sed 's/harmonic frequencies (cm^-1): //')" \
        "$(grep 'largest gradient' run.out | sed 's/largest gradient component //')" \
        "$(grep 'free energy at' run.out | sed 's/.*: G = /G /; s/ Eh//')" $(( $(date +%s) - t0 ))
    cd ..
  done
done
