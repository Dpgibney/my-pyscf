# PYSCF_CONFIG_FILE of every benchmark and validation worker (exec'd into pyscf.__config__).
# Knobs a configuration can set through its environment (CONFIG = LABEL=TREE:VAR=VAL,...):
#   BENCH_SEMI_DIRECT=0|1   SCF.semi_direct of the perf tree (unknown to other trees)
import os

if 'BENCH_SEMI_DIRECT' in os.environ:
    scf_hf_SCF_semi_direct = os.environ['BENCH_SEMI_DIRECT'] == '1'
