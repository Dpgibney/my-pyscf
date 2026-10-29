# PYSCF_CONFIG_FILE of every benchmark and validation worker (exec'd into pyscf.__config__).
# Knobs a configuration can set through its environment (CONFIG = LABEL=TREE:VAR=VAL,...):
#   BENCH_SEMI_DIRECT=0|1   SCF.semi_direct of the perf tree (unknown to other trees)
#   BENCH_FUSED_XC=0|1      numint.FUSED_XC of the perf tree (unknown to other trees)
import os

if 'BENCH_SEMI_DIRECT' in os.environ:
    scf_hf_SCF_semi_direct = os.environ['BENCH_SEMI_DIRECT'] == '1'
if 'BENCH_FUSED_XC' in os.environ:
    dft_numint_fused_xc = os.environ['BENCH_FUSED_XC'] == '1'
