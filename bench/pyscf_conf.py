# PYSCF_CONFIG_FILE of every benchmark and validation worker (exec'd into pyscf.__config__).
# Knobs a configuration can set through its environment (CONFIG = LABEL=TREE:VAR=VAL,...):
#   BENCH_SEMI_DIRECT=0|1   SCF.semi_direct of the perf tree (unknown to other trees)
#   BENCH_FUSED_XC=0|1      numint.FUSED_XC of the perf tree (unknown to other trees)
#   BENCH_INT2E_BATCH=0|1   _vhf.INT2E_BATCH of the perf tree: qcint's int2e_sph_batch in the direct J/K
import os

if 'BENCH_SEMI_DIRECT' in os.environ:
    scf_hf_SCF_semi_direct = os.environ['BENCH_SEMI_DIRECT'] == '1'
if 'BENCH_FUSED_XC' in os.environ:
    dft_numint_fused_xc = os.environ['BENCH_FUSED_XC'] == '1'
if 'BENCH_INT2E_BATCH' in os.environ:
    scf_vhf_int2e_batch = os.environ['BENCH_INT2E_BATCH'] == '1'
