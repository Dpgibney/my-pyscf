# PYSCF_CONFIG_FILE of every benchmark and validation worker (exec'd into pyscf.__config__).
# Knobs a configuration can set through its environment (CONFIG = LABEL=TREE:VAR=VAL,...):
#   BENCH_SEMI_DIRECT=0|1   SCF.semi_direct of the perf tree (unknown to other trees)
#   BENCH_FUSED_XC=0|1      numint.FUSED_XC of the perf tree (unknown to other trees)
#   BENCH_INT2E_BATCH=0|1   _vhf.INT2E_BATCH of the perf tree: qcint's int2e_sph_batch in the direct J/K
#   BENCH_JK_GEMM=0|1       _vhf.JK_GEMM of the perf tree: J/K of many dms (CPHF, TDDFT) by dense products
#   BENCH_HESS_JK_ENERGY=0|1  hessian.rhf.JK_ENERGY of the perf tree: the Hessian's two-electron blocks as
#                           energies contracted in the integral loop (hess_jk_energy)
#   BENCH_HESS_H1_SCREEN=0|1  hessian.rhf.MAKE_H1_TOL of the perf tree: 1e-14, or 0 (make_h1's int2e_ip1 J/K
#                           unscreened, as upstream)
#   BENCH_HESS_JK_DOT=0|1   hessian.rhf.JK_ENERGY_DOT of the perf tree: hess_jk_energy's integrals contracted in
#                           qcint's _dot kernels, or in PySCF
import os

if 'BENCH_SEMI_DIRECT' in os.environ:
    scf_hf_SCF_semi_direct = os.environ['BENCH_SEMI_DIRECT'] == '1'
if 'BENCH_FUSED_XC' in os.environ:
    dft_numint_fused_xc = os.environ['BENCH_FUSED_XC'] == '1'
if 'BENCH_INT2E_BATCH' in os.environ:
    scf_vhf_int2e_batch = os.environ['BENCH_INT2E_BATCH'] == '1'
if 'BENCH_JK_GEMM' in os.environ:
    scf_vhf_jk_gemm = os.environ['BENCH_JK_GEMM'] == '1'
if 'BENCH_HESS_JK_ENERGY' in os.environ:
    hessian_rhf_jk_energy = os.environ['BENCH_HESS_JK_ENERGY'] == '1'
if 'BENCH_HESS_H1_SCREEN' in os.environ:
    hessian_rhf_make_h1_tol = 1e-14 if os.environ['BENCH_HESS_H1_SCREEN'] == '1' else 0
if 'BENCH_HESS_JK_DOT' in os.environ:
    hessian_rhf_jk_energy_dot = os.environ['BENCH_HESS_JK_DOT'] == '1'
