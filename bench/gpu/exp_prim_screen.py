"""Primitive screening in GPU4PySCF's Rys K and J+K kernels (gpu4pyscf/lib/gvhf-rys), generated from the unpatched
sources at BASE (a git revision of TREE, default perf) and written into TREE.
  unrolled_rys_k.cu, unrolled_rys_jk.cu, the 19 + 19 kernels with one thread per quartet (rys_roots_for_k with
  stride 1, so no thread reads another thread's roots):
    --barrier remove   drop the __syncthreads() at the top of the bra-primitive loop (needed before any ket skip,
                       since threads skip different ket primitives)
    --ket zero|cutoff  skip a ket primitive pair when ckcl == 0 (padding threads, underflowed exp) / |ckcl| < e^-60
    --bra zero|cutoff  the same for the bra pair's cicj
  the 6 + 6 multi-thread unrolled kernels and the generic rys_k_kernel / rys_jk_kernel (rys_contract_k.cu,
  rys_contract_jk.cu): their threads exchange data through shared memory, so the barriers stay. Only the --bra skip
  applies, placed before the barrier: all threads of a block test the same bra pair and skip together.
--launch sets the launch shape of the one-thread kernels (K, J+K and the derivative families): blocks of NSQ threads,
and as many per SM as the occupancy calculator allows, up to BLOCKS and the POOL_BLOCKS_PER_SM task queues the pool
holds. env reads NSQ and BLOCKS for K and J+K at run time (GPU4PYSCF_RYS_NSQ, _BLOCKS), for sweeps; it leaves the
derivative kernels alone, whose block size is a compile-time constant.
--gen-k writes generated_rys_k.cu (gen_rys_k.py): multi-thread K kernels for classes that would otherwise run the
generic kernel, built into libgvhf_rys and tried by RYS_build_k after the unrolled kernels. The default is the 18
classes of part 7; off leaves them out.
PRIM_CUTOFF (vhf.cuh) is e^-60 = 8.8e-27, libcint's expcutoff. A skipped primitive pair enters its integrals with a
factor below that, many orders of magnitude under the SCF's 1e-13 shell-quartet cutoff.
--j-tiles auto makes the bra tile of the J engine's unrolled kernels (unrolled_md_j.cu, unrolled_md_j_4dm.cu) a
run-time value, chosen so that a launch fills the SMs (part 22).
--gen-j writes generated_md_j.cu (gen_md_j.py): MD J kernels for classes that would otherwise run the generic
md_j_1dm_kernel, built into libgvhf_md and tried by MD_build_j after the unrolled kernels (part 23).
--j-grad adds MD_build_j_ip1 (the J engine with both pairs' Hermite orders one higher) and md_j_grad.cu (the
pairs' derivative Hermite densities against those potentials) for the J engine's nuclear gradient (part 24);
--gen-j-ip1 the gradient's own kernels for those classes (generated_md_j_ip1.cu), which MD_build_j_ip1 tries first.
--gen-ip2 writes generated_ejk_ip2.cu (gen_rys_ejk_ip2.py): the Hessian's second-derivative kernels with sums per
Cartesian tuple, tried by RYS_per_atom_jk_ip2_type12 and _type3 before their unrolled and generic kernels.
Usage: exp_prim_screen.py TREE [--base perf] [--barrier keep|remove] [--ket none|zero|cutoff]
                               [--bra none|zero|cutoff] [--multi on|off] [--launch 128x4|off|NSQxBLOCKS|env]
                               [--generic on|off] [--gen-k CLASS:NSQxGS,...|off] [--gen-first]
                               [--gen-ip2 KIND:CLASS:NSQxGS,...|off] [--gen-ip2-files N] [--gen-ip2-env]
                               [--j-tiles auto|off] [--j-blocks-per-sm 64] [--gen-j 'LIJ,LKL[:TILEY] ...'|off]
                               [--j-grad on|off] [--gen-j-ip1 'LIJ,LKL ...'|off]"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('tree')
ap.add_argument('--base', default='perf')
ap.add_argument('--barrier', default='remove', choices=['keep', 'remove'])
ap.add_argument('--ket', default='cutoff', choices=['none', 'zero', 'cutoff'])
ap.add_argument('--bra', default='cutoff', choices=['none', 'zero', 'cutoff'])
ap.add_argument('--multi', default='on', choices=['on', 'off'],
                help='also the bra skip in the multi-thread and generic kernels')
# Every class that ran the generic K kernel and fits at up to 75 outputs per thread without heavy spills (part 7)
GEN_K = ('3111:32x8,3210:64x4,3120:64x4,3021:64x4,2121:32x8,3220:32x8,3310:32x8,3030:64x4,3130:32x8,3031:32x8,'
         '3121:32x8,3211:32x8,2211:32x8,3022:32x8,2220:32x8,3300:64x4,2022:32x8,3032:32x8')
ap.add_argument('--gen-k', default=GEN_K,
                help='comma-separated CLASS:NSQxGS specs of generated multi-thread K kernels (gen_rys_k.py), or off')
# The generated gradient kernels of part 8, by measured class time on caffeine/def2-TZVP: one thread per quartet
# (NSQx1) for the small classes, 64x4 or 32x8 where 32 quartets fit in 64 KB of shared memory, 16x16 and 8x32 (two or
# four gout_ids per warp, one instruction stream per warp) for the larger ones; (ff|ff) does not fit and stays generic
GEN_EJK = ('0000:128x1,1000:128x1,1010:128x1,1011:128x1,1100:128x1,1110:128x1,2000:128x1,2010:128x1,2020:128x1,'
           '2100:128x1,2200:128x1,3000:128x1,3010:128x1,3100:128x1,'
           '2011:64x4,2021:64x4,2110:64x4,3011:64x4,3020:64x4,3110:64x4,'
           '1111:32x8,2111:32x8,2120:32x8,2210:32x8,2022:32x8,2121:32x8,2211:32x8,2220:32x8,3021:32x8,3022:32x8,'
           '3030:32x8,3031:32x8,3111:32x8,3120:32x8,3130:32x8,3200:32x8,3210:32x8,3220:32x8,3300:32x8,3310:32x8,'
           '2122:16x16,2221:16x16,2222:16x16,3032:16x16,3033:16x16,3121:16x16,3122:16x16,3131:16x16,3132:16x16,'
           '3211:16x16,3221:16x16,3230:16x16,3231:16x16,3311:16x16,3320:16x16,3330:16x16,'
           '3133:8x32,3222:8x32,3232:8x32,3233:8x32,3321:8x32,3322:8x32,3331:8x32,3332:8x32')
ap.add_argument('--gen-ejk', default=GEN_EJK,
                help='comma-separated CLASS:NSQxGS specs of generated gradient kernels (gen_rys_ejk.py; NSQx1 is one '
                     'thread per quartet), or off')
ap.add_argument('--gen-ejk-files', type=int, default=4,
                help='translation units for the generated gradient kernels, so that the library compiles in parallel')
# The generated second-derivative kernels of part 16 (gen_rys_ejk_ip2.py), KIND:CLASS:NSQxGS with KIND 12 or 3: every
# class with d functions whose g arrays and factor table fit 8 or more quartets in 64 KB of shared memory, at the
# most quartets per block that fit (256 threads), and (pp|pp), (pp|ps), (ps|pp), where they beat GPU4PySCF's unrolled
# kernels; the smaller s and p classes keep its one-thread kernels, which are faster there. (dd|dd), and type12
# (dd|dp) and (dp|dd), do not fit and stay generic.
GEN_IP2 = ('12:2110:16x16,12:2111:8x32,3:2110:32x8,3:2111:16x16,12:2011:16x16,12:2010:32x8,3:2011:32x8,'
           '12:2121:8x32,3:2121:8x32,3:2010:64x4,12:2100:32x8,12:2211:8x32,12:2210:8x32,12:2120:8x32,3:2211:8x32,'
           '3:2100:32x8,3:2120:16x16,12:2000:32x8,3:2210:16x16,12:2021:8x32,3:2021:16x16,3:2000:64x4,12:2020:16x16,'
           '3:2020:32x8,3:2221:8x32,3:2122:8x32,12:2200:16x16,12:2220:8x32,3:2220:16x16,3:2200:32x8,3:2022:16x16,'
           '12:2022:8x32,12:1111:16x16,3:1111:16x16,12:1110:16x16,3:1110:32x8,12:1011:16x16,3:1011:32x8')
ap.add_argument('--gen-ip2', default=GEN_IP2,
                help='comma-separated KIND:CLASS:NSQxGS specs of generated Hessian second-derivative kernels '
                     '(gen_rys_ejk_ip2.py; KIND 12 or 3), or off')
ap.add_argument('--gen-ip2-files', type=int, default=4,
                help='translation units for the generated second-derivative kernels')
ap.add_argument('--gen-ip2-env', action='store_true',
                help='experiment: GPU4PYSCF_RYS_GEN_IP2_OFF=1 skips the generated second-derivative kernels at run time')
ap.add_argument('--gen-first', action='store_true',
                help='experiment: GPU4PYSCF_RYS_GEN_FIRST=1 tries the generated kernels before the unrolled ones, '
                     'GPU4PYSCF_RYS_GEN_OFF=1 skips them')
ap.add_argument('--generic', default='on', choices=['on', 'off'],
                help='generic kernels: aij and aj/aij from registers in the root loop, and no TRR on padding quartets')
ap.add_argument('--j-tiles', default='auto', choices=['auto', 'off'],
                help="J engine (gvhf-md): auto picks the unrolled kernels' bra tile at run time to fill the SMs")
ap.add_argument('--j-blocks-per-sm', type=int, default=64,
                help='--j-tiles auto: the least blocks per SM a J launch gets (twice that for diagonal classes)')
# The classes of f functions (lij, lkl <= 6) without an unrolled MD J kernel: lij + lkl >= 6, lij >= lkl, and (3,4)
# ((fs|dd), whose bra pair comes later in the J engine's loop although its lij is smaller). On a GTX 1650 Super each
# runs 1.1-2.4x faster than the generic md_j_1dm_kernel, even those that spill registers; (6,5) and (6,6) spill
# 38 and 104 KB per thread and run slower, so they stay generic (part 23).
GEN_J = '3,3 4,2 5,1 6,0 4,3 5,2 6,1 3,4 4,4 5,3 6,2 5,4 6,3 5,5 6,4'
ap.add_argument('--gen-j', default=GEN_J,
                help='space-separated LIJ,LKL[:TILEY] specs of generated MD J kernels (gen_md_j.py), or off')
ap.add_argument('--j-prefactor', default='rsqrt', choices=['rsqrt', 'old'],
                help="J engine kernels: each quartet's prefactor and theta from 1/aij and 1/akl kept with the pairs "
                     "and one rsqrt, where they took two divisions and a sqrt (part 27)")
ap.add_argument('--vv10', default='sym', choices=['sym', 'old'],
                help="VV10's Fock kernel and the gradient's grid-response pair term (gdft/vv10.cu): sym evaluates each "
                     "pair of grid points once (parts 26 and 28)")
ap.add_argument('--boys', default='new', choices=['new', 'old'],
                help="the MD J engine's Boys function: new multiplies by tabulated 1/(n+1/2) in the series and the "
                     "downward recursion, and skips erf where it is 1 (part 25)")
ap.add_argument('--j-grad', default='on', choices=['on', 'off'],
                help="J engine: MD_build_j_ip1 and md_j_grad.cu for the J engine's nuclear gradient")
# The gradient's own kernels (gen_md_j.py gLIJ,LKL): both potentials one order above the densities with R to that order
# only, where the raised J kernels run R one order higher and half their contraction on zeros (part 24)
GEN_J_IP1 = ('0,0 1,0 1,1 2,0 2,1 2,2 3,0 3,1 3,2 3,3 4,0 4,1 4,2 4,3 4,4 5,0 5,1 5,2 5,3 5,4 5,5 3,4 '
             '6,0 6,1 6,2 6,3')
ap.add_argument('--gen-j-ip1', default=GEN_J_IP1,
                help='--j-grad on: space-separated LIJ,LKL[:TILEY] classes with their own gradient kernels, or off')
ap.add_argument('--launch', default='128x4',
                help="launch shape of the one-thread kernels: off (one block of 256 per SM, as in v1.9.0), NSQxBLOCKS "
                     "(blocks of NSQ threads, at most BLOCKS per SM), or env (K and J+K from GPU4PYSCF_RYS_NSQ and "
                     "GPU4PYSCF_RYS_BLOCKS at run time)")
args = ap.parse_args()
assert args.barrier == 'remove' or args.ket == 'none', 'a ket skip needs the barrier removed (threads diverge)'

D = 'gpu4pyscf/lib/gvhf-rys'
test = {'zero': '{} == 0', 'cutoff': 'fabs({}) < PRIM_CUTOFF'}
old_k = '            double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;\n'
old_loop = '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {\n                __syncthreads();\n'
new_loop = '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {\n'
old_c = '                double cicj = cicj_cache[ijp];\n'


def source(rel):
    return subprocess.run(['git', '-C', args.tree, 'show', f'{args.base}:{rel}'], check=True, capture_output=True,
                          text=True).stdout


def bra_skip_before_barrier(part):
    assert part.count(old_loop) == 1
    if args.bra == 'none' or args.multi == 'off':
        return part
    return part.replace(old_loop, new_loop + f"                if ({test[args.bra].format('cicj_cache[ijp]')}) continue;\n"
                                             "                __syncthreads();\n")


counts = {}
for name in ('unrolled_rys_k.cu', 'unrolled_rys_jk.cu'):
    s = source(f'{D}/{name}')
    parts = re.split(r'(?=__global__ static\nvoid rys_j?k_\d{4}\(JKMATRIX_KERNEL_ARGS\))', s)
    n1 = n4 = 0
    for k, part in enumerate(parts):
        if not re.match(r'__global__ static\nvoid rys_j?k_\d{4}\(', part):
            continue
        if 'nsq_per_block, 1, 0);' in part:
            assert part.count(old_k) == part.count(old_loop) == part.count(old_c) == 1
            if args.ket != 'none':
                part = part.replace(old_k, old_k + f"            if ({test[args.ket].format('ckcl')}) continue;\n")
            if args.barrier == 'remove':
                part = part.replace(old_loop, new_loop)
            if args.bra != 'none':
                part = part.replace(old_c, old_c + f"                if ({test[args.bra].format('cicj')}) continue;\n")
            n1 += 1
        else:
            assert 'nsq_per_block, 4, gout_id);' in part
            part = bra_skip_before_barrier(part)
            n4 += 1
        parts[k] = part
    assert (n1, n4) == (19, 6), (name, n1, n4)
    open(f'{args.tree}/{D}/{name}', 'w').write(''.join(parts))
    counts[name] = (n1, n4)

for name in ('rys_contract_k.cu', 'rys_contract_jk.cu'):
    s = bra_skip_before_barrier(source(f'{D}/{name}'))
    open(f'{args.tree}/{D}/{name}', 'w').write(s)

# Derivative kernels: the energy derivatives of gradients and TDDFT gradients (unrolled_ejk_ip1.cu) and Hessians
# (unrolled_ejk_ip2_type12.cu, unrolled_ejk_ip2_type3.cu), and the J/K derivative matrices (unrolled_rys_jk_ip1.cu).
# Their one-thread kernels have no barrier at the top of the bra loop, but padding threads run the full primitive
# loops (they skip only the accumulation) and nothing skips primitive pairs.
#   energy kernels: padding threads leave the task loop at once (no barrier follows inside it), and zero or
#     negligible ket and bra primitive pairs are skipped (ckcl and cicj scale every term of the integral).
#   J/K derivative matrices: thread 0 publishes aij, aj/aij and 2 ai through shared memory (aij_cache) behind a
#     barrier per primitive pair, though every thread has just computed the same values. Each thread uses its own,
#     the barrier goes, and then the skips apply (ckcl carries fac_sym, which is 0 on padding threads).
# Multi-thread kernels get the block-uniform bra skip; generic kernels with a barrier at the top of the bra loop too.
task_top = '    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {\n'
ejk_k = '            double ckcl = env[ck+kp] * env[cl+lp] * Kcd;\n'
ijp_top = '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {\n'
aij_write = ('                __syncthreads();\n'
             '                if (sq_id == 0) {\n'
             '                    aij_cache[0] = aij;\n'
             '                    aij_cache[1] = aj_aij;\n'
             '                    aij_cache[2] = ai * 2;\n'
             '                }\n')


def bra_skip_uniform(part):
    '''bra skip at the top of a bra loop without a barrier there, for kernels whose threads share a quartet'''
    assert part.count(ijp_top) >= 1 and old_loop not in part
    if args.bra == 'none' or args.multi == 'off':
        return part
    return part.replace(ijp_top, ijp_top + f"                if ({test[args.bra].format('cicj_cache[ijp]')}) continue;\n")


deriv = {}
for name in ('unrolled_ejk_ip1.cu', 'unrolled_ejk_ip2_type12.cu', 'unrolled_ejk_ip2_type3.cu', 'unrolled_rys_jk_ip1.cu'):
    s = source(f'{D}/{name}')
    parts = re.split(r'(?=__global__ static\nvoid rys_[ev]jk_ip[12]\w*\((?:JKMATRIX|JKENERGY)_KERNEL_ARGS\))', s)
    n1 = n8 = 0
    for k, part in enumerate(parts):
        if not re.match(r'__global__ static\nvoid rys_[ev]jk_ip', part):
            continue
        if 'nsq_per_block, 1, 0);' in part:
            assert part.count(task_top) == 1 and part.count(old_c) == part.count(ijp_top) >= 1
            if name == 'unrolled_rys_jk_ip1.cu':
                assert part.count(old_k) == part.count(aij_write) == part.count(ijp_top)
                if args.barrier == 'remove':
                    part = part.replace(aij_write, '')
                    part = re.sub(r'^ *double aij = aij_cache\[0\];\n', '', part, flags=re.M)
                    part = part.replace('double ai2 = aij_cache[2];', 'double ai2 = ai * 2;')
                    part = part.replace('aij_cache[1]', 'aj_aij')
                    assert 'aij_cache[' not in part
                if args.ket != 'none':
                    assert args.barrier == 'remove'
                    part = part.replace(old_k, old_k + f"            if ({test[args.ket].format('ckcl')}) continue;\n")
            else:
                assert part.count(ejk_k) == 1 and '__syncthreads' not in part[part.index(task_top):]
                part = part.replace(task_top, task_top + '        if (task_id >= ntasks) break;\n')
                if args.ket != 'none':
                    part = part.replace(ejk_k, ejk_k + f"            if ({test[args.ket].format('ckcl')}) continue;\n")
            if args.bra != 'none':
                part = part.replace(old_c, old_c + f"                if ({test[args.bra].format('cicj')}) continue;\n")
            if args.barrier == 'remove':
                assert '__syncthreads' not in part[part.index(task_top):]
            n1 += 1
        else:
            part = bra_skip_uniform(part)
            n8 += 1
        parts[k] = part
    open(f'{args.tree}/{D}/{name}', 'w').write(''.join(parts))
    deriv[name] = (n1, n8)
assert deriv == {'unrolled_ejk_ip1.cu': (11, 7), 'unrolled_ejk_ip2_type12.cu': (4, 3), 'unrolled_ejk_ip2_type3.cu': (4, 3),
                 'unrolled_rys_jk_ip1.cu': (25, 8)}, deriv

n_generic = 0
for name in ('rys_contract_jk_ip1.cu', 'rys_contract_jk_ip2.cu'):
    parts = re.split(r'(?=__global__ static\nvoid )', source(f'{D}/{name}'))
    for k, part in enumerate(parts):
        if old_loop in part:
            parts[k] = bra_skip_before_barrier(part)
            n_generic += 1
    open(f'{args.tree}/{D}/{name}', 'w').write(''.join(parts))
assert n_generic == 5

# Generic kernels (rys_k_kernel, and the BUILD_4C_GXYZ macro behind rys_jk_kernel, the four generic gradient kernels,
# the two generic Hessian kernels and int4c2e). Inside the root loop they re-read aij and aj/aij from shared memory
# (aij_cache) after a barrier, although every thread holds the same values in registers, so .5/aij could not leave the
# loop. And padding quartets ran the full TRR, whose g arrays nobody reads: the HRR, the products and the
# accumulation all skip task_id >= ntasks. Both are bitwise exact; the barriers stay where they are.
if args.generic == 'on':
    s = source(f'{D}/build_rys_gxyz.cuh')
    m4 = s.index('#define BUILD_4C_GXYZ(lj, ll, active)')
    m3 = s.index('#define BUILD_3C_GXYZ')
    mac = s[m4:m3]
    for old, new in (('        double aij = aij_cache[0]; \\\n', ''),
                     ('            double aj_aij = aij_cache[1]; \\\n', ''),
                     ('            for (int n = gout_id; n < 3; n += gout_stride) { \\\n',
                      '            for (int n = gout_id; n < 3 && (active); n += gout_stride) { \\\n')):
        assert mac.count(old) == 1, old
        mac = mac.replace(old, new)
    mac, nrep = re.subn(r'if \(n < lij3\) \{ \\\n', 'if (n < lij3 && (active)) { \\\n', mac)
    assert nrep == 2 and 'aij_cache' not in mac
    open(f'{args.tree}/{D}/build_rys_gxyz.cuh', 'w').write(s[:m4] + mac + s[m3:])
    p = f'{args.tree}/{D}/rys_contract_k.cu'
    s = open(p).read()
    for old, new in (('                    double aij = aij_cache[0];\n', ''),
                     ('                        double aj_aij = aij_cache[1];\n', ''),
                     ('                        for (int n = gout_id; n < 3; n += gout_stride) {\n',
                      '                        for (int n = gout_id; n < 3 && task_id < ntasks; n += gout_stride) {\n'),
                     # the deeper line first: the shallower pattern is a substring of it
                     ('                                if (n < lij3) {\n', '                                if (n < lij3 && task_id < ntasks) {\n'),
                     ('                            if (n < lij3) {\n', '                            if (n < lij3 && task_id < ntasks) {\n')):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    open(p, 'w').write(s)

# Generated multi-thread K kernels (bench/gpu/gen_rys_k.py) for classes that would otherwise run the generic kernel:
# generated_rys_k.cu, built into libgvhf_rys, and tried by RYS_build_k after rys_k_unrolled.
if args.gen_k != 'off':
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, 'gen_rys_k.py'), f'{args.tree}/{D}/generated_rys_k.cu']
                   + args.gen_k.split(','), check=True)
    p = f'{args.tree}/{D}/rys_contract_k.cu'
    s = open(p).read()
    decl = 'extern int rys_k_unrolled(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds,\n'
    assert s.count(decl) == 1
    s = s.replace(decl, ('extern int rys_k_generated(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds,\n'
                         '                           float *q_cond_ij, float *q_cond_kl, float dm_penalty,\n'
                         '                           float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,\n'
                         '                           uint32_t *pool, int *head, int workers);\n') + decl)
    old = ('    if (!rys_k_unrolled(envs, &kmat, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
           '                        s_cond_ij, s_cond_kl, diffuse_exps, pool, head, workers)) {\n')
    assert s.count(old) == 1
    call = 'envs, &kmat, &bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head, workers'
    if args.gen_first:
        new = (f'    int launched = 0;\n'
               f'    const char *gen_first = getenv("GPU4PYSCF_RYS_GEN_FIRST");  // experiment builds only\n'
               f'    const char *gen_off = getenv("GPU4PYSCF_RYS_GEN_OFF");\n'
               f'    int use_gen = !(gen_off && atoi(gen_off));\n'
               f'    if (use_gen && gen_first && atoi(gen_first)) launched = rys_k_generated({call});\n'
               f'    if (!launched) launched = rys_k_unrolled({call});\n'
               f'    if (!launched && use_gen) launched = rys_k_generated({call});\n'
               f'    if (!launched) {{\n')
    else:
        new = ('    if (!rys_k_unrolled(envs, &kmat, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
               '                        s_cond_ij, s_cond_kl, diffuse_exps, pool, head, workers) &&\n'
               '        !rys_k_generated(envs, &kmat, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
               '                         s_cond_ij, s_cond_kl, diffuse_exps, pool, head, workers)) {\n')
    s = s.replace(old, new)
    open(p, 'w').write(s)

s = source(f'{D}/CMakeLists.txt')
for on, old, new in ((args.gen_k != 'off', '  rys_contract_k.cu unrolled_rys_k.cu\n',
                      '  rys_contract_k.cu unrolled_rys_k.cu generated_rys_k.cu\n'),
                     (args.gen_ejk != 'off', '  rys_contract_jk_ip1.cu unrolled_rys_jk_ip1.cu unrolled_ejk_ip1.cu\n',
                      '  rys_contract_jk_ip1.cu unrolled_rys_jk_ip1.cu unrolled_ejk_ip1.cu generated_ejk_ip1.cu\n'
                      + ''.join(f'  generated_ejk_ip1_{f}.cu\n' for f in range(1, args.gen_ejk_files))),
                     (args.gen_ip2 != 'off', '  rys_contract_jk_ip2.cu unrolled_ejk_ip2_type12.cu unrolled_ejk_ip2_type3.cu\n',
                      '  rys_contract_jk_ip2.cu unrolled_ejk_ip2_type12.cu unrolled_ejk_ip2_type3.cu generated_ejk_ip2.cu\n'
                      + ''.join(f'  generated_ejk_ip2_{f}.cu\n' for f in range(1, args.gen_ip2_files)))):
    assert s.count(old) == 1
    if on:
        s = s.replace(old, new)
open(f'{args.tree}/{D}/CMakeLists.txt', 'w').write(s)

s = source(f'{D}/vhf.cuh')
pi_fac = '#define PI_FAC          34.98683665524972497\n'
assert s.count(pi_fac) == 1
s = s.replace(pi_fac, pi_fac + '// exp(-60), the expcutoff of libcint: the Rys J/K kernels skip primitive pairs whose\n'
                               '// c_i c_j exp(-a_i a_j/(a_i+a_j) |R_ij|^2) is smaller\n'
                               '#define PRIM_CUTOFF     8.75651076269652e-27\n')
if args.launch != 'off':
    qd = '#define QUEUE_DEPTH     65536\n'
    assert s.count(qd) == 1
    s = s.replace(qd, qd + '// Task queues of QUEUE_DEPTH per SM in the pools jk.py allocates for RYS_build_k and RYS_build_jk\n'
                           '// (multiProcessorCount * 262144 + 3 entries); the three heads follow the last queue.\n'
                           '#define POOL_BLOCKS_PER_SM 4\n')
open(f'{args.tree}/{D}/vhf.cuh', 'w').write(s)

# Launch shape of the one-thread K and J+K kernels. Every Rys kernel ran one persistent block per SM, with 256
# threads (512 below 129 registers). The one-thread kernels have no barrier inside their task loop, so a warp that has
# finished its share of a bra pair waits at the next one for the slowest warp of the block: 44-50% of their warp time
# (Nsight Compute). Smaller blocks shorten that tail, and the occupancy calculator then fills the SM with as many of
# them as registers and shared memory allow, up to the queues the pool holds.
if args.launch != 'off':
    MULTI = {'2021', '2111', '2120', '2210', '3011', '3110'}
    s = source(f'{D}/unrolled_kernels.cuh')
    anchor = '#define JKENERGY_KERNEL_ARGS \\\n'
    assert s.count(anchor) == 1
    nsq, blocks = (int(x) for x in args.launch.split('x')) if args.launch != 'env' else ('rys_launch_param("GPU4PYSCF_RYS_NSQ", 128)', 'rys_launch_param("GPU4PYSCF_RYS_BLOCKS", POOL_BLOCKS_PER_SM)')
    macro = ('// The one-thread kernels: as many persistent blocks per SM as fit, at most POOL_BLOCKS_PER_SM\n'
             '#define LAUNCH_JKMATRIX_KERNEL_PERSISTENT(KERNEL) { \\\n'
             '    int blocks_per_sm = 1; \\\n'
             '    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, KERNEL, nsq_per_block, buflen*sizeof(double)); \\\n'
             f'    blocks_per_sm = MAX(1, MIN(blocks_per_sm, MIN({blocks}, POOL_BLOCKS_PER_SM))); \\\n'
             '    KERNEL<<<workers*blocks_per_sm, threads, buflen*sizeof(double)>>>( \\\n'
             '    *envs, *jk, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); }\n\n')
    if args.launch == 'env':
        macro = ('#include <stdlib.h>\n'
                 'static inline int rys_launch_param(const char *name, int dflt)\n'
                 '{\n    const char *v = getenv(name);\n    return v ? atoi(v) : dflt;\n}\n') + macro
    s = s.replace(anchor, macro + anchor)
    open(f'{args.tree}/{D}/unrolled_kernels.cuh', 'w').write(s)
    for name in ('unrolled_rys_k.cu', 'unrolled_rys_jk.cu'):
        p = f'{args.tree}/{D}/{name}'
        s = open(p).read()
        sig = 'int rys_jk_unrolled(' if name == 'unrolled_rys_jk.cu' else 'int rys_k_unrolled('
        assert s.count(sig) == 1
        head, disp = s.split(sig)
        disp = sig + disp
        assert disp.count('    int nsq_per_block = 256;\n') == 1
        disp = disp.replace('    int nsq_per_block = 256;\n', f'    int nsq_per_block = {nsq};\n')
        disp, n_adj = re.subn(r'\n        adjust_threads\(rys_k_\d{4}, nsq_per_block\);', '', disp)
        n_launch = 0
        for m in re.findall(r'LAUNCH_JKMATRIX_KERNEL\((rys_j?k_(\d{4}))\)', disp):
            if m[1] not in MULTI:
                disp = disp.replace(f'LAUNCH_JKMATRIX_KERNEL({m[0]})', f'LAUNCH_JKMATRIX_KERNEL_PERSISTENT({m[0]})')
                n_launch += 1
        assert n_launch == 19 and 'adjust_threads' not in disp, (name, n_launch, n_adj)
        open(p, 'w').write(head + disp)
    old_head = '    int *head = (int *)(pool + workers * QUEUE_DEPTH);\n'
    new_head = '    int *head = (int *)(pool + POOL_BLOCKS_PER_SM * workers * QUEUE_DEPTH);\n'
    for name in ('rys_contract_k.cu', 'rys_contract_jk.cu'):
        p = f'{args.tree}/{D}/{name}'
        s = open(p).read()
        assert s.count(old_head) == 1
        s = s.replace(old_head, new_head)
        open(p, 'w').write(s)

    # The same for the one-thread derivative kernels. The energy kernels fix their block size at compile time
    # (constexpr nsq_per_block = 256, and two root offsets written as *256), so the size is fixed here too, and the
    # dispatchers no longer double it below 129 registers (adjust_threads), which would not have matched the constant.
    if args.launch != 'env':
        p = f'{args.tree}/{D}/unrolled_kernels.cuh'
        s = open(p).read()
        anchor = '// The one-thread kernels: as many persistent blocks per SM as fit, at most POOL_BLOCKS_PER_SM\n'
        s = s.replace(anchor, anchor.replace('kernels:', 'K, J+K and J/K-derivative kernels:'), 1)
        s += ('\n// The one-thread energy-derivative kernels: as many persistent blocks per SM as fit, at most POOL_BLOCKS_PER_SM\n'
              '#define LAUNCH_JKENERGY_KERNEL_PERSISTENT(KERNEL) { \\\n'
              '    int blocks_per_sm = 1; \\\n'
              '    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, KERNEL, nsq_per_block, buflen*sizeof(double)); \\\n'
              f'    blocks_per_sm = MAX(1, MIN(blocks_per_sm, MIN({blocks}, POOL_BLOCKS_PER_SM))); \\\n'
              '    KERNEL<<<workers*blocks_per_sm, threads, buflen*sizeof(double)>>>( \\\n'
              '    *envs, *jk, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head); }\n')
        open(p, 'w').write(s)
        dispatchers = {'unrolled_ejk_ip1.cu': ('int rys_ejk_ip1_unrolled(', 'LAUNCH_JKENERGY_KERNEL'),
                       'unrolled_ejk_ip2_type12.cu': ('int rys_ejk_ip2_type12_unrolled(', 'LAUNCH_JKENERGY_KERNEL'),
                       'unrolled_ejk_ip2_type3.cu': ('int rys_ejk_ip2_type3_unrolled(', 'LAUNCH_JKENERGY_KERNEL'),
                       'unrolled_rys_jk_ip1.cu': ('int rys_vjk_ip1_unrolled(', 'LAUNCH_JKMATRIX_KERNEL')}
        for name, (sig, launch) in dispatchers.items():
            p = f'{args.tree}/{D}/{name}'
            s = open(p).read()
            assert s.count(sig) == 1, name
            body, disp = s.split(sig)
            parts = re.split(r'(?=__global__ static\nvoid )', body)
            one_thread = set()
            for k, part in enumerate(parts):
                m = re.match(r'__global__ static\nvoid (\w+)\(', part)
                if not m or 'nsq_per_block, 1, 0);' not in part:
                    continue
                one_thread.add(m.group(1))
                if name != 'unrolled_rys_jk_ip1.cu':
                    assert part.count('    constexpr int nsq_per_block = 256;\n') == 1
                    part = part.replace('    constexpr int nsq_per_block = 256;\n', f'    constexpr int nsq_per_block = {nsq};\n')
                    part = part.replace('rw[(2*irys+1)*256]', 'rw[(2*irys+1)*nsq_per_block]')
                    part = part.replace('rw[ 2*irys   *256]', 'rw[ 2*irys   *nsq_per_block]')
                    assert not re.search(r'\b256\b', part), m.group(1)
                else:
                    assert 'int nsq_per_block = _nsq_per_block;' in part
                parts[k] = part
            disp = sig + disp
            assert disp.count('    int nsq_per_block = 256;\n') == 1
            disp = disp.replace('    int nsq_per_block = 256;\n', f'    int nsq_per_block = {nsq};\n')
            disp = re.sub(r'\n        adjust_threads\(\w+, nsq_per_block\);', '', disp)
            n_launch = 0
            for kern in sorted(one_thread):
                old = f'{launch}({kern});'
                assert disp.count(old) == 1, (name, kern)
                disp = disp.replace(old, f'{launch}_PERSISTENT({kern});')
                n_launch += 1
            assert 'adjust_threads' not in disp and n_launch == len(one_thread)
            open(p, 'w').write(''.join(parts) + disp)
        for name, funcs in (('rys_contract_jk_ip1.cu', ('int RYS_build_jk_ip1(', 'int RYS_per_atom_jk_ip1(')),
                            ('rys_contract_jk_ip2.cu', ('int RYS_per_atom_jk_ip2_type12(', 'int RYS_per_atom_jk_ip2_type3('))):
            p = f'{args.tree}/{D}/{name}'
            s = open(p).read()
            for f in funcs:
                i = s.index(f)
                j = s.index(old_head, i)
                assert s.index('\n}\n', i) > j, f   # the head line lies inside this function
                s = s[:j] + new_head + s[j + len(old_head):]
            open(p, 'w').write(s)
# Generated multi-thread gradient kernels (bench/gpu/gen_rys_ejk.py): generated_ejk_ip1.cu, tried by
# RYS_per_atom_jk_ip1 before rys_ejk_ip1_unrolled, since they also replace its multi-thread kernels. In --gen-first
# builds GPU4PYSCF_RYS_GEN_OFF=1 skips them at run time.
if args.gen_ejk != 'off':
    here = os.path.dirname(os.path.abspath(__file__))
    for f in range(1, 64):   # earlier runs may have written more files
        stale = f'{args.tree}/{D}/generated_ejk_ip1_{f}.cu'
        if os.path.exists(stale):
            os.remove(stale)
    subprocess.run([sys.executable, os.path.join(here, 'gen_rys_ejk.py'), f'{args.tree}/{D}/generated_ejk_ip1.cu',
                    '--files', str(args.gen_ejk_files)] + args.gen_ejk.split(','), check=True)
    p = f'{args.tree}/{D}/rys_contract_jk_ip1.cu'
    s = open(p).read()
    decl = 'extern int rys_ejk_ip1_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,\n'
    assert s.count(decl) == 1
    s = s.replace(decl, ('extern int rys_ejk_ip1_generated(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,\n'
                         '                                 float *q_cond_ij, float *q_cond_kl, float dm_penalty,\n'
                         '                                 float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,\n'
                         '                                 uint32_t *pool, double *dd_pool, int *head, int workers);\n')
                  + decl)
    old = ('    if (!rys_ejk_ip1_unrolled(&envs, &jk, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
           '                              s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers)) {\n')
    assert s.count(old) == 1
    gen = ('rys_ejk_ip1_generated(&envs, &jk, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
           '                               s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers)')
    if args.gen_first:
        new = ('    const char *gen_off = getenv("GPU4PYSCF_RYS_GEN_OFF");  // experiment builds only\n'
               '    int use_gen = !(gen_off && atoi(gen_off));\n'
               '    if (!(use_gen && ' + gen.replace('\n ', '\n        ') + ') &&\n'
               '        !rys_ejk_ip1_unrolled(&envs, &jk, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
               '                              s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers)) {\n')
    else:
        new = ('    if (!' + gen + ' &&\n'
               '        !rys_ejk_ip1_unrolled(&envs, &jk, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
               '                              s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers)) {\n')
    s = s.replace(old, new)
    open(p, 'w').write(s)

# Generated multi-thread second-derivative kernels (bench/gpu/gen_rys_ejk_ip2.py): generated_ejk_ip2.cu, tried by
# RYS_per_atom_jk_ip2_type12 and _type3 before their unrolled and generic kernels. In --gen-ip2-env builds
# GPU4PYSCF_RYS_GEN_IP2_OFF=1 skips them at run time.
if args.gen_ip2 != 'off':
    here = os.path.dirname(os.path.abspath(__file__))
    for f in range(1, 64):   # earlier runs may have written more files
        stale = f'{args.tree}/{D}/generated_ejk_ip2_{f}.cu'
        if os.path.exists(stale):
            os.remove(stale)
    subprocess.run([sys.executable, os.path.join(here, 'gen_rys_ejk_ip2.py'), f'{args.tree}/{D}/generated_ejk_ip2.cu',
                    '--files', str(args.gen_ip2_files)] + args.gen_ip2.split(','), check=True)
    p = f'{args.tree}/{D}/rys_contract_jk_ip2.cu'
    s = open(p).read()
    for kind in ('type12', 'type3'):
        decl = f'extern int rys_ejk_ip2_{kind}_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,\n'
        assert s.count(decl) == 1, decl
        pad = ' ' * len(f'extern int rys_ejk_ip2_{kind}_generated(')
        s = s.replace(decl, (f'extern int rys_ejk_ip2_{kind}_generated(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,\n'
                             f'{pad}float *q_cond_ij, float *q_cond_kl, float dm_penalty,\n'
                             f'{pad}float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,\n'
                             f'{pad}uint32_t *pool, double *dd_pool, int *head, int workers);\n') + decl)
        call = ('(&envs, &jk, &bounds, q_cond_ij, q_cond_kl, dm_penalty,\n'
                '{pad}s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers)')
        old = f'    if (!rys_ejk_ip2_{kind}_unrolled' + call.format(pad=' ' * len(f'    if (!rys_ejk_ip2_{kind}_unrolled(')) + ') {\n'
        assert s.count(old) == 1, old
        gen = f'rys_ejk_ip2_{kind}_generated' + call.format(pad=' ' * len(f'        !rys_ejk_ip2_{kind}_generated('))
        unr = f'rys_ejk_ip2_{kind}_unrolled' + call.format(pad=' ' * len(f'        !rys_ejk_ip2_{kind}_unrolled('))
        if args.gen_ip2_env:
            new = ('    const char *gen_off = getenv("GPU4PYSCF_RYS_GEN_IP2_OFF");  // experiment builds only\n'
                   '    int use_gen = !(gen_off && atoi(gen_off));\n'
                   f'    if (!(use_gen && {gen}) &&\n        !{unr}) {{\n')
        else:
            new = f'    if (!{gen} &&\n        !{unr}) {{\n'
        s = s.replace(old, new)
    open(p, 'w').write(s)

# J engine (gvhf-md): each block of the unrolled MD kernels takes a fixed tile of TILEX x TILEY batches of 16 bra and
# ket pairs, so small systems get small grids (part 18: 7-24 blocks on 20 SMs for caffeine/def2-SVP, 26% of the J
# build with SMs idle). The bra tile only sets the batch loop, the batch indices of the batch-level screening
# (qd_ij_max holds the maximum per 16-pair batch, independent of the tile) and the block's triangle skip; the ket tile
# also lays out shared memory and stays. With --j-tiles auto each kernel derives its bra tile from gridDim.x, and the
# launcher (md_j_grid_x) picks gridDim.x so that a launch has at least --j-blocks-per-sm blocks per SM (twice that for
# diagonal classes, whose blocks above the diagonal return at once), with the compiled TILEX as the largest tile, and
# no more blocks than the tiles fill (with t = ceil(n/g) batches per tile, ceil(n/t) blocks give the kernel the same t).
# The integrals and the screening stay the same; only the order of the atomic adds changes.
if args.j_tiles != 'off':
    tile_re = re.compile(r'// TILEX=(\d+), TILEY=(\d+)\n')
    grid_x = (
        '#include <algorithm>\n'
        '// gridDim.x of an unrolled J launch: bra tiles of at most tilex_max batches of 16 pairs, and enough of them\n'
        '// that the grid has at least MD_J_BLOCKS_PER_SM blocks per SM (twice that if bra and ket share their pairs)\n'
        f'#define MD_J_BLOCKS_PER_SM {args.j_blocks_per_sm}\n'
        'static int md_j_grid_x(int npairs_ij, int tilex_max, int grid_y, int diagonal)\n'
        '{\n'
        '    int dev, n_sm;\n'
        '    cudaGetDevice(&dev);\n'
        '    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);\n'
        '    int nbatch = (npairs_ij + 15) / 16;\n'
        '    int gx = (nbatch + tilex_max - 1) / tilex_max;\n'
        '    int want = MD_J_BLOCKS_PER_SM * n_sm * (diagonal ? 2 : 1);\n'
        '    if (gx * grid_y < want && nbatch > 0) {\n'
        '        gx = std::min(nbatch, (want + grid_y - 1) / grid_y);\n'
        '        // only as many blocks as the kernel\'s tiles of ceil(nbatch / gridDim.x) batches fill: a block past\n'
        '        // the last tile would read q_cond_ij past its end before returning\n'
        '        int tilex = (nbatch + gx - 1) / gx;\n'
        '        gx = (nbatch + tilex - 1) / tilex;\n'
        '    }\n'
        '    return std::max(gx, 1);\n'
        '}\n\n')
    n_kernels = {}
    for f in ('unrolled_md_j.cu', 'unrolled_md_j_4dm.cu'):
        rel = f'gpu4pyscf/lib/gvhf-md/{f}'
        s = source(rel)
        launcher = s.index('\nint md_j_unrolled(' if f == 'unrolled_md_j.cu' else '\nint md_j_4dm_unrolled(')
        head, disp = s[:launcher], s[launcher:]
        parts = tile_re.split(head)
        out = [parts[0]]
        tiles = []
        for k in range(1, len(parts), 3):
            tx, ty, body = int(parts[k]), int(parts[k+1]), parts[k+2]
            tiles.append((tx, ty))
            subs = [(f'block_x * {16*tx};', 'block_x * (16*tilex);', 1),
                    (f'batch_ij < {tx}; ++batch_ij', 'batch_ij < tilex; ++batch_ij', 1),
                    (f'blockIdx.x * {16*tx} + batch_ij * 16', 'blockIdx.x * (16*tilex) + batch_ij * 16', 1),
                    (f'qd_ij_max[batch_ij+blockIdx.x*{tx}]', 'qd_ij_max[batch_ij+blockIdx.x*tilex]', 1),
                    ('    KERNEL_SETUP();\n',
                     '    KERNEL_SETUP();\n'
                     '    // the bra tile, from the grid that md_j_grid_x chose\n'
                     '    int tilex = ((bounds.npairs_ij + 15) / 16 + gridDim.x - 1) / gridDim.x;\n', 1)]
            for old, new, cnt in subs:
                assert body.count(old) == cnt, (f, tx, ty, old, body.count(old))
                body = body.replace(old, new)
            old = f'task_ij0+{16*tx} <= task_kl0'
            assert body.count(old) <= 1 and 16*tx != 16, (f, tx, old)
            body = body.replace(old, 'task_ij0+16*tilex <= task_kl0')
            out += [f'// TILEX={tx} (largest; run-time tilex), TILEY={ty}\n', body]
        head = ''.join(out)
        for tx, ty in set(tiles):
            old = f'dim3 blocks((npairs_ij + {16*tx-1}) / {16*tx}, (npairs_kl + {16*ty-1}) / {16*ty}, 1);'
            gy = f'(npairs_kl + {16*ty-1}) / {16*ty}'
            new = (f'dim3 blocks(md_j_grid_x(npairs_ij, {tx}, {gy}, bounds->pair_ij_mapping == bounds->pair_kl_mapping),'
                   f'\n                    {gy}, 1);')
            assert disp.count(old) == tiles.count((tx, ty)), (f, tx, ty, disp.count(old))
            disp = disp.replace(old, new)
        n_kernels[f] = len(tiles)
        open(f'{args.tree}/{rel}', 'w').write(head + '\n' + grid_x + disp.lstrip('\n'))
    print(f'J engine: run-time bra tiles in {n_kernels["unrolled_md_j.cu"]} + {n_kernels["unrolled_md_j_4dm.cu"]} '
          f'unrolled kernels, at least {args.j_blocks_per_sm} blocks per SM')

# VV10's Fock kernel (gdft/vv10.cu, VXC_vv10nlc_fock_eval_UWE; the SCF's nr_nlc_vxc and the gradient's NLC terms, part
# 26). It summed over all ordered pairs of grid points; Phi_ij, r_ij^2, g_ij + g_ji and the reciprocal are symmetric.
# With --vv10 sym each unordered pair is evaluated once and added to both points: tiles of 128 points, a block takes one
# I tile against a run of VV10_JTILES tiles J >= I; in a warp each lane keeps its i and meets the 32 j's of a chunk in
# turn, the j side's sums travelling with their j from lane to lane (a shuffle per step), so no per-j reduction is
# needed. U, W, E are summed with atomics and scaled after. Same terms; shorter sums (bench/gpu/vv10_check.cu: 1.43x on
# 60000 synthetic points, and 10-40x closer to a long double reference than the unpatched kernel's sequential sums).
if args.vv10 == 'sym':
    p = f'{args.tree}/gpu4pyscf/lib/gdft/vv10.cu'
    s = source('gpu4pyscf/lib/gdft/vv10.cu')
    proto = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vv10_check.cu')).read()
    k0 = proto.index('#define VV10_TILE       128\n')
    k1 = proto.index('// The symmetric version: block (I, g) takes')
    k2 = proto.index('// The symmetric version without per-j reductions:')
    k3 = proto.index('static void run_sym(')
    new_kernels = proto[k0:k1] + proto[k2:k3]
    i0 = s.index('__global__\nstatic void vv10_fock_eval_UWE_kernel(')
    i1 = s.index('__global__\nstatic void vv10_fock_eval_omega_derivative_kernel(')
    s = s[:i0] + ('#define VV10_JTILES     4\n'
                  '// VV10_TILE (128) points per tile; VV10_JTILES J tiles per block (part 26)\n') + new_kernels + s[i1:]
    old = ("    const dim3 threads(NG_PER_BLOCK);\n"
           "    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);\n"
           "    vv10_fock_eval_UWE_kernel<<<blocks, threads, 0, stream>>>(U, W, E,\n"
           "                                                              grid_coord, rho_weight, omega, kappa, ngrids);\n")
    assert s.count(old) == 1, 'vv10 launcher'
    new = ("    // each pair of points once (part 26): the kernel sums into U, W and E, vv10_scale_UWE_kernel applies the factors\n"
           "    if (ngrids <= 0) {\n"
           "        return 0;\n"
           "    }\n"
           "    cudaMemsetAsync(U, 0, ngrids * sizeof(double), stream);\n"
           "    cudaMemsetAsync(W, 0, ngrids * sizeof(double), stream);\n"
           "    cudaMemsetAsync(E, 0, ngrids * sizeof(double), stream);\n"
           "    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;\n"
           "    const dim3 blocks(ntiles, (ntiles + VV10_JTILES - 1) / VV10_JTILES);\n"
           "    vv10_fock_eval_UWE_rot_kernel<<<blocks, VV10_TILE, 0, stream>>>(U, W, E, grid_coord, rho_weight, omega, kappa,\n"
           "                                                                    ngrids, VV10_JTILES);\n"
           "    vv10_scale_UWE_kernel<<<(ngrids + 255) / 256, 256, 0, stream>>>(U, W, E, ngrids);\n")
    s = s.replace(old, new)
    # the grid response's pair term (part 28): de[B] = 3 sum_{j on B} T_j with T_j = sum over points i on other atoms of
    # K_ij (r_i - r_j), K_ij = rho_w_i rho_w_j Phi_ij f_ij symmetric, each unordered pair once by the same rotation;
    # grad.rks.get_nlc_exc_full_response calls it in place of the per-(atom, point) kernel and its batches
    gproto = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vv10_grid_check.cu')).read()
    g0 = gproto.index('// part 28: T_j = sum over i on other atoms')
    g1 = gproto.index('__global__ static void T_to_atoms(')
    anchor = 'extern "C" {\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, gproto[g0:g1] + anchor + (
        '// VV10 gradient, the grid response\'s pair term (part 28): T (3 x ngrids, zeroed here) with T_j = sum over the\n'
        '// points i on other atoms of K_ij (r_i - r_j); the term for atom B is 3 sum_{j on B} T_j\n'
        '__host__\n'
        'int VXC_vv10nlc_grad_grid_response_T(const cudaStream_t stream, double* T, const double* grid_coord,\n'
        '                                     const double* rho_weight, const double* omega, const double* kappa,\n'
        '                                     const int* atom, const int ngrids)\n'
        '{\n'
        '    if (ngrids <= 0) {\n'
        '        return 0;\n'
        '    }\n'
        '    cudaMemsetAsync(T, 0, 3 * ngrids * sizeof(double), stream);\n'
        '    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;\n'
        '    const dim3 blocks(ntiles, (ntiles + VV10_JTILES - 1) / VV10_JTILES);\n'
        '    vv10_grad_grid_response_T_kernel<<<blocks, VV10_TILE, 0, stream>>>(T, grid_coord, rho_weight, omega, kappa,\n'
        '                                                                       atom, ngrids, VV10_JTILES);\n'
        '    const cudaError_t err = cudaGetLastError();\n'
        '    if (err != cudaSuccess) {\n'
        '        fprintf(stderr, "CUDA Error of vv10 grad grid response T: %s\\n", cudaGetErrorString(err));\n'
        '        return 1;\n'
        '    }\n'
        '    return 0;\n'
        '}\n\n'))
    open(p, 'w').write(s)
    print('VV10 Fock kernel: each pair of grid points once (rotation within warps); the grid response\'s pair term too')

# The MD J engine's Boys function (gvhf-rys/gamma_inc.cu, eval_gamma_inc_fn, which only gvhf-md includes; part 25). The
# series branch (t < m/2 + 1/2) divided once per term, about 20 terms, and once per step of the downward recursion: a
# DDIV each, on the 6-12% of lanes that take the branch while the rest of their warp waits (part 23's profile: that one
# line was 18-30% of the low-order J kernels' instructions). With --boys new both multiply by 1/(n + 1/2) from a table
# in constant memory (the same index on all lanes), which changes F_n by at most 8 ulps (bench/gpu/boys_check.cu).
# The erf branch skips erf(sqrt(t)) for t >= 49, where CUDA's erf is exactly 1 (erfc(7) = 4e-23; boys_check.cu tests
# 2^30 points of [7, 40]), so those values are the same to the bit.
if args.boys == 'new':
    p = f'{args.tree}/gpu4pyscf/lib/gvhf-rys/gamma_inc.cu'
    s = source('gpu4pyscf/lib/gvhf-rys/gamma_inc.cu')
    old = ("    } else if (m > 0 && t < m*.5+.5) {\n"
           "        double bi = m + .5;\n"
           "        double e = .5 * exp(-t);\n"
           "        double x = e;\n"
           "        double s = e;\n"
           "        double tol = EPS_FLOAT64 * e;\n"
           "        while (x > tol) {\n"
           "            bi += 1.;\n"
           "            x *= t / bi;\n"
           "            s += x;\n"
           "        }\n"
           "        double b = m + 0.5;\n"
           "        double fval = s / b;\n"
           "        f[sq_id + m*block_size] = fval;\n"
           "        for (int i = m-1; i >= 0; i--) {\n"
           "            b -= 1.;\n"
           "            fval = (e + t * fval) / b;\n"
           "            f[sq_id + i*block_size] = fval;\n"
           "        }\n"
           "    } else {\n"
           "        double tt = sqrt(t);\n"
           "        double fval = SQRTPIE4 / tt * erf(tt);\n")
    new = ("    } else if (m > 0 && t < m*.5+.5) {\n"
           "        double e = .5 * exp(-t);\n"
           "        double x = e;\n"
           "        double s = e;\n"
           "        double tol = EPS_FLOAT64 * e;\n"
           "        // the terms shrink at least twofold (t/(k+1/2) < 1/2), so k stays below m + 55\n"
           "        for (int k = m+1; x > tol && k < BOYS_NINV; k++) {\n"
           "            x *= t * c_inv_half[k];\n"
           "            s += x;\n"
           "        }\n"
           "        double fval = s * c_inv_half[m];\n"
           "        f[sq_id + m*block_size] = fval;\n"
           "        for (int i = m-1; i >= 0; i--) {\n"
           "            fval = (e + t * fval) * c_inv_half[i];\n"
           "            f[sq_id + i*block_size] = fval;\n"
           "        }\n"
           "    } else {\n"
           "        double tt = sqrt(t);\n"
           "        double fval = SQRTPIE4 / tt;\n"
           "        if (t < ERF_ONE_T) {\n"
           "            fval *= erf(tt);\n"
           "        }\n")
    assert s.count(old) == 1, 'gamma_inc.cu'
    s = s.replace(old, new)
    old = "#define SQRTPIE4        .886226925452758013\n"
    assert s.count(old) == 1
    table = ',\n'.join('    ' + ', '.join(repr(1. / (k + .5)) for k in range(k0, min(k0 + 4, 96))) for k0 in range(0, 96, 4))
    s = s.replace(old, old + (
        "// 1/(n + 1/2): the series and the downward recursion multiply where they divided (my-pyscf part 25)\n"
        "#define BOYS_NINV       96\n"
        "static __constant__ double c_inv_half[BOYS_NINV] = {\n" + table + "\n};\n"
        "// erf(sqrt(t)) is exactly 1 in double precision from here on (erfc(7) = 4.2e-23)\n"
        "#define ERF_ONE_T       49.\n"))
    open(p, 'w').write(s)
    print('Boys function: tabulated 1/(n+1/2) in the series and the downward recursion, erf skipped for t >= 49')

# The J kernels' quartet prefactor (part 27): each quartet took fac / (aij*akl*sqrt(aij+akl)) and aij*akl/(aij+akl), two
# divisions and a square root (33% of (ss|ss)'s FP64 work, 10-13% of the other low-order classes'). With --j-prefactor
# rsqrt the pair caches keep 1/aij (Rp, a fifth row of 16) and 1/akl (Rq, a fifth row of the ket tile), computed once
# per pair, and the quartet takes one rsqrt(aij + akl): theta = aij*akl*rs^2, fac *= (1/aij)(1/akl)rs. rsqrt is within
# 1 ulp, so theta and fac change by a few ulps. Everything after the caches moves up (16 doubles after Rp, the ket tile's
# size after Rq), and so does each launch's shared memory. The unrolled files are patched here, the generated ones by
# gen_md_j.py --prefactor; the generic kernels (md_j_1dm_kernel, md_j_4dm_kernel) keep theirs.
def j_prefactor_patch(text, fname):
    kernels = re.split(r'(?=\nvoid md_j_)', text)   # each kernel's body starts a piece
    out = [kernels[0]]
    n = 0
    for k in kernels[1:]:
        m = re.search(r'double \*Rp_cache = Rq_cache \+ (\d+);', k)
        if not m:
            out.append(k)
            continue
        bsy4 = int(m.group(1))
        assert bsy4 % 4 == 0
        bsy = bsy4 // 4
        assert re.search(rf'int task_kl0 = block_y \* {bsy};', k), (fname, bsy)
        subs = [
            (rf'double \*Rp_cache = Rq_cache \+ {bsy4};', f'double *Rp_cache = Rq_cache + {5 * bsy};'),
            (r'double \*dm_ij_cache = Rp_cache \+ 64 \+ tx;', 'double *dm_ij_cache = Rp_cache + 80 + tx;'),
            (r'(\n(\s*)Rq_cache\[n\+' + str(3 * bsy) + r'\] = akl;)', r'\1\n\g<2>Rq_cache[n+' + str(4 * bsy) + '] = 1. / akl;'),
            (r'(\n(\s*)Rq_cache\[n\+' + str(3 * bsy) + r'\] = 1\.;)', r'\1\n\g<2>Rq_cache[n+' + str(4 * bsy) + '] = 1.;'),
            (r'(\n(\s*)Rp_cache\[thread_id\+48\] = aij;)', r'\1\n\g<2>Rp_cache[thread_id+64] = 1. / aij;'),
            (r'(\n(\s*)Rp_cache\[thread_id\+48\] = 1\.; // aij)', r'\1\n\g<2>Rp_cache[thread_id+64] = 1.;'),
            (r'(\n(\s*)double aij = Rp_cache\[tx\+48\];)', r'\1\n\g<2>double inv_aij = Rp_cache[tx+64];'),
            (r'(\n(\s*)double akl = Rq_cache\[sq_kl\+' + str(3 * bsy) + r'\];)',
             r'\1\n\g<2>double inv_akl = Rq_cache[sq_kl+' + str(4 * bsy) + '];'),
            (r'\n(\s*)fac = fac / \(aij\*akl\*sqrt\(aij\+akl\)\);',
             r'\n\1double rs = rsqrt(aij+akl);\n\1fac = fac * (inv_aij * inv_akl * rs);'),
            (r'double theta = aij \* akl / \(aij \+ akl\);', 'double theta = aij * akl * (rs * rs);'),
        ]
        for pat, rep in subs:
            k, cnt = re.subn(pat, rep, k)
            assert cnt == 1, (fname, bsy, pat, cnt)
        g = re.search(r'double \*gamma_inc = Rp_cache \+ (\d+) \+ sq_id;', k)
        k = k.replace(g.group(0), f'double *gamma_inc = Rp_cache + {int(g.group(1)) + 16} + sq_id;')
        # the first zeroing loop covers the caches up to Rp's first 64 (vj_kl_cache onwards, or Rq onwards with 4 dms)
        z = re.search(r'for \(int n = thread_id; n < (\d+); n \+= 256\) \{\n(\s*)(vj_kl_cache|Rq_cache)\[n\] = 0\.;\n\s*\}\n\s*__syncthreads\(\);', k)
        assert z, (fname, bsy, 'zeroing')
        k = k.replace(z.group(0), z.group(0).replace(f'n < {z.group(1)};', f'n < {int(z.group(1)) + bsy + 16};', 1))
        out.append(k)
        n += 1
    text = ''.join(out)
    # the launches: each case's shared memory grows by the ket tile (one row) and 16 (the bra row). A launch above 48 KB
    # needs cudaFuncSetAttribute, which the unpatched code calls only where it was already above (with the range-
    # separated buffer, md_j_4dm_3_1 now crosses it: test_j_engine_multiple_dms), so every launch gets one
    def case_fix(m):
        bsy = 16 * int(m.group(2))
        body = re.sub(r'\((\d+)\+addition_buf\)', lambda q: f'({int(q.group(1)) + bsy + 16}+addition_buf)', m.group(3))
        if 'cudaFuncSetAttribute' not in body:
            body = re.sub(r'\n(\s*)LAUNCH_KERNEL\((\w+), (\(\d+\+addition_buf\)\*sizeof\(double\))\);',
                          r'\n\1cudaFuncSetAttribute(\2, cudaFuncAttributeMaxDynamicSharedMemorySize, \3);'
                          r'\n\1LAUNCH_KERNEL(\2, \3);', body, count=1)
        return m.group(1) + body
    text, ncase = re.subn(r'(case \d+: \{ // lij=\d, lkl=\d, tilex=\d+, tiley=(\d+)\n)(.*?)(?=\n    case |\n    default:)',
                          case_fix, text, flags=re.S)
    assert ncase == n, (fname, ncase, n)
    return text, n

if args.j_prefactor == 'rsqrt':
    for f in ('unrolled_md_j.cu', 'unrolled_md_j_4dm.cu'):
        rel = f'gpu4pyscf/lib/gvhf-md/{f}'
        path = f'{args.tree}/{rel}'
        text = open(path).read() if args.j_tiles != 'off' else source(rel)
        text, n = j_prefactor_patch(text, f)
        open(path, 'w').write(text)
        print(f'J prefactor: 1/aij, 1/akl with the pairs and one rsqrt per quartet in {n} kernels of {f}')

# Generated MD J kernels (bench/gpu/gen_md_j.py) for classes that would otherwise run the generic kernel:
# generated_md_j.cu, built into libgvhf_md, and tried by MD_build_j after md_j_unrolled (single density). With --j-grad,
# the J engine's gradient (part 24): MD_build_j_ip1, MD_build_j with both pairs' Hermite orders one higher (their
# densities in that layout, the top components zero), and md_j_grad.cu (Rt_ip1_dot_dm: the pairs' derivative Hermite
# densities against those potentials, per atom).
G = 'gpu4pyscf/lib/gvhf-md'
here = os.path.dirname(os.path.abspath(__file__))
s = source(f'{G}/md_contract_j.cu')
cmake_files = ''
if args.gen_j != 'off':
    subprocess.run([sys.executable, os.path.join(here, 'gen_md_j.py'), f'{args.tree}/{G}/generated_md_j.cu',
                    f'--prefactor={args.j_prefactor}'] + args.gen_j.split(), check=True)
    decl = 'int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
    assert s.count(decl) == 1
    s = s.replace(decl, ('int md_j_generated(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
                         '                   float *q_cond_ij, float *q_cond_kl, double omega);\n') + decl)
    old = '        if (!md_j_unrolled(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega)) {\n'
    assert s.count(old) == 1
    s = s.replace(old, '        if (!md_j_unrolled(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega) &&\n'
                       '            !md_j_generated(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega)) {\n')
    cmake_files += ' generated_md_j.cu'
if args.j_grad == 'on':
    args_c = ('double *vj, double *dm, int n_dm, int dm_size,\n'
              '                RysIntEnvVars *envs, int *scheme, int *shls_slice,\n'
              '                int npairs_ij, int npairs_kl,\n'
              '                int *pair_ij_mapping, int *pair_kl_mapping,\n'
              '                int *pair_ij_loc, int *pair_kl_loc,\n'
              '                float *qd_ij_max, float *qd_kl_max,\n'
              '                float *q_cond_ij, float *q_cond_kl, float cutoff,\n'
              '                int *atm, int natm, int *bas, int nbas, double *env)\n')
    old = 'int MD_build_j(' + args_c
    assert s.count(old) == 1
    s = s.replace(old, 'static int md_build_j(int lshift, ' + args_c)
    old = ('    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];\n'
           '    int lij = li + lj;\n'
           '    int lkl = lk + ll;\n'
           '    int order = lij + lkl;\n'
           '    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;\n')
    assert s.count(old) == 1
    s = s.replace(old, ('    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];\n'
                        '    // MD_build_j_ip1: both pairs\' Hermite potentials lshift orders above their densities\n'
                        '    li += lshift;\n'
                        '    lk += lshift;\n'
                        '    int lij = li + lj;\n'
                        '    int lkl = lk + ll;\n'
                        '    int order = lij + lkl;\n'
                        '    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;\n'))
    old = ('        fprintf(stderr, "CUDA Error in MD_build_j: %s\\n", cudaGetErrorString(err));\n'
           '        return 1;\n'
           '    }\n'
           '    return 0;\n'
           '}\n')
    assert s.count(old) == 1, 'MD_build_j end'
    call = ('vj, dm, n_dm, dm_size, envs, scheme, shls_slice, npairs_ij, npairs_kl, pair_ij_mapping, '
            'pair_kl_mapping,\n                      pair_ij_loc, pair_kl_loc, qd_ij_max, qd_kl_max, q_cond_ij, '
            'q_cond_kl, cutoff, atm, natm, bas, nbas, env')
    s = s.replace(old, old + ('\nint MD_build_j(' + args_c + '{\n    return md_build_j(0, ' + call + ');\n}\n'
                              '\n// The J engine\'s gradient: the Hermite potentials of both pairs one order above their densities, which\n'
                              '// are given in that layout with their top-order components zero\n'
                              'int MD_build_j_ip1(' + args_c + '{\n    return md_build_j(1, ' + call + ');\n}\n'))
    shutil.copyfile(os.path.join(here, 'md_j_grad.cu'), f'{args.tree}/{G}/md_j_grad.cu')
    cmake_files += ' md_j_grad.cu'
    if args.gen_j_ip1 != 'off':
        subprocess.run([sys.executable, os.path.join(here, 'gen_md_j.py'), f'{args.tree}/{G}/generated_md_j_ip1.cu',
                        f'--prefactor={args.j_prefactor}'] + ['g' + c for c in args.gen_j_ip1.split()], check=True)
        decl = 'int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
        assert s.count(decl) == 1
        s = s.replace(decl, ('int md_j_generated_ip1(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
                             '                       float *q_cond_ij, float *q_cond_kl, double omega);\n') + decl)
        old = '        if (!md_j_unrolled(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega)'
        assert s.count(old) == 1
        s = s.replace(old, '        if (!(lshift && md_j_generated_ip1(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega)) &&\n'
                           '            !md_j_unrolled(envs, &jk, &bounds, q_cond_ij, q_cond_kl, omega)')
        cmake_files += ' generated_md_j_ip1.cu'
open(f'{args.tree}/{G}/md_contract_j.cu', 'w').write(s)
s = source(f'{G}/CMakeLists.txt')
old = '  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu\n'
assert s.count(old) == 1
s = s.replace(old, '  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu' + cmake_files + '\n')
open(f'{args.tree}/{G}/CMakeLists.txt', 'w').write(s)

print(f'one-thread kernels (K, J+K): {counts["unrolled_rys_k.cu"][0]} + {counts["unrolled_rys_jk.cu"][0]}: '
      f'barrier {args.barrier}, ket skip {args.ket}, bra skip {args.bra}; multi-thread kernels '
      f'{counts["unrolled_rys_k.cu"][1]} + {counts["unrolled_rys_jk.cu"][1]} and the two generic kernels: '
      f'bra skip {args.bra if args.multi == "on" else "none"}')
print('derivative kernels (one-thread, multi-thread): ' + ', '.join(f'{k} {v}' for k, v in deriv.items()) +
      f'; {n_generic} generic kernels with the bra skip')
