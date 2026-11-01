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
--gen-ip2 writes generated_ejk_ip2.cu (gen_rys_ejk_ip2.py): the Hessian's second-derivative kernels with sums per
Cartesian tuple, tried by RYS_per_atom_jk_ip2_type12 and _type3 before their unrolled and generic kernels.
Usage: exp_prim_screen.py TREE [--base perf] [--barrier keep|remove] [--ket none|zero|cutoff]
                               [--bra none|zero|cutoff] [--multi on|off] [--launch 128x4|off|NSQxBLOCKS|env]
                               [--generic on|off] [--gen-k CLASS:NSQxGS,...|off] [--gen-first]
                               [--gen-ip2 KIND:CLASS:NSQxGS,...|off] [--gen-ip2-files N] [--gen-ip2-env]"""
import argparse
import os
import re
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

print(f'one-thread kernels (K, J+K): {counts["unrolled_rys_k.cu"][0]} + {counts["unrolled_rys_jk.cu"][0]}: '
      f'barrier {args.barrier}, ket skip {args.ket}, bra skip {args.bra}; multi-thread kernels '
      f'{counts["unrolled_rys_k.cu"][1]} + {counts["unrolled_rys_jk.cu"][1]} and the two generic kernels: '
      f'bra skip {args.bra if args.multi == "on" else "none"}')
print('derivative kernels (one-thread, multi-thread): ' + ', '.join(f'{k} {v}' for k, v in deriv.items()) +
      f'; {n_generic} generic kernels with the bra skip')
