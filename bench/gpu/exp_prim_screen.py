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
PRIM_CUTOFF (vhf.cuh) is e^-60 = 8.8e-27, libcint's expcutoff. A skipped primitive pair enters its integrals with a
factor below that, many orders of magnitude under the SCF's 1e-13 shell-quartet cutoff.
Usage: exp_prim_screen.py TREE [--base perf] [--barrier keep|remove] [--ket none|zero|cutoff]
                               [--bra none|zero|cutoff] [--multi on|off]"""
import argparse
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('tree')
ap.add_argument('--base', default='perf')
ap.add_argument('--barrier', default='remove', choices=['keep', 'remove'])
ap.add_argument('--ket', default='cutoff', choices=['none', 'zero', 'cutoff'])
ap.add_argument('--bra', default='cutoff', choices=['none', 'zero', 'cutoff'])
ap.add_argument('--multi', default='on', choices=['on', 'off'],
                help='also the bra skip in the multi-thread and generic kernels')
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

s = source(f'{D}/vhf.cuh')
pi_fac = '#define PI_FAC          34.98683665524972497\n'
assert s.count(pi_fac) == 1
s = s.replace(pi_fac, pi_fac + '// exp(-60), the expcutoff of libcint: the Rys J/K kernels skip primitive pairs whose\n'
                               '// c_i c_j exp(-a_i a_j/(a_i+a_j) |R_ij|^2) is smaller\n'
                               '#define PRIM_CUTOFF     8.75651076269652e-27\n')
open(f'{args.tree}/{D}/vhf.cuh', 'w').write(s)
print(f'one-thread kernels (K, J+K): {counts["unrolled_rys_k.cu"][0]} + {counts["unrolled_rys_jk.cu"][0]}: '
      f'barrier {args.barrier}, ket skip {args.ket}, bra skip {args.bra}; multi-thread kernels '
      f'{counts["unrolled_rys_k.cu"][1]} + {counts["unrolled_rys_jk.cu"][1]} and the two generic kernels: '
      f'bra skip {args.bra if args.multi == "on" else "none"}')
print('derivative kernels (one-thread, multi-thread): ' + ', '.join(f'{k} {v}' for k, v in deriv.items()) +
      f'; {n_generic} generic kernels with the bra skip')
