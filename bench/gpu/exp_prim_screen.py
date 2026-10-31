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

s = source(f'{D}/vhf.cuh')
pi_fac = '#define PI_FAC          34.98683665524972497\n'
assert s.count(pi_fac) == 1
s = s.replace(pi_fac, pi_fac + '// exp(-60), the expcutoff of libcint: the K and J+K kernels skip primitive pairs whose\n'
                               '// c_i c_j exp(-a_i a_j/(a_i+a_j) |R_ij|^2) is smaller\n'
                               '#define PRIM_CUTOFF     8.75651076269652e-27\n')
open(f'{args.tree}/{D}/vhf.cuh', 'w').write(s)
print(f'one-thread kernels (K, J+K): {counts["unrolled_rys_k.cu"][0]} + {counts["unrolled_rys_jk.cu"][0]}: '
      f'barrier {args.barrier}, ket skip {args.ket}, bra skip {args.bra}; multi-thread kernels '
      f'{counts["unrolled_rys_k.cu"][1]} + {counts["unrolled_rys_jk.cu"][1]} and the two generic kernels: '
      f'bra skip {args.bra if args.multi == "on" else "none"}')
