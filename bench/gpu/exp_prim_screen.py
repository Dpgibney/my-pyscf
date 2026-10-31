"""Experiment: patch the one-thread-per-quartet Rys K kernels in gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu (19 of 25;
those call rys_roots_for_k with stride 1, so no other thread reads a thread's roots). Starts from the committed
file each time.
  --barrier remove   drop the __syncthreads() at the top of the bra-primitive loop (needed before any ket skip,
                     since threads skip different ket primitives)
  --ket zero|cutoff  skip a ket primitive pair when ckcl == 0 (padding threads, underflowed exp) / |ckcl| < e^-60
  --bra zero|cutoff  the same for the bra pair's cicj (uniform across the block)
e^-60 is libcint's expcutoff; a skipped term contributes < 1e-24 to any integral.
Usage: exp_prim_screen.py TREE [--barrier keep|remove] [--ket none|zero|cutoff] [--bra none|zero|cutoff]"""
import argparse
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('tree')
ap.add_argument('--barrier', default='remove', choices=['keep', 'remove'])
ap.add_argument('--ket', default='cutoff', choices=['none', 'zero', 'cutoff'])
ap.add_argument('--bra', default='cutoff', choices=['none', 'zero', 'cutoff'])
args = ap.parse_args()
assert args.barrier == 'remove' or args.ket == 'none', 'a ket skip needs the barrier removed (threads diverge)'

rel = 'gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu'
p = f'{args.tree}/{rel}'
s = subprocess.run(['git', '-C', args.tree, 'show', f'HEAD:{rel}'], check=True, capture_output=True, text=True).stdout
head = '#include "gvhf-rys/unrolled_kernels.cuh"\n'
s = s.replace(head, head + '#define PRIM_CUTOFF 8.75651076269652e-27  // e^-60, libcint expcutoff\n')
test = {'zero': '{} == 0', 'cutoff': 'fabs({}) < PRIM_CUTOFF'}
parts = re.split(r'(?=__global__ static\nvoid rys_k_\d{4}\(JKMATRIX_KERNEL_ARGS\))', s)
n = 0
for k, part in enumerate(parts):
    if not (re.match(r'__global__ static\nvoid rys_k_\d{4}\(', part) and 'nsq_per_block, 1, 0);' in part):
        continue
    old_k = '            double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;\n'
    old_loop = '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {\n                __syncthreads();\n'
    old_c = '                double cicj = cicj_cache[ijp];\n'
    assert part.count(old_k) == part.count(old_loop) == part.count(old_c) == 1
    if args.ket != 'none':
        part = part.replace(old_k, old_k + f"            if ({test[args.ket].format('ckcl')}) continue;\n")
    if args.barrier == 'remove':
        part = part.replace(old_loop, '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {\n')
    if args.bra != 'none':
        part = part.replace(old_c, old_c + f"                if ({test[args.bra].format('cicj')}) continue;\n")
    parts[k] = part
    n += 1
open(p, 'w').write(''.join(parts))
print(f'{n} kernels: barrier {args.barrier}, ket skip {args.ket}, bra skip {args.bra}')
