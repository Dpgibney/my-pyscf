"""What the density loop of GPU4PySCF's unrolled J+K kernels compiles to, as written and with the density loads
through __ldg (node1; compile only, nothing runs on the GPU). Copies the chosen kernels of unrolled_rys_jk.cu with the
file's includes into OUT_DIR in three variants: as written (orig), with __restrict__ on the loop's dm, vj and vk
pointers (restrict), and with every density load `= dm[...]` as `= __ldg(dm+...)` (ldg). Each is compiled as the
build does for sm_75 (relocatable device code, ptxas -v) and disassembled. Per kernel and variant it prints the
registers and spills, the counts of FP64 instructions, FP64 atomic adds and 64-bit global loads (the FP64 counts
show whether the arithmetic changed), and the order of loads (L), FP64 instructions (F) and atomics (A) from the
first density load to the last atomic, with the number of places where a load follows an atomic: each is a load the
compiler kept behind the atomics before it, so its latency is exposed again.
Usage: ldg_sass_check.py [--kernels rys_k_1000,rys_k_2110] [--out-dir scratch/ldg_test] [--lib LIB_DIR]
                         [--cuda ~/opt/cuda-12.9.1]"""
import argparse
import os
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--kernels', default='rys_k_1000,rys_k_2110')
ap.add_argument('--out-dir', default='scratch/ldg_test')
ap.add_argument('--lib', default=os.path.expanduser('~/Chemistry/my-pyscf/gpu4pyscf/gpu4pyscf/lib'))
ap.add_argument('--cuda', default=os.path.expanduser('~/opt/cuda-12.9.1'))
args = ap.parse_args()

lines = open(os.path.join(args.lib, 'gvhf-rys', 'unrolled_rys_jk.cu')).read().split('\n')
head = '\n'.join(lines[:8])   # the includes


def kernel(name):
    '''The kernel's text, from its "__global__ static" line to the next kernel, made non-static'''
    a = next(n for n, x in enumerate(lines) if x.startswith(f'void {name}('))
    b = next((n for n in range(a + 1, len(lines)) if lines[n].startswith(('void ', '__global__'))), len(lines))
    body = lines[a - 1:b]
    body[0] = body[0].replace('static', '')
    return '\n'.join(body)


def demangled(sym):
    '''The function name of a mangled _Z<length><name>... symbol'''
    m = re.match(r'_Z(\d+)', sym)
    return sym[m.end():m.end() + int(m.group(1))] if m else sym


names = args.kernels.split(',')
src = head + '\n' + '\n'.join(kernel(k) for k in names) + '\n'
variants = {
    'orig': src,
    'restrict': (src.replace('double *dm = jk.dm + i_dm * nao2;', 'const double *__restrict__ dm = jk.dm + i_dm * nao2;')
                 .replace('double *vk = jk.vk + i_dm * nao2;', 'double *__restrict__ vk = jk.vk + i_dm * nao2;')
                 .replace('double *vj = jk.vj + i_dm * nao2;', 'double *__restrict__ vj = jk.vj + i_dm * nao2;')),
    'ldg': re.sub(r'= dm\[([^;]*)\];', r'= __ldg(dm+\1);', src),
}
os.makedirs(args.out_dir, exist_ok=True)
nvcc = os.path.join(args.cuda, 'bin', 'nvcc')
cuobjdump = os.path.join(args.cuda, 'bin', 'cuobjdump')
for v, text in variants.items():
    cu = os.path.join(args.out_dir, f'k_{v}.cu')
    obj = cu[:-3] + '.o'
    open(cu, 'w').write(text)
    p = subprocess.run([nvcc, '-lineinfo', '-O2', '-std=c++17', '-arch=sm_75', '-dc', '--ptxas-options=-v',
                        f'-I{args.lib}', '-o', obj, cu], capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-2000:])
        raise SystemExit(f'{v}: nvcc failed')
    ptxas = {}
    cur = None
    for x in p.stderr.split('\n'):
        m = re.search(r"Compiling entry function '(_Z\w+)'", x)
        if m:
            cur = demangled(m.group(1))
        m = re.search(r'(\d+) bytes spill stores', x)
        if m and cur:
            ptxas[cur] = [None, int(m.group(1))]
        m = re.search(r'Used (\d+) registers', x)
        if m and cur:
            ptxas[cur][0] = int(m.group(1))
    sass = subprocess.run([cuobjdump, '-sass', obj], capture_output=True, text=True).stdout
    for f in re.split(r'\n\s+Function : ', sass)[1:]:
        name = demangled(f.split('\n')[0].strip())
        if not name.startswith('rys_'):
            continue
        seq = [(o[1], o[2]) for o in re.findall(
            r'/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9]*)(\.[A-Z0-9_.]+)?', f)]
        count = {}
        for o, mod in seq:
            k = (o + mod) if o in ('ATOMG', 'LDG') else o
            if o in ('DFMA', 'DMUL', 'DADD') or k.startswith(('ATOMG.E.ADD.F64', 'LDG.E.64')):
                count[k] = count.get(k, 0) + 1
        atoms = [n for n, (o, mod) in enumerate(seq) if o == 'ATOMG' and 'F64' in mod]
        s = atoms[0]
        while s > 0 and seq[s - 1][0] not in ('BRA', 'BAR', 'BSYNC'):
            s -= 1
        runs = []
        for o, mod in seq[s:atoms[-1] + 1]:
            k = {'LDG': 'L', 'ATOMG': 'A', 'DFMA': 'F', 'DMUL': 'F', 'DADD': 'F'}.get(o)
            if k and runs and runs[-1][0] == k:
                runs[-1][1] += 1
            elif k:
                runs.append([k, 1])
        exposed = sum(1 for a, b in zip(runs, runs[1:]) if a[0] == 'A' and b[0] == 'L')
        regs, spill = ptxas.get(name, (None, None))
        print(f'{v:8s} {name}: {regs} registers, {spill} bytes spilled; ' +
              ', '.join(f'{c} {k}' for k, c in sorted(count.items())))
        print(f'         loop order: {exposed} loads behind atomics; ' + ' '.join(f'{k}{n}' for k, n in runs[:48]) +
              (' ...' if len(runs) > 48 else ''))
