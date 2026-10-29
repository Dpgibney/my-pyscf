"""Leaf-function breakdown of qcint inside PySCF's in-memory ERI build (mol.intor('int2e', aosym='s8')),
from py-spy --native raw samples. Usage: eri_native_profile.py SYSTEM [int2e|int2e_ip1]
(run under bench/diag/run_in.py with 1 thread)."""
import collections
import os
import subprocess
import sys
import tempfile

system = sys.argv[1]
intor = sys.argv[2] if len(sys.argv) > 2 else 'int2e'
if os.environ.get('ERI_NATIVE_CHILD'):
    import time
    sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
    from molecules import BENZENE, CAFFEINE
    from pyscf import gto
    mol = gto.M(atom={'benzene': BENZENE, 'caffeine': CAFFEINE}[system], basis='def2-svp', verbose=0)
    t0 = time.perf_counter()
    if intor == 'int2e':
        mol.intor('int2e', aosym='s8')
    else:
        mol.intor(intor, comp=3, aosym='s2kl')
    print(f'{intor}: {time.perf_counter() - t0:.2f} s', file=sys.stderr)
    sys.exit(0)

out = tempfile.mktemp(suffix='.txt')
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '400', '-o', out, '--',
                sys.executable, os.path.abspath(__file__), system, intor],
               env=dict(os.environ, ERI_NATIVE_CHILD='1'))  # py-spy may exit 1 after the child ends
leaf = collections.Counter()
total = 0
for line in open(out):
    stack, _, count = line.rstrip().rpartition(' ')
    names = [f.split(' (')[0] for f in stack.split(';')]
    if not any(n.startswith(('int2e', 'GTOnr2e', 'GTO2e')) for n in names):
        continue
    total += int(count)
    leaf[names[-1]] += int(count)
print(f'{total} samples inside the integral driver')
for name, c in leaf.most_common(30):
    print(f'  {100 * c / total:5.1f}%  {name}')
os.unlink(out)
