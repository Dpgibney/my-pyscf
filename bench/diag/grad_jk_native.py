"""Leaf-function breakdown of the gradient J/K build (int2e_ip1, s2kl, 3 components) from py-spy
--native raw samples. Usage: grad_jk_native.py SYSTEM  (run under bench/diag/run_in.py with 1 thread)."""
import collections
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
system = sys.argv[1]
if os.environ.get('JK_NATIVE_CHILD'):
    import time
    sys.path.append(os.path.join(here, '..'))
    from molecules import BENZENE, CAFFEINE
    from pyscf import gto, scf
    from pyscf.grad import rhf as rhf_grad
    atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[system]
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    dm = scf.RHF(mol).get_init_guess(key='minao')
    t0 = time.perf_counter()
    rhf_grad.get_jk(mol, dm)
    print(f'grad J/K build {time.perf_counter() - t0:.2f} s', file=sys.stderr)
    sys.exit(0)

out = tempfile.mktemp(suffix='.txt')
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '250', '-o', out, '--',
                sys.executable, os.path.abspath(__file__), system],
               env=dict(os.environ, JK_NATIVE_CHILD='1'))  # py-spy may exit 1 after the child ends
leaf = collections.Counter()
total = 0
for line in open(out):
    stack, _, count = line.rstrip().rpartition(' ')
    names = [f.split(' (')[0] for f in stack.split(';')]
    if 'CVHFnr_direct_drv' not in names:
        continue
    total += int(count)
    leaf[names[-1]] += int(count)
print(f'{total} samples inside CVHFnr_direct_drv')
for name, c in leaf.most_common(20):
    print(f'  {100 * c / total:5.1f}%  {name}')
os.unlink(out)
