"""Leaf functions of semi-direct J/K calls that read stored integrals (py-spy --native, attached after
the call that stores them), or with --time their wall times at the current thread count.
Usage: semidirect_profile.py SYSTEM [NCALLS] [--time]   (SYSTEM: c20, c20-tz, caffeine; with 1 thread
for the profile: OpenMP workers are not Python threads, so py-spy samples only the master thread)."""
import collections
import os
import subprocess
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
system = sys.argv[1]
ncalls = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
if os.environ.get('SD_PROFILE_CHILD') or '--time' in sys.argv:
    sys.path.append(os.path.join(here, '..'))
    from molecules import CAFFEINE, n_alkane
    from pyscf import dft, gto, lib, scf
    from pyscf.scf import _vhf
    atom, basis = {'caffeine': (CAFFEINE, 'def2-svp'), 'c20': (n_alkane(20), 'def2-svp'),
                   'c20-tz': (n_alkane(20), 'def2-tzvp')}[system]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='B3LYP')
    vhfopt = mf.init_direct_scf()
    vhfopt.semi_direct_eri = _vhf.semi_direct_eri(vhfopt, mol._atm, mol._bas, mol._env,
                                                  float(os.environ.get('SD_MEMORY', 15000)))
    dm = mf.get_init_guess(key='minao')
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dm, 1, vhfopt)
    t1 = time.perf_counter()
    print(f'store {t1 - t0:.2f} s', file=sys.stderr, flush=True)
    print('READY', flush=True)
    time.sleep(1)
    for i in range(ncalls):
        t0 = time.perf_counter()
        scf.hf.get_jk(mol, dm, 1, vhfopt)
        print(f'read {time.perf_counter() - t0:.2f} s', file=sys.stderr, flush=True)
    sys.exit(0)

out = tempfile.mktemp(suffix='.txt')
env = dict(os.environ, SD_PROFILE_CHILD='1')
child = subprocess.Popen([sys.executable, os.path.abspath(__file__), system, str(ncalls)], env=env,
                         stdout=subprocess.PIPE, text=True)
assert child.stdout.readline().strip() == 'READY'
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '250', '-o', out,
                '--pid', str(child.pid)])
child.wait()
leaf = collections.Counter()
total = 0
for line in open(out):
    stack, _, count = line.rstrip().rpartition(' ')
    count = int(count)
    frames = stack.split(';')
    if not any('CVHFnr_direct_jk_batch' in f for f in frames):
        continue
    leaf[frames[-1].split(' (')[0]] += count
    total += count
print(f'{total} samples in CVHFnr_direct_jk_batch')
for name, n in leaf.most_common(15):
    print(f'  {100 * n / total:5.1f}%  {name}')
