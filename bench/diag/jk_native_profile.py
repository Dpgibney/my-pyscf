"""Leaf-function breakdown of one J/K build, from py-spy --native raw samples.
Usage: jk_native_profile.py SYSTEM incore|direct NDM [jk|j|k]  (run under bench/diag/run_in.py, ideally
with 1 thread: OpenMP workers are not Python threads, so py-spy only samples the master thread)."""
import collections
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
system, kind, ndm = sys.argv[1:4]
what = sys.argv[4] if len(sys.argv) > 4 else 'jk'
if os.environ.get('JK_NATIVE_CHILD'):
    sys.argv = [sys.argv[0], system, kind, ndm]
    sys.path.append(here)
    import time
    import numpy as np
    sys.path.append(os.path.join(here, '..'))
    from molecules import BENZENE, CAFFEINE, n_alkane
    from pyscf import gto, lib, scf
    from pyscf.scf import _vhf
    atom, basis = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
                   'c20': (n_alkane(20), 'def2-svp')}[system]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    vhfopt = mf.init_direct_scf()
    dm0 = mf.get_init_guess(key='minao')
    n = int(ndm)
    if n == 1:
        dms, hermi = dm0, 1
    else:
        mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
        nocc = mol.nelectron // 2
        x = np.random.default_rng(7).standard_normal((n, nocc, mol.nao - nocc))
        dms, hermi = np.einsum('pi,nia,qa->npq', mo_c[:, :nocc], x, mo_c[:, nocc:]), 0
    wj, wk = 'j' in what, 'k' in what
    if kind == 'incore':
        eri = mol.intor('int2e', aosym='s8')
        t0 = time.perf_counter()
        _vhf.incore(eri, dms, hermi, wj, wk)
    else:
        prescreen = ('CVHFnrs8_prescreen' if wj and wk else
                     'CVHFnrs8_vj_prescreen' if wj else 'CVHFnrs8_vk_prescreen')
        t0 = time.perf_counter()
        with lib.temporary_env(vhfopt, prescreen=prescreen):
            scf.hf.get_jk(mol, dms, hermi, vhfopt, wj, wk)
    print(f'J/K build {time.perf_counter() - t0:.2f} s', file=sys.stderr)
    sys.exit(0)

out = tempfile.mktemp(suffix='.txt')
env = dict(os.environ, JK_NATIVE_CHILD='1')
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '250', '-o', out, '--',
                sys.executable, os.path.abspath(__file__), system, kind, ndm, what], env=env)  # py-spy may exit 1 after the child ends
leaf = collections.Counter()
under = collections.Counter()
total = 0
for line in open(out):
    stack, _, count = line.rstrip().rpartition(' ')
    count = int(count)
    frames = stack.split(';')
    names = [f.split(' (')[0] for f in frames]
    # only the J/K build: skip setup samples (integral screening setup, guess, eri build)
    if not any(n in ('CVHFnr_direct_drv', 'CVHFnrs8_incore_drv', 'CVHFnr_direct_jk_batch') for n in names):
        continue
    total += count
    leaf[names[-1]] += count
    for n in set(names):
        under[n] += count
print(f'{total} samples inside the J/K driver')
print('top leaf functions:')
for name, c in leaf.most_common(25):
    print(f'  {100 * c / total:5.1f}%  {name}')
print('inclusive:')
for name in ('CVHFdot_nrs8', 'int2e_sph', 'CVHFnrs8_prescreen', 'CVHFnrs8_vj_prescreen',
             'CVHFnrs8_prescreen_block', 'JKOperator_write_back'):
    if under[name]:
        print(f'  {100 * under[name] / total:5.1f}%  {name}')
os.unlink(out)
