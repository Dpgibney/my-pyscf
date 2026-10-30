"""Native profile of one bench case at 1 thread: self (leaf) and inclusive time per function.
py-spy --native samples the master thread only (OpenMP workers are not Python threads), so the case runs
with OMP_NUM_THREADS=1 and the samples cover all of its CPU time. Includes the case's setup (SCF for
post-SCF cases) and worker.warmup().
Usage (in a config's environment): native_profile.py CASE [NTOP]
"""
import collections
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('NATIVE_PROFILE_CHILD'):
    sys.path.append(os.path.join(here, '..'))
    import cases
    import worker
    worker.warmup()
    setup, _ = cases.CASES[sys.argv[1]]
    setup(False)()
    sys.exit(0)


def frame_name(frame):
    '''"func (path:line)" -> "func (file)"'''
    func, _, where = frame.partition(' (')
    path = where.rstrip(')').rsplit(':', 1)[0]
    return f'{func} ({os.path.basename(path)})' if path else func


case = sys.argv[1]
ntop = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out = tempfile.mktemp(suffix='.txt')
env = dict(os.environ, NATIVE_PROFILE_CHILD='1', OMP_NUM_THREADS='1')
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
for attempt in range(3):  # py-spy sometimes fails to attach to the process it starts
    p = subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '100', '-o', out, '--',
                        sys.executable, os.path.abspath(__file__), case], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        break
    print(f'py-spy failed (attempt {attempt + 1}): {p.stderr.strip()[-500:]}', file=sys.stderr)
else:
    sys.exit(1)
leaf, incl = collections.Counter(), collections.Counter()
total = 0
for line in open(out):
    stack, _, count = line.rstrip().rpartition(' ')
    count = int(count)
    frames = [frame_name(f) for f in stack.split(';')]
    total += count
    leaf[frames[-1]] += count
    for f in set(frames):
        incl[f] += count
os.remove(out)
print(f'{case}: {total} samples (100 Hz) at 1 thread')
print('self (leaf) time:')
for name, n in leaf.most_common(ntop):
    print(f'  {100 * n / total:5.1f}%  {name}')
print('inclusive time:')
for name, n in incl.most_common(ntop):
    print(f'  {100 * n / total:5.1f}%  {name}')
