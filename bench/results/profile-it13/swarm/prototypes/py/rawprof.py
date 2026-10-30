"""py-spy --native raw profile of one bench case at 1 thread; keeps the raw file (in this dir)."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('RAWPROF_CHILD'):
    sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
    import cases, worker
    worker.warmup()
    setup, _ = cases.CASES[sys.argv[1]]
    setup(False)()
    sys.exit(0)
case = sys.argv[1]
out = os.path.join(here, f'raw_{case}.txt')
env = dict(os.environ, RAWPROF_CHILD='1', OMP_NUM_THREADS='1')
pyspy = os.path.join(os.path.dirname(sys.executable), 'py-spy')
for attempt in range(3):
    p = subprocess.run([pyspy, 'record', '--native', '--format', 'raw', '--rate', '100', '-o', out, '--',
                        sys.executable, os.path.abspath(__file__), case], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        break
    print('py-spy failed', p.stderr[-300:], file=sys.stderr)
print('done', out)
