'''The SCF work of runs of examples/wb97mv_ts.py or wb97mv_ts_gpu.py (the directories of ts_example_variants.sh):
from the summary line that the two-level SCF writes at the end of each SCF ("two-level SCF: N Fock builds of the
model, M of the cheap one after its SCF of K cycles" on the CPU, "N evaluations of the model, M builds of the cheap
one ..." on the GPU), the model's Fock builds and the cheap model's builds and cycles of the first SCF (the start
Hessian's), the second (the search's first), the last (at the saddle point) and of all, with the steps of the search.

    python scf_work.py DIR [DIR ...]
'''
import glob
import os
import re
import sys

PAT = re.compile(r'two-level SCF: (\d+) (?:evaluations|Fock builds) of the model, (\d+) (?:builds )?of the cheap one'
                 r' after its SCF of (\d+) cycles')


def fmt(t):
    return '%d+%d+%d' % t


print('%-22s %5s %4s  %-10s %-10s %-10s %-12s' % ('run', 'steps', 'SCFs', 'first', 'second', 'last', 'all'))
print('%-22s %5s %4s  (each: Fock builds of the model + of the cheap model + cycles of its SCF)' % ('', '', ''))
for d in sys.argv[1:]:
    logs = [f for f in glob.glob(os.path.join(d, 'wb97mv_ts*.out'))]
    if not logs:
        continue
    text = open(logs[0]).read()
    scfs = [tuple(int(x) for x in m.groups()) for m in PAT.finditer(text)]
    steps = 0
    run = os.path.join(d, 'run.out')
    if os.path.exists(run):
        steps = sum(1 for line in open(run) if line.startswith('Step'))
    if not scfs:
        print('%-22s %5d %4d  (no two-level summary lines)' % (os.path.basename(d.rstrip('/')), steps, 0))
        continue
    tot = tuple(sum(x[i] for x in scfs) for i in range(3))
    print('%-22s %5d %4d  %-10s %-10s %-10s %-12s' % (
        os.path.basename(d.rstrip('/')), steps, len(scfs), fmt(scfs[0]), fmt(scfs[1]) if len(scfs) > 1 else '-',
        fmt(scfs[-1]), fmt(tot)))
