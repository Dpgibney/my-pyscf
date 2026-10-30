"""Experiment only: switch the group loops of VXCfused_dot_wv, VXCfused_dot_ao_mo and VXCfused_fxc_lowrank in
the pyscf tree between their deterministic schedule(static, 1) and a non-deterministic schedule(dynamic, 4)
(each thread then sums whichever groups it took), to measure the static schedule's load imbalance on the
mixed Zen 5 / Zen 5c cores. Rebuild after each switch, and switch back with `static` (no '// EXPERIMENT'
line may remain in the source).
Usage: fused_sched_toggle.py static|dynamic
"""
import re, sys
path = '/home/danny/Chemistry/my-pyscf/pyscf/pyscf/lib/dft/nr_numint_fused.c'
src = open(path).read()
want = sys.argv[1]
out, n, fn = [], 0, None
for line in src.split('\n'):
    m = re.match(r'void (VXCfused_\w+)\(', line)
    if m:
        fn = m.group(1)
    if fn in ('VXCfused_dot_wv', 'VXCfused_dot_ao_mo', 'VXCfused_fxc_lowrank') and \
            line.strip() in ('#pragma omp for schedule(static, 1)', '#pragma omp for schedule(dynamic, 4) // EXPERIMENT'):
        line = ('#pragma omp for schedule(static, 1)' if want == 'static'
                else '#pragma omp for schedule(dynamic, 4) // EXPERIMENT')
        n += 1
    out.append(line)
open(path, 'w').write('\n'.join(out))
print(f'{n} loops set to {want}')
