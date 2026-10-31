"""Summarize the timers of a GPU4PySCF log written at verbose 6 (scf_parts.py writes them): for each label the calls
and the summed GPU-event time, then the J (MD J-engine, labels with 'scheme=') and K (Rys) kernel time per
angular-momentum class. The GPU-event time is the measure to use: GPU4PySCF's timer reads the host clock before it
synchronizes, so a label's host wall time also contains waiting for the work queued under the previous label.
Usage: parse_gpu_log.py LOG... [--top 10]"""
import argparse
import re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--top', type=int, default=10)
args = ap.parse_args()

pat = re.compile(r'CPU time for (.+?)\s+[-\d.]+ sec, wall time\s+[-\d.]+ sec, GPU time\s+([-\d.]+) ms')
STAGES = ['vj', 'vk', 'vxc', 'nr_rks', 'eval rho on Device 0', 'eval vxc on Device 0', 'eval integration on 0',
          'eig', 'DIIS', 'SCF initialization', 'setting up grids', 'Initialize q_cond']
for log in args.logs:
    gpu, calls = Counter(), Counter()
    cycles = 0
    for line in open(log):
        m = pat.search(line)
        if not m:
            continue
        label = re.sub(r'\s*tasks ~= \d+', '', m.group(1)).strip()
        if label.startswith('cycle='):
            cycles += 1
        gpu[label] += float(m.group(2)) / 1e3
        calls[label] += 1
    print(f'== {log}: {cycles} cycles; GPU-event time (s): total, per call')
    for s in STAGES:
        if calls[s]:
            print(f'   {s:28s} {calls[s]:4d} calls {gpu[s]:8.3f} {gpu[s] / calls[s]:8.4f}')
    for name, test in (('J engine', lambda k: 'scheme=' in k), ('K (Rys)', lambda k: 'scheme=' not in k)):
        cls = {k: v for k, v in gpu.items() if re.match(r'processing \(\w+\|\w+\)', k) and test(k)}
        if not cls:
            continue
        tot = sum(cls.values())
        print(f'   {name}: {len(cls)} classes, {sum(calls[k] for k in cls)} launches, {tot:.3f} s')
        for k, v in sorted(cls.items(), key=lambda kv: -kv[1])[:args.top]:
            short = re.sub(r'processing (\(\w+\|\w+\)).*', r'\1', k)
            print(f'      {short:10s} {calls[k]:5d} launches {v:8.3f} s {100 * v / tot:5.1f}%')
