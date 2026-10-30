"""Readable summary of a run_bench results file: per case the wall time, the time categories and the top
functions by cProfile self time (C kernels are charged to their Python wrappers).
Usage: profile_summary.py RESULTS.json [NTOP]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
ntop = int(sys.argv[2]) if len(sys.argv) > 2 else 30
m = d['meta']
print(f"{m['tag']} {m['time']}, {m['threads']} threads, shas {m['shas']}")
for r in d['runs']:
    cats = ', '.join(f'{k} {v:.2f}' for k, v in sorted(r['categories'].items(), key=lambda kv: -kv[1]) if v >= .005)
    print(f"\n== {r['case']} [{r['config']}]  wall {r['wall']:.2f} s  rss {r.get('max_rss_mb', 0):.0f} MB")
    print(f"   categories (s): {cats}")
    print('   top functions by self time (s, calls):')
    for name, tot, ncalls in r['top'][:ntop]:
        print(f'     {tot:8.3f} {ncalls:>8}  {name}')
