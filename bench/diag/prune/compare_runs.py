'''Two sets of runs of a TS example, old against new (OLD/*/run.out against NEW/*/run.out, as the scripts
ts_example_variants.sh leave them): the steps of each search, E and G at the saddle point, the largest change of a
frequency at the 0.1 cm^-1 they are printed with. Usage: compare_runs.py OLD NEW'''
import glob, os, re, sys

KCAL = 627.509474


def parse(path):
    s = open(path).read()
    r = {'steps': len(re.findall(r'^Step', s, re.M))}
    m = re.search(r'^E = (\S+) Eh', s, re.M)
    r['E'] = float(m.group(1)) if m else None
    m = re.search(r'^harmonic frequencies \(cm\^-1\): (.*)$', s, re.M)
    r['freq'] = [float(x) for x in m.group(1).split()] if m else []
    m = re.search(r': G = (\S+) Eh', s)
    r['G'] = float(m.group(1)) if m else None
    m = re.search(r'^largest gradient component (\S+)', s, re.M)
    r['grad'] = m.group(1) if m else '-'
    return r


old_dir, new_dir = sys.argv[1:3]
print('%-14s %-11s %-12s %-26s %-12s %s' % ('run', 'steps o/n', 'dE (Eh)', 'max |d freq| (cm^-1, 0.1)',
                                           'dG (Eh)', 'dG (kcal/mol)'))
for d in sorted(glob.glob(os.path.join(old_dir, '*', 'run.out'))):
    name = os.path.basename(os.path.dirname(d))
    o = parse(d)
    p = os.path.join(new_dir, name, 'run.out')
    if not os.path.exists(p):
        print('%-14s no new run' % name)
        continue
    n = parse(p)
    if o['G'] is None or n['G'] is None or len(o['freq']) != len(n['freq']):
        print('%-14s incomplete: old %s new %s' % (name, o, n))
        continue
    df = max(abs(a - b) for a, b in zip(o['freq'], n['freq']))
    print('%-14s %2d / %-6d %+.2e    %-26.1f %+.2e    %+.2e' % (
        name, o['steps'], n['steps'], n['E'] - o['E'], df, n['G'] - o['G'], (n['G'] - o['G']) * KCAL))
