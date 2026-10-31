"""Fraction of primitive pairs in K's significant shell pairs whose Gaussian-product factor |ci cj| exp(-theta R^2)
is below e^-60 (libcint's expcutoff), per group pair; and the K time those would save, from the ncu labels."""
import sys, json, csv, re, numpy
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto
from gpu4pyscf.scf import jk
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
opt = jk._VHFOpt(mol).build()
smol = opt.sorted_mol
nbas = smol.nbas
loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
ngrp = len(smol.l_ctr_counts)
coords = smol.atom_coords()[smol._bas[:, 0]]
EXPCUT = 60.
frac = {}
for (i, j), cache in opt.bas_pair_cache.items():
    pairs = cache[0].get() if hasattr(cache[0], 'get') else numpy.asarray(cache[0])
    tot = sig = 0
    for bij in pairs:
        ish, jsh = divmod(int(bij), nbas)
        ai, aj = smol.bas_exp(ish), smol.bas_exp(jsh)
        ci = numpy.abs(smol._libcint_ctr_coeff(ish)).max(axis=1)
        cj = numpy.abs(smol._libcint_ctr_coeff(jsh)).max(axis=1)
        rr = ((coords[ish] - coords[jsh])**2).sum()
        x = numpy.outer(ai, aj) / numpy.add.outer(ai, aj) * rr - numpy.log(numpy.outer(ci, cj))
        tot += x.size
        sig += (x <= EXPCUT).sum()
    frac[(i, j)] = (len(pairs), tot, sig)
names = [f"{'spdf'[smol.bas_angular(loc[g])]}{smol.bas_nprim(loc[g])}" for g in range(ngrp)]
print('group pair      pairs  prim pairs  kept by expcutoff')
for (i, j), (n, tot, sig) in sorted(frac.items()):
    print(f'({names[i]:3s},{names[j]:3s})  {n:7d} {tot:11d} {sig:11d}  {sig / max(tot, 1):6.3f}')
# K time weighted by the surviving primitive-quartet fraction (bra fraction x ket fraction per group combination)
labels = json.load(open(sys.argv[1]))
rows = [r for r in csv.reader(open(sys.argv[2])) if r and not r[0].startswith('==')]
hdr = rows[0]; c = {h: k for k, h in enumerate(hdr)}
t_k = [float(r[c['gpu__time_duration.sum']].replace(',', '')) * 1e-9 for r in rows[2:]
       if re.sub(r'\(.*', '', r[c['Kernel Name']]).startswith('rys_k_')]
def f(a, b):
    a, b = max(a, b), min(a, b)
    n, tot, sig = frac.get((a, b), (0, 1, 1))
    return sig / max(tot, 1)
saved = 0.; total = sum(t_k)
by = {}
for lab, t in zip(labels, t_k):
    g = lab['groups']
    keep = f(g[0], g[1]) * f(g[2], g[3])
    saved += t * (1 - keep)
    key = '(' + names[g[0]] + names[g[1]] + '|' + names[g[2]] + names[g[3]] + ')'
    by.setdefault(key, [0., 0.]); by[key][0] += t; by[key][1] += t * (1 - keep)
print(f'K kernel time {total:.3f} s; if cost scales with primitive quartets, expcutoff screening saves {saved:.3f} s ({100 * saved / total:.0f}%)')
for k, (t, s) in sorted(by.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f'   {k:22s} {t:6.3f} s, saves {s:6.3f} s ({100 * s / t:3.0f}%)')
bra = sum(t * (1 - f(lab['groups'][0], lab['groups'][1])) for lab, t in zip(labels, t_k))
print(f'bra-side screening alone saves {bra:.3f} s ({100 * bra / total:.0f}%)')
for name in ('rys_k_0000', 'rys_k_1000'):
    ts = [(lab, t) for lab, t, r in zip(labels, t_k, [r for r in rows[2:] if re.sub(r'\(.*', '', r[c['Kernel Name']]).startswith('rys_k_')])
          if re.sub(r'\(.*', '', r[c['Kernel Name']]) == name]
    tt = sum(t for _, t in ts); bs = sum(t * (1 - f(l['groups'][0], l['groups'][1])) for l, t in ts)
    print(f'  {name}: {tt:.3f} s, bra-side saves {bs:.3f} s ({100 * bs / max(tt, 1e-30):.0f}%)')
