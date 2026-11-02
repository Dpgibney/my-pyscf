"""The broken-symmetry singlet table from bs_singlet.py's JSON lines: for each molecule and functional, the state each
run reached (<S^2>, the spin populations of the two halves, cycles), then the current tree against v1.9.0 and CPU
PySCF against the current tree: the largest differences in E, <S^2>, the gradient (own orbitals, and at the other
run's orbitals), the Hessian and the TDA excitation energies. Markdown. Runs of the same case (e.g. one with the
Hessian, a later one with TDA) are merged, the first run's values kept.
Usage: bs_table.py bs.jsonl"""
import json
import sys

runs = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get('label') not in ('base', 'current', 'cpu'):
        continue
    case = runs.setdefault((r['mol'], r['xc'], r.get('df', False)), {})
    if r['label'] in case:
        case[r['label']].setdefault('diff', {})
        for k, v in r.get('diff', {}).items():
            case[r['label']]['diff'].setdefault(k, v)
    else:
        case[r['label']] = r


def fmt(d, k):
    return f'{d[k]:.1e}' if k in d else '–'


print('| case | run | ⟨S²⟩ | spin of the halves | cycles | ΔE | Δ⟨S²⟩ | Δgradient | Δgradient, same orbitals '
      '| ΔHessian | ΔHessian, same orbitals | ΔTDA |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for (mol, xc, df), rr in runs.items():
    name = f"{mol.replace('_', ' ')} {xc.upper().replace('_', '-')}{' (DF)' if df else ''}"
    first = True
    for label, against in (('base', None), ('current', 'v1.9.0'), ('cpu', 'current')):
        if label not in rr:
            continue
        r = rr[label]
        d = r.get('diff', {})
        run = {'base': 'v1.9.0', 'current': 'current', 'cpu': 'CPU PySCF'}[label]
        cells = [fmt(d, k) for k in ('e', 's2', 'grad', 'grad_at_ref_orbitals', 'hess', 'hess_at_ref_orbitals', 'tda')]
        if against is None:
            cells = ['(reference)'] + [''] * 6
        conv = '' if r['converged'] else ' (not converged)'
        spin = f"{r['spin_halves'][0]:+.3f} / {r['spin_halves'][1]:+.3f}"
        print(f"| {name if first else ''} | {run}{' vs ' + against if against else ''} | {r['S2']:.4f} | {spin} | "
              f"{r['cycles']}{conv} | " + ' | '.join(cells) + ' |')
        first = False
