"""Model of part 8's sums per Cartesian tuple applied to the Hessian's second-derivative kernels (RYS_per_atom_jk_ip2
type12 and type3), per angular-momentum class (no GPU needed).

What GPU4PySCF's kernels do per output (ij|kl) and root: type12 builds, for each direction, the first-derivative
factors of i, j, k and l and the second-derivative factors ij, ii, jj, kl, kk, ll, and adds 42 terms; type3 builds
the first-derivative factors and the mixed factors ik, il, jk, jl, and adds 36 terms. That is about 180 and 160 FP64
operations per output (measured, ncu_source_regions.py).

With sums per tuple:
  - the same-direction terms (xx, yy, zz) are F(t_x) S_x(t_x), S_x(t) = sum over the outputs with x tuple t of
    dd Iy Iz, as in the gradient;
  - a mixed term such as f2x f1y Iz dd is grouped by the x tuple: T(t_x) = sum of f1y(t_y) Iz dd. The four factors
    of the second direction (type12: f1, f2 of the bra, f1', f2' of the ket; type3: fi, fj, fk, fl) give four such
    sums per direction pair, and each tuple then multiplies them by its factors (8 FMAs per pair).
  Per output and root: three passes (by x, by y, by z tuples) with 11 + 7 + 2 FP64 operations, 20 in all, when
  the factors of the other directions come from per-tuple tables ('table'); 44 when each output recomputes them
  from its g values (2 operations each, 'recompute', no tables in shared memory).
  Per tuple and root: its factors (type12 about 36 FP64, type3 about 32) in each direction, and its contributions
  (type12 22 + 14 + 6, type3 20 + 12 + 4 FMAs over the three passes).
So FP64 per root = 20 n_out + 150 n_t (type12) or 20 n_out + 132 n_t (type3), with n_out the outputs of a quartet
and n_t = (li+1)(lj+1)(lk+1)(ll+1) the tuples of a direction.

With --profile SRC.csv --order LABELS.json --classes ... (as for ncu_source_regions.py) and --times PROFILE.txt
(hess_ejk2_profile.txt), it also predicts each profiled class's time, assuming the output phase's time scales with its FP64 instructions (it is
FP64-bound) and the rest stays: t_new = t (1 - s + s r), with s the output phase's share of the stall samples and
r = FP64_new / FP64_now.
Usage: ip2_tuple_model.py [--profile SRC.csv --order LABELS.json --classes '(dp|ps),...' --times PROFILE.txt]"""
import argparse
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

L = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
PER_OUT = {'table': 20, 'recompute': 44}
PER_TUPLE = {'t12': 150, 't3': 132}
NOW = {'t12': 181, 't3': 159}   # measured for (dp|ps); per-class values come from --profile


def nf(l):
    return (l + 1) * (l + 2) // 2


def counts(cls):
    li, lj, lk, ll = (L[c] for c in cls.strip('()').replace('|', ''))
    n_out = nf(li) * nf(lj) * nf(lk) * nf(ll)
    n_t = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1)
    return n_out, n_t


def fp64_new(cls, kind, variant='table'):
    n_out, n_t = counts(cls)
    return PER_OUT[variant] * n_out + PER_TUPLE[kind] * n_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default=None)
    ap.add_argument('--order', default=None)
    ap.add_argument('--classes', default=None)
    ap.add_argument('--times', default=None)
    args = ap.parse_args()
    classes = ['(ss|ss)', '(ps|ss)', '(ps|ps)', '(pp|ss)', '(pp|ps)', '(pp|pp)', '(ds|ss)', '(ds|ps)', '(ds|pp)',
               '(dp|ss)', '(dp|ps)', '(dp|pp)', '(ds|ds)', '(dp|ds)', '(dp|dp)', '(dd|ss)', '(dd|ps)', '(dd|pp)',
               '(dd|ds)', '(dd|dp)', '(dd|dd)']
    print('class    outputs tuples   FP64 per root now -> per tuple (t12)   now -> per tuple (t3)')
    for c in classes:
        n_out, n_t = counts(c)
        a12, a3 = NOW['t12'] * n_out, NOW['t3'] * n_out
        b12, b3 = fp64_new(c, 't12'), fp64_new(c, 't3')   # with tables
        print(f'{c} {n_out:7d} {n_t:6d}   {a12:8d} -> {b12:7d} ({a12 / b12:4.1f}x)    '
              f'{a3:8d} -> {b3:7d} ({a3 / b3:4.1f}x)')
    if not (args.profile and args.times):
        return
    import ncu_source_regions as nsr
    prof = nsr.regions_by_launch(args.profile, nsr.launch_order(args.order, args.classes))
    times = {}
    for line in open(args.times):
        m = re.match(r'(\(\w\w\|\w\w\)) (t12|t3) .*?(\d+\.\d+) ->', line)
        if m:
            times[m.group(1), m.group(2)] = float(m.group(3))
    total = sum(times.values())
    print(f'\nprofiled launches (time from {os.path.basename(args.times)}, all classes {total:.2f} s):')
    print('class    kind  time    output share  FP64/out/root   FP64 ratio (table, recompute)   predicted time')
    t_old, t_new = 0., {'table': 0., 'recompute': 0.}
    for (c, kind), p in sorted(prof.items(), key=lambda kv: -times.get(kv[0], 0)):
        t = times.get((c, kind))
        if t is None or not p.get('iters'):
            continue
        n_out, _ = counts(c)
        now = p['fp64_out'] / p['iters']
        s = p['stall_out'] / p['stall']
        t_old += t
        r, tn = {}, {}
        for v in t_new:
            r[v] = fp64_new(c, kind, v) / (now * n_out)
            tn[v] = t * (1 - s + s * r[v])
            t_new[v] += tn[v]
        print(f'{c} {kind:4s} {t:6.3f} s  {100 * s:5.1f}%       {now:6.1f}         {1 / r["table"]:4.1f}x, '
              f'{1 / r["recompute"]:4.1f}x                {tn["table"]:6.3f}, {tn["recompute"]:6.3f} s')
    print(f'profiled generic launches: {t_old:.2f} s -> {t_new["table"]:.2f} s ({t_old / t_new["table"]:.2f}x, table) '
          f'or {t_new["recompute"]:.2f} s ({t_old / t_new["recompute"]:.2f}x, recompute), of {total:.2f} s in all')


if __name__ == '__main__':
    main()
