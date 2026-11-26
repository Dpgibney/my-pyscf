'''The variants of prune_check.py (its --json file): frequencies, G and the Hessian's time by kind of work, against
ref (the reference SCF from the initial guess) and against rho:0 (an SCF from ref's density with nothing pruned: the
path of the rho variants, whose own convergence point differs from ref's by 0.002 cm-1 in the frequencies).
XC = the XC parts on the SCF's grid (energy term and Fock derivative) and on the response grid; VV10 =
_get_enlc_deriv2 + _get_vnlc_deriv1 + get_vnlc_resp_mo; kern = the two pair kernels the VV10 cutoff applies to.
The XC time on the SCF's grid is also given relative to parts that no pruning touches (the two-electron terms and the
CPHF solver), which takes out most of the drift of the laptop between Hessians.
Usage: summarize.py RAD.jsonl'''
import json, sys
import numpy

KCAL = 627.509474
recs = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    recs[r['variant']] = r
XC_GRID = ('xc _get_exc_deriv2_grid_response', 'xc uks._get_vxc_diag', 'xc uks._get_vxc_deriv2_dm',
           'xc uks._get_vxc_deriv1')
XC_RESP = ('xc uks_fxc_mo (response grid)',)
VV = ('vv10 _get_enlc_deriv2', 'vv10 _get_vnlc_deriv1', 'vv10 get_vnlc_resp_mo')
CTRL = ('hess_jk_energy', '_partial_hess_ejk', 'cphf ucphf.solve_residual')


def against(base):
    r0 = recs[base]
    f0 = numpy.array(r0['freq'])
    h0 = numpy.array(r0['hess'])
    print(f'against {base}: dE (Eh), the largest change of a frequency (cm-1, and the mode), the lowest three, the '
          f'largest of a Hessian element (Eh/Bohr^2), dG (kcal/mol)')
    for var, r in recs.items():
        f = numpy.array(r['freq'])
        d = f - f0
        i = int(numpy.argmax(abs(d)))
        print('  %-12s dE %+.2e  max|df| %.4f (%6.1f)  lowest %+.4f %+.4f %+.4f  max|dH| %.1e  dG %+.6f' % (
            var, r['e_tot'] - r0['e_tot'], abs(d).max(), f0[i], d[0], d[1], d[2],
            abs(numpy.array(r['hess']) - h0).max(), (r['G'] - r0['G']) * KCAL))


def soft(base, below=200.):
    '''the imaginary modes (negative) and those below `below` cm-1: ref's values and each variant's change'''
    f0 = numpy.array(recs[base]['freq'])
    sel = numpy.nonzero(f0 < below)[0]
    print(f'the imaginary and soft modes (< {below:.0f} cm-1) of {base}: ' + ' '.join('%.2f' % f0[i] for i in sel) +
          '; the change of each:')
    for var, r in recs.items():
        d = numpy.array(r['freq'])[sel] - f0[sel]
        print('  %-12s ' % var + ' '.join('%+.4f' % x for x in d))


print('lowest frequencies of ref (cm-1):', ' '.join('%.2f' % x for x in recs['ref']['freq'][:8]))
against('ref')
if 'rho:0' in recs:
    against('rho:0')
soft('ref')
print('time (s): the Hessian; XC on the SCF grid, on the response grid; VV10; the cut kernels; the untouched parts; '
      'XC on the SCF grid / untouched, relative to rho:0 (or ref); points; VV10 block pairs kept')
base = recs.get('rho:0', recs['ref'])
b = sum(base['parts'].get(k, 0) for k in XC_GRID) / sum(base['parts'].get(k, 0) for k in CTRL)
for var, r in recs.items():
    p = r['parts']
    xg = sum(p.get(k, 0) for k in XC_GRID)
    xr = sum(p.get(k, 0) for k in XC_RESP)
    vv = sum(p.get(k, 0) for k in VV)
    ct = sum(p.get(k, 0) for k in CTRL)
    ker = sum(v[1] for v in r['kernels'].values())
    kept = [v for v in r['kernels'].values() if v[3]]
    pairs = '%.3f' % (sum(v[2] for v in kept) / sum(v[3] for v in kept)) if kept else '-'
    print('  %-12s %7.1f  XC %6.1f %6.1f  VV10 %6.1f  kern %5.1f  untouched %6.1f  %.3f  XC %d VV10 %d  pairs %s' % (
        var, r['hessian_s'], xg, xr, vv, ker, ct, xg / ct / b, r['xc_points'], r['nlc_points'], pairs))
