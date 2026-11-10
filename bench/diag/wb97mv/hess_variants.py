'''wB97M-V Hessian: where the time goes by phase, and the error and time of approximations of the CPHF response.
The SCF, the partial Hessian and H1 are computed once (exact); each variant solves the CPHF again and its Hessian
and harmonic frequencies are compared with the reference (exact response, tight CPHF).

Usage: hess_variants.py XYZFILE|NAME [--basis def2-tzvpd] [--grid 99,590] [--nlc 50,194] [--charge 0]
                        [--tol 1e-5] [--ref-tol 1e-7] [--variants a,b,...] [--save PREFIX]
Variants (combine with '+'):
  exact        the reference path at --tol
  stock        the stock Krylov solver (cphf.solve), exact response
  g2=L         XC response on mf.second_grids of level L (or sg1)
  nonlc        no VV10 in the response (gen_response(with_nlc=False))
  vg=L         the VV10 of the response on an NLC grid of level L (or sg1); the partial Hessian and H1 keep mf.nlcgrids
  sgxL         hobj.sgx_response on SGX grid level L (direct J + SGX-K of both operators)
  lrsgxL       exact J and full-range K, the long-range K alone from SGX on level L
'''
import argparse
import contextlib
import sys
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.lib import logger
from pyscf.dft import numint, gen_grid
from pyscf.scf import hf, _response_functions
from pyscf.hessian import rhf as rhf_hess, rks as rks_hess, thermo

ap = argparse.ArgumentParser()
ap.add_argument('mol')
ap.add_argument('--basis', default='def2-tzvpd')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlc', default='50,194')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--tol', type=float, default=1e-5)
ap.add_argument('--ref-tol', type=float, default=1e-7)
ap.add_argument('--variants', default='exact,stock,g2=1,g2=sg1,g2=2,nonlc,sgx0,sgx1,lrsgx0,lrsgx1,g2=1+nonlc+sgx0,g2=1+nonlc+lrsgx0')
ap.add_argument('--save', default=None)
ap.add_argument('--mo', default=None, help='npz with mo_coeff, mo_occ, mo_energy of a converged SCF (skips the SCF)')
ap.add_argument('--skip-grad', action='store_true')
ap.add_argument('--hess-nlc', default=None, help='after the SCF, the VV10 grid of every Hessian term: a level or sg1')
ap.add_argument('--compare-ref', default=None, help='.npy Hessian of another run to compare the reference with')
ap.add_argument('--atom-name', default=None, help='molecules.py geometry name (caffeine, benzene, cNhM) instead of an xyz')
args = ap.parse_args()
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)

if args.atom_name:
    from molecules import CAFFEINE, BENZENE, n_alkane
    nm = args.atom_name
    atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}.get(nm) or n_alkane(int(nm[1:nm.index('h')]))
else:
    atom = args.mol
mol = gto.M(atom=atom, basis=args.basis, charge=args.charge, verbose=0)


def setgrid(g, spec):
    if ',' in spec:
        g.atom_grid = tuple(int(x) for x in spec.split(','))
    else:
        g.level = int(spec)


T = {}
STACK = []


@contextlib.contextmanager
def timers(specs):
    '''specs: (owner, name, key); wall time of each call credited to key unless a call of key is already running'''
    saved = []
    for owner, name, key in specs:
        if not hasattr(owner, name):
            continue
        fn = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        raw = fn

        def make(raw, key):
            def w(*a, **k):
                kk = key(a, k) if callable(key) else key
                if any(x[:12] == kk[:12] for x in STACK):
                    return raw(*a, **k)
                STACK.append(kk)
                t = time.perf_counter()
                try:
                    return raw(*a, **k)
                finally:
                    T[kk] = T.get(kk, 0.) + time.perf_counter() - t
                    STACK.pop()
            return w
        saved.append((owner, name, fn))
        setattr(owner, name, make(raw, key))
    try:
        yield
    finally:
        for owner, name, fn in saved:
            setattr(owner, name, fn)


def jkkey(a, k):
    om = k.get('omega')
    wj, wk = k.get('with_j', True), k.get('with_k', True)
    return 'J/K: ' + ('LR-K' if om else ('J+K' if wj and wk else ('J' if wj else 'K')))


from pyscf.sgx import sgx as sgx_mod  # noqa: E402
SPECS = [
    (hf.RHF, 'get_jk', jkkey),
    (sgx_mod._SGXHF, 'get_jk', lambda a, k: 'SGX obj: ' + ('LR-K' if k.get('omega') else 'J(+K)')),
    (sgx_mod.SGX, 'get_k_mo', lambda a, k: 'SGX-K from factors' + (' LR' if (k.get('omega') or (len(a) > 5 and a[5])) else ' full')),
    (rks_hess, 'get_vnlc_resp_mo', 'VV10 response'),
    (rks_hess, 'get_vnlc_resp1', 'VV10 response'),
    (numint._FusedXC, 'rks_fxc_mo', 'XC response'),
    (numint.NumInt, 'nr_rks_fxc', 'XC response'),
    (numint.NumInt, 'cache_xc_kernel', 'XC kernel setup (cache_xc_kernel)'),
]

mf = dft.RKS(mol, xc=args.xc)
setgrid(mf.grids, args.grid)
setgrid(mf.nlcgrids, args.nlc)
mf.conv_tol = 1e-10
t = time.perf_counter()
if args.mo:
    z = numpy.load(args.mo)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = z['mo_coeff'], z['mo_occ'], z['mo_energy']
    mf.e_tot = float(z['e_tot'])
    mf.converged = True
    mf.cycles = 0
    mf.grids.build(with_non0tab=True)
    mf.nlcgrids.build(with_non0tab=True)
else:
    mf.kernel()
t_scf = time.perf_counter() - t
mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
nocc = int((mo_occ > 0).sum())
nmo = mo_coeff.shape[1]
atmlst = list(range(mol.natm))
print(f'{args.mol} {args.basis} {args.xc}: natm {mol.natm} nao {mol.nao} nocc {nocc}; SCF {t_scf:.1f} s, {mf.cycles} cycles, '
      f'E {mf.e_tot:.10f}, conv {mf.converged}; XC grid {mf.grids.weights.size}, VV10 grid {mf.nlcgrids.weights.size}; '
      f'max_memory {mf.max_memory}', flush=True)

if args.hess_nlc is not None:
    if args.hess_nlc == 'sg1':
        g = gen_grid.sg1_grids(mol)
    else:
        g = gen_grid.Grids(mol)
        g.level = int(args.hess_nlc)
    g.build(with_non0tab=True)
    print(f'VV10 grid of the Hessian terms: {args.hess_nlc}, {g.weights.size} points (SCF: {mf.nlcgrids.weights.size})')
    mf.nlcgrids = g

# ---- the gradient norm, to know how far from a stationary point the geometry is
if not args.skip_grad:
    g = mf.Gradients().kernel()
    print(f'gradient: max {abs(g).max():.2e}, rms {numpy.sqrt((g**2).mean()):.2e} Eh/bohr', flush=True)

# ---- partial Hessian and H1 once, by part
hobj0 = mf.Hessian()
log = logger.new_logger(hobj0, 0)
PH = [
    (rhf_hess, '_partial_hess_2e', lambda a, k: 'partial: 2e ' + ('LR-K' if k.get('omega') else 'J+K')),
    (rks_hess, '_get_vxc_diag', 'partial: XC diag'),
    (rks_hess, '_get_vxc_deriv2_dm', 'partial: XC deriv2'),
    (rks_hess, '_get_enlc_deriv2', 'partial: VV10 deriv2'),
    (rks_hess, '_get_vxc_deriv1', 'make_h1: XC deriv1'),
    (rks_hess, '_get_vnlc_deriv1', 'make_h1: VV10 deriv1'),
]
with timers(PH):
    t = time.perf_counter()
    de2_partial = hobj0.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, mf.max_memory, log)
    t_partial = time.perf_counter() - t
    t = time.perf_counter()
    h1ao = hobj0.make_h1(mo_coeff, mo_occ, None, atmlst, log)
    t_h1 = time.perf_counter() - t
h_nuc = hobj0.hess_nuc(mol, atmlst=atmlst)
print(f'partial_hess_elec {t_partial:.1f} s, make_h1 {t_h1:.1f} s')
for k in sorted(T):
    print(f'    {k:28s} {T[k]:8.1f} s')
t_h1_jk = t_h1 - sum(v for k, v in T.items() if k.startswith('make_h1'))
print(f'    {"make_h1: J/K deriv + hcore":28s} {t_h1_jk:8.1f} s', flush=True)


def lr_sgx_mf(level):
    '''A copy of mf whose long-range K comes from SGX on grid level, all else exact'''
    sg = rhf_hess._sgx_response_scf(mf, level).with_df
    mfh = mf.copy()

    def get_k(mol=None, dm=None, hermi=1, omega=None):
        if omega is None:
            return mf.get_k(mol, dm, hermi)
        t = time.perf_counter()
        vk = sg.get_jk(dm, hermi, None, with_j=False, with_k=True, omega=omega)[1]
        T['LR-K via SGX (AO dms)'] = T.get('LR-K via SGX (AO dms)', 0.) + time.perf_counter() - t
        return vk
    mfh.get_k = get_k
    return mfh


def solve(variant, tol):
    hobj = mf.Hessian()
    hobj.cphf_residual_tol = tol
    parts = variant.split('+')
    second = None
    nonlc = False
    fx_mf = None
    vgrid = None
    for p in parts:
        if p.startswith('g2='):
            second = p[3:]
        elif p == 'nonlc':
            nonlc = True
        elif p.startswith('vg='):
            vgrid = p[3:]
        elif p.startswith('sgx'):
            hobj.sgx_response = True
            hobj.sgx_response_level = int(p[3:])
        elif p.startswith('dfj='):
            hobj.sgx_response_auxbasis = p[4:]
        elif p.startswith('lrsgx'):
            fx_mf = lr_sgx_mf(int(p[5:]))
        elif p == 'stock':
            hobj.cphf_residual_tol = None
        elif p != 'exact':
            raise ValueError(p)
    mf.second_grids = None
    if second is not None:
        mf.set_second_grids(second if second == 'sg1' else int(second))
    nlc0 = mf.nlcgrids
    if vgrid is not None:
        if vgrid == 'sg1':
            g = gen_grid.sg1_grids(mol)
        else:
            g = gen_grid.Grids(mol)
            g.level = int(vgrid)
        g.build(with_non0tab=True)
        mf.nlcgrids = g
        if fx_mf is not None:
            fx_mf.nlcgrids = g
    gen0 = hf.RHF.__dict__['gen_response']
    if nonlc:
        def gen(mf_, *a, **k):
            k['with_nlc'] = False
            return gen0(mf_, *a, **k)
        hf.RHF.gen_response = gen
    count = {'calls': 0, 'vectors': 0}
    gen_vind0 = rhf_hess.gen_vind

    def counted(mf_, c, o):
        fx = gen_vind0(mf_, c, o)

        def f(mo1):
            count['calls'] += 1
            count['vectors'] += len(mo1.reshape(-1, nmo * nocc))
            return fx(mo1)
        return f
    rhf_hess.gen_vind = counted
    T.clear()
    try:
        with timers(SPECS):
            t = time.perf_counter()
            fx = None
            if fx_mf is not None:
                # the copy would otherwise hold mf's second_grids setting at the time of the copy: set it too
                fx_mf.second_grids = mf.second_grids
                fx = counted(fx_mf, mo_coeff, mo_occ)
            mo1, mo_e1 = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, fx, atmlst, mf.max_memory, log)
            t_cphf = time.perf_counter() - t
    finally:
        rhf_hess.gen_vind = gen_vind0
        hf.RHF.gen_response = gen0
        mf.second_grids = None
        mf.nlcgrids = nlc0
    hobj.partial_hess_elec = lambda *a, **k: de2_partial.copy()
    de = hobj.hess_elec(mo_energy, mo_coeff, mo_occ, mo1=mo1, mo_e1=mo_e1, h1ao=h1ao, atmlst=atmlst) + h_nuc
    return de, t_cphf, dict(T), count


def freqs(h):
    r = thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    return numpy.asarray(r['freq_wavenumber'])


t = time.perf_counter()
h_ref, t_ref, parts_ref, c_ref = solve('exact', args.ref_tol)
f_ref = freqs(h_ref)
print(f'\nreference (exact response, tol {args.ref_tol:g}): CPHF {t_ref:.1f} s, {c_ref["calls"]} calls, {c_ref["vectors"]} vectors; '
      f'max|H| {abs(h_ref).max():.4f}')
print('  frequencies (cm-1):', ' '.join(f'{x:.1f}' for x in f_ref))
print('  CPHF parts:', ', '.join(f'{k} {v:.1f}' for k, v in sorted(parts_ref.items())), flush=True)
if args.save:
    numpy.save(args.save + '_ref.npy', h_ref)
if args.compare_ref:
    h_other = numpy.load(args.compare_ref)
    f_other = freqs(h_other)
    df = f_ref - f_other
    i = abs(df).argmax()
    hi = abs(f_other) > 200
    print(f'this reference against {args.compare_ref}: max|dH| {abs(h_ref - h_other).max():.2e}, max|df| {abs(df).max():.3f} cm-1 '
          f'at {f_other[i]:.1f}, mean|df| {abs(df).mean():.3f}, max|df| above 200 cm-1 {abs(df[hi]).max():.3f}', flush=True)

print(f'\n{"variant":26s} {"CPHF s":>8s} {"calls":>5s} {"vecs":>5s} {"max|dH|":>9s} {"max|df|":>8s} {"mean|df|":>8s} '
      f'{"at f=":>8s} {"max|df| f>200":>13s}  parts')
for v in args.variants.split(','):
    try:
        h, tc, parts, c = solve(v, args.tol)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'{v:26s} failed: {type(e).__name__}: {e}', flush=True)
        continue
    f = freqs(h)
    df = f - f_ref
    i = abs(df).argmax()
    hi = abs(f_ref) > 200
    ps = ', '.join(f'{k} {x:.1f}' for k, x in sorted(parts.items()))
    print(f'{v:26s} {tc:8.1f} {c["calls"]:5d} {c["vectors"]:5d} {abs(h - h_ref).max():9.2e} {abs(df).max():8.3f} '
          f'{abs(df).mean():8.3f} {f_ref[i]:8.1f} {abs(df[hi]).max():13.3f}  {ps}', flush=True)
    if args.save:
        numpy.save(args.save + '_' + v.replace('=', '').replace('+', '_') + '.npy', h)
print('done', flush=True)
