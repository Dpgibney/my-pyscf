"""The CPHF equations of the production-like Hessian, solved by several solvers on one set of orbitals and one
right-hand side.

The SCF and the first-order Fock matrices (make_h1) are computed once and kept in a file (--state); each variant then
runs solve_mo1 with its settings of the Hessian object, with the parts of the response timed and counted (the model's
own response and, for the two-level CPHF, the cheap model's), and its solution is compared with a reference solution:
the first-order orbitals, and the terms of the Hessian that are made of them (with the second-order term of
solve_mo1). The reference is the variant named by --ref, kept in the state file once it has run, or the first variant.

    python cphf_bench.py --basis def2-svp --solvent none --state scratch/cphf_svp_gas \\
        --variant residual:cphf_two_level=False --variant "2l:cphf_two_level=True" \\
        --variant "2l_nlc:cphf_two_level=True,cphf_two_level_parts=('nlc',)"
"""
import argparse
import ast
import collections
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf import lib as pyscf_lib  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess, rks as rks_hess  # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract  # noqa: E402
from gpu4pyscf.sgx import sgx as sgx_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--grid', default='99,590')
ap.add_argument('--nlcgrid', default='50,194')
ap.add_argument('--solvent', default='smd', choices=['smd', 'none'])
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--set', action='append', default=[], help='attribute=value of the SCF object')
ap.add_argument('--state', default=None, help='file (.npz added) that keeps the orbitals and h1; made if missing')
ap.add_argument('--variant', action='append', default=[], help='name:attr=value,attr=value of the Hessian object')
ap.add_argument('--ref', default=None, help='the variant whose solution is kept in the state file as the reference')
ap.add_argument('--label', default='')
ap.add_argument('--out', default=None)
ap.add_argument('--verbose', type=int, default=4, help='of the CPHF solvers')
args = ap.parse_args()
MOLS = {'caffeine': CAFFEINE, 'benzene': BENZENE, 'water': WATER}


def sync():
    cupy.cuda.Device().synchronize()


class Parts:
    """Inclusive and exclusive wall times of wrapped callables (as prod_parts.py)"""

    def __init__(self):
        self.rows = collections.OrderedDict()
        self.stack = []

    def wrap(self, owner, name, label, keyfn=None):
        if not hasattr(owner, name):
            return
        orig = getattr(owner, name)
        parts = self

        def timed(*a, **kw):
            key = label + keyfn(*a, **kw) if keyfn else label
            sync()
            parts.stack.append([key, 0.0])
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                sync()
                dt = time.perf_counter() - t0
                _, inner = parts.stack.pop()
                row = parts.rows.setdefault(key, [0, 0.0, 0.0])
                row[0] += 1
                row[1] += dt
                row[2] += dt - inner
                if parts.stack:
                    parts.stack[-1][1] += dt
        timed.__name__ = getattr(orig, '__name__', name)
        setattr(owner, name, timed)

    def take(self):
        rows, self.rows = self.rows, collections.OrderedDict()
        return rows


parts = Parts()
_response = {'cheap': False}


def _cheap_suffix(*a, **kw):
    return ' (cheap)' if _response['cheap'] else ''


def _veff_resp_suffix(hessobj, *a, **kw):
    _response['cheap'] = getattr(hessobj, '_cheap_response', None) is not None
    return _cheap_suffix()


def value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def settings(spec):
    """'a=1,b=(2,3)' -> {'a': 1, 'b': (2, 3)}: commas inside brackets belong to the value"""
    out, depth, start = {}, 0, 0
    items = []
    for n, ch in enumerate(spec):
        depth += ch in '([{'
        depth -= ch in ')]}'
        if ch == ',' and depth == 0:
            items.append(spec[start:n])
            start = n + 1
    items.append(spec[start:])
    for item in items:
        if item.strip():
            key, val = item.split('=', 1)
            out[key.strip()] = value(val.strip())
    return out


mol = gto.M(atom=MOLS.get(args.mol, args.mol), basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
mf.grids.atom_grid = tuple(int(x) for x in args.grid.split(','))
if args.nlcgrid != 'none':
    mf.nlcgrids.atom_grid = tuple(int(x) for x in args.nlcgrid.split(','))
if args.solvent == 'smd':
    mf = mf.SMD()
    mf.with_solvent.solvent = 'water'
mf.conv_tol = args.conv_tol
for spec in args.set:
    for key, val in settings(spec).items():
        setattr(mf, key, val)

state = args.state + '.npz' if args.state else None
if state and os.path.exists(state):
    kept = numpy.load(state)
    assert kept['mo_coeff'].shape[0] == mol.nao
    mf.mo_coeff = cupy.asarray(kept['mo_coeff'])
    mf.mo_energy = cupy.asarray(kept['mo_energy'])
    mf.mo_occ = cupy.asarray(kept['mo_occ'])
    mf.e_tot = float(kept['e_tot'])
    mf.converged = True
    h1mo = cupy.asarray(kept['h1mo'])
    print(f'orbitals and h1 from {state}: E = {mf.e_tot:.12f}', flush=True)
    kept = {key: kept[key] for key in kept.files}
else:
    t0 = time.perf_counter()
    mf.kernel()
    sync()
    t1 = time.perf_counter()
    assert mf.converged
    hobj = mf.Hessian()
    h1mo = hobj.make_h1(cupy.asarray(mf.mo_coeff), cupy.asarray(mf.mo_occ), None, None, None)
    sync()
    print(f'E = {mf.e_tot:.12f}: SCF {t1 - t0:.1f} s, make_h1 {time.perf_counter() - t1:.1f} s', flush=True)
    kept = dict(mo_coeff=cupy.asnumpy(mf.mo_coeff), mo_energy=cupy.asnumpy(mf.mo_energy),
                mo_occ=cupy.asnumpy(mf.mo_occ), e_tot=mf.e_tot, h1mo=cupy.asnumpy(h1mo))
    if state:
        numpy.savez(state, **kept)
    del hobj
mo_coeff = cupy.asarray(mf.mo_coeff)
mo_energy = cupy.asarray(mf.mo_energy)
mo_occ = cupy.asarray(mf.mo_occ)
with_solvent = getattr(mf, 'with_solvent', None)
print(f"{args.label or 'cphf'}: {args.mol} {args.basis} {args.xc} solvent={args.solvent} nao={mol.nao} "
      f"natm={mol.natm} tree={os.path.dirname(gpu4pyscf.__file__)}", flush=True)

# the parts of a response
parts.wrap(rks_hess.Hessian, 'get_veff_resp_mo', 'veff_resp_mo', _veff_resp_suffix)
parts.wrap(rks_hess.Hessian, 'get_jk_mo', 'jk_mo', _cheap_suffix)
for name in ('nr_rks_fxc_mo', 'nr_rks_fnlc_mo', 'get_nlc_response_ground_state'):
    parts.wrap(rks_hess, name, name, _cheap_suffix)
parts.wrap(rhf_hess, '_cheap_start_and_trials', 'cheap_start_and_trials')
parts.wrap(sgx_mod.SGX, 'get_k', 'sgx.get_k', _cheap_suffix)
if with_solvent is not None:
    from gpu4pyscf.solvent.hessian import pcm as pcm_hess
    parts.wrap(type(with_solvent), '_B_dot_x', 'solvent._B_dot_x', _cheap_suffix)
    parts.wrap(pcm_hess, '_response_from_mo_integrals', 'solvent.response_mo', _cheap_suffix)
    parts.wrap(pcm_hess, 'surface_mo_integrals', 'solvent.mo_integrals')
if hasattr(mf._numint, 'cache_xc_kernel'):
    parts.wrap(type(mf._numint), 'cache_xc_kernel', 'cache_xc_kernel', _cheap_suffix)

def response_terms(mo1, mo_e1, second):
    """The terms of the Hessian that hess_elec (restricted) makes of the first-order orbitals and orbital energies,
    with the second-order term of solve_mo1"""
    from gpu4pyscf.pbc.gto import int1e
    mo1 = cupy.asarray(mo1)
    mo_e1 = cupy.asarray(mo_e1)
    de2 = contract('kxpi,lypi->klxy', h1mo, mo1) * 4
    if second is not None:
        de2 += cupy.asarray(second)
    mo1 = contract('kxai,pa->kxpi', mo1, mo_coeff)
    nao = mo_coeff.shape[0]
    mocc = mo_coeff[:, mo_occ > 0]
    mocc_e = mocc * mo_energy[mo_occ > 0]
    s1a = -int1e.int1e_ipovlp(mol)
    for i0, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        s1ao = cupy.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1oo = contract('xiq,qj->xij', contract('xpq,pi->xiq', s1ao, mocc), mocc)
        de2[i0] -= contract('xij,kyij->kxy', s1oo, mo_e1) * 2
        s1mo = contract('xpq,qi->xpi', s1ao, mocc_e)
        de2[i0] -= contract('xpi,kypi->kxy', s1mo, mo1) * 4
    return (de2 + de2.transpose(1, 0, 3, 2)) * .5


# the memory that the response routines find available (their batches of trial densities follow it)
_avail = []
_get_avail_mem = rks_hess.get_avail_mem


def _recording_avail_mem(*a, **kw):
    out = _get_avail_mem(*a, **kw)
    _avail.append(out)
    return out


rks_hess.get_avail_mem = _recording_avail_mem

first = None
if 'ref_mo1' in kept:
    first = (str(kept['ref_name']), cupy.asarray(kept['ref_mo1']), cupy.asarray(kept['ref_de2']))
    print(f'reference solution from {state}: {first[0]}', flush=True)
records = []
for spec in args.variant or ['residual:cphf_two_level=False']:
    name, _, conf = spec.partition(':')
    conf = settings(conf)
    hobj = mf.Hessian()
    for key, val in conf.items():
        assert hasattr(hobj, key), key
        setattr(hobj, key, val)
    _response['cheap'] = False
    print(f'--- {name}: {conf}', flush=True)
    cupy.get_default_memory_pool().free_all_blocks()
    sync()
    t0 = time.perf_counter()
    failed = None
    try:
        fx = hobj.gen_vind(mo_coeff, mo_occ)
        if with_solvent is None:
            mo1, mo_e1 = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx, None, 4000, args.verbose)
        else:
            # as inside the kernel of the solvent's Hessian object: with the solvent's response
            with pyscf_lib.temporary_env(with_solvent, equilibrium_solvation=True):
                mo1, mo_e1 = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx, None, 4000, args.verbose)
            hobj._solvent_mo_integrals = None
        mo1 = cupy.asarray(mo1)
    except Exception as err:
        import traceback
        traceback.print_exc(file=sys.stdout)
        failed = f'{type(err).__name__}: {str(err)[:80]}'
        parts.stack.clear()
    sync()
    dt = time.perf_counter() - t0
    rows = parts.take()
    pool = cupy.get_default_memory_pool()
    print(f"    {name}: {dt:.1f} s{'  FAILED: ' + failed if failed else ''}")
    if _avail:
        print(f"    available memory seen by the response routines: {min(_avail) / 2**20:.0f} to "
              f"{max(_avail) / 2**20:.0f} MiB; pool now {pool.used_bytes() / 2**20:.0f} MiB used, "
              f"{pool.total_bytes() / 2**20:.0f} MiB held; device free {cupy.cuda.Device().mem_info[0] / 2**20:.0f} MiB")
        del _avail[:]
    print(f"    {'part':40s} {'calls':>5s} {'incl s':>9s} {'excl s':>9s}")
    for label, (n, incl, excl) in rows.items():
        print(f"    {label:40s} {n:5d} {incl:9.2f} {excl:9.2f}")
    rec = {'label': args.label, 'variant': name, 'settings': {k: repr(v) for k, v in conf.items()},
           'basis': args.basis, 'solvent': args.solvent, 'time': round(dt, 2), 'failed': failed,
           'parts': {k: [v[0], round(v[1], 2), round(v[2], 2)] for k, v in rows.items()}}
    if failed is None:
        # the term of the Hessian made of the first-order orbitals (hess_elec), with what solve_mo1 hands to
        # hess_elec for its second order in the residuals, in both of its index orders
        second = getattr(hobj, '_cphf_second_order', None)
        if second is not None:
            rec['max_second_order'] = float(abs(second).max())
        de2 = response_terms(mo1, mo_e1, second)
        if state and name == args.ref and 'ref_mo1' not in kept:
            kept.update(ref_name=name, ref_mo1=cupy.asnumpy(mo1), ref_de2=cupy.asnumpy(de2))
            numpy.savez(state, **kept)
            first = None
        if first is None:
            first = (name, mo1, de2)
        else:
            rec['max_dmo1'] = float(abs(mo1 - first[1]).max())
            rec['max_dhess'] = float(abs(de2 - first[2]).max())
            print(f"    vs {first[0]}: max |dmo1| = {rec['max_dmo1']:.3e}, max |dH| of the response terms = "
                  f"{rec['max_dhess']:.3e} (largest element {float(abs(first[2]).max()):.3f}; second-order term "
                  f"{rec.get('max_second_order', 0.):.3e})")
        del mo1, de2, mo_e1
    records.append(rec)
    print(flush=True)
    del hobj
if args.out:
    with open(args.out, 'a') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
print('CPHF_BENCH_DONE')
