'''The XC and VV10 Fock matrix derivatives (the split forms of make_h1: uks._get_vxc_deriv1, rks._get_vnlc_deriv1)
of a UKS molecule from saved orbitals, timed by their parts, in the tree on PYTHONPATH.
Usage: xc_h1_big.py XYZ SPIN NPZ [--no-vv10] [--no-xc] [--enlc]
  --enlc  the VV10 energy second derivative first (it hands its kernel terms to the VV10 Fock derivative)'''
import sys, time, functools, numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad
from pyscf.hessian import rks as rks_hess, uks as uks_hess
args = [x for x in sys.argv[1:] if not x.startswith('--')]
opts = [x for x in sys.argv[1:] if x.startswith('--')]
T0 = time.perf_counter()
TIMES = {}
def say(msg):
    print(f'>>> [{time.perf_counter() - T0:6.0f} s, {lib.current_memory()[0] / 1e3:5.1f} GB] {msg}', flush=True)
import inspect
def wrap(obj, name, label=None):
    if not hasattr(obj, name):
        return
    static = isinstance(obj, type) and isinstance(inspect.getattr_static(obj, name), staticmethod)
    fn = getattr(obj, name)
    label = label or name
    @functools.wraps(fn)
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            n, tot = TIMES.get(label, (0, 0.))
            TIMES[label] = (n + 1, tot + time.perf_counter() - t)
    setattr(obj, name, staticmethod(wrapper) if static else wrapper)
def table(title):
    say(title)
    for label, (n, tot) in sorted(TIMES.items(), key=lambda kv: -kv[1][1]):
        print(f'      {tot:9.1f} s  {n:6d} call(s)  {label}', flush=True)
    TIMES.clear()
for cls_name in dir(numint):
    cls = getattr(numint, cls_name)
    if isinstance(cls, type):
        for meth in ('dot_ao_mo', 'dot_terms_owner_mo', 'dot_terms', '_dot_wv', 'rho_lowrank'):
            if meth in vars(cls):
                wrap(cls, meth, 'fused.' + meth)
for mod, names in ((rks_hess, ('_becke_weight_derivs', '_make_dR_rho1', '_grid_response_wv_', '_grid_motion_vmat_',
                               '_vv10_drho_dA', '_vv10_add_grid_response_', '_vv10_dot_terms', '_becke_radii_table')),
                   (rks_grad, ('_gga_grad_sum_', '_tau_grad_dot_', '_d1_dot_')),
                   (numint, ('_dot_ao_dm', '_dot_sets_aow_ao', 'eval_ao'))):
    for name in names:
        wrap(mod, name)
for name in ('f_t', 'weight1_pair', 'grid_response_pair', '__init__'):
    wrap(rks_hess._VV10HessData, name, 'vv10.' + name)
wrap(numint.NumInt, 'eval_xc_eff'); wrap(numint.NumInt, 'eval_rho2'); wrap(numint.NumInt, 'eval_ao', 'ni.eval_ao')

mol = gto.M(atom=args[0], basis='def2-tzvp', spin=int(args[1]), verbose=0)
mf = dft.UKS(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
z = numpy.load(args[2])
mf.mo_coeff, mf.mo_occ, mf.mo_energy = z['mo_coeff'], z['mo_occ'], z['mo_energy']
mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
hobj = mf.Hessian()
hobj.grid_response = True
nao = mol.nao
import pyscf
say(f'pyscf {pyscf.__file__}; {mol.natm} atoms, {nao} AOs, {lib.num_threads()} threads, max_memory {mf.max_memory}')
mocc = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(2)]
max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
if '--no-xc' not in opts:
    h1occ = tuple(numpy.zeros((mol.natm, 3, nao, mocc[s].shape[1])) for s in range(2))
    t = time.perf_counter()
    v = uks_hess._get_vxc_deriv1(hobj, mo_coeff, mo_occ, max_memory, occ_out=h1occ)
    say(f'uks._get_vxc_deriv1: {time.perf_counter() - t:.1f} s; XC grid {mf.grids.weights.size} points; '
        f'|h1ao| {numpy.linalg.norm(v[0]):.8f} {numpy.linalg.norm(v[1]):.8f}, |h1occ| {numpy.linalg.norm(h1occ[0]):.8f} '
        f'{numpy.linalg.norm(h1occ[1]):.8f}')
    table('parts of uks._get_vxc_deriv1 (nested parts overlap):')
    v = h1occ = None
if '--no-vv10' not in opts:
    mo_sf, occ_sf = uks_hess._total_density_orbitals(mo_coeff, mo_occ)
    if '--enlc' in opts:
        t = time.perf_counter()
        d2 = rks_hess._get_enlc_deriv2(hobj, mo_sf, occ_sf, max_memory)
        say(f'_get_enlc_deriv2: {time.perf_counter() - t:.1f} s; |d2e| {numpy.linalg.norm(d2):.8f}')
        table('parts of _get_enlc_deriv2 (nested parts overlap):')
    occ_out = numpy.zeros((mol.natm, 3, nao, mo_sf.shape[1]))
    t = time.perf_counter()
    v = rks_hess._get_vnlc_deriv1(hobj, mo_sf, occ_sf, max_memory, occ_out=occ_out)
    say(f'_get_vnlc_deriv1: {time.perf_counter() - t:.1f} s; VV10 grid {mf.nlcgrids.weights.size} points; '
        f'|h1| {numpy.linalg.norm(v):.8f}, |occ_out| {numpy.linalg.norm(occ_out):.8f}')
    table('parts of _get_vnlc_deriv1 (nested parts overlap):')
