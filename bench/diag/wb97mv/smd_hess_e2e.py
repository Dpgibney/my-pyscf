'''A wB97M-V/def2-TZVPD Hessian in SMD water end to end through the public options: two-level SCF, gradient, then the
Hessian with the exact CPHF response and with the response approximations of NOTES (second_grids 2, second_nlcgrids 1,
sgx_response level 1). A line per step. Usage: smd_hess_e2e.py XYZ|NAME [--charge 0] [--skip-exact]'''
import argparse, time, numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.hessian import thermo
import molecules
p = argparse.ArgumentParser()
p.add_argument('mol'); p.add_argument('--charge', type=int, default=0); p.add_argument('--skip-exact', action='store_true')
p.add_argument('--save'); p.add_argument('--compare')
a = p.parse_args()
T0 = time.perf_counter()
def say(msg):
    print(f'[{time.perf_counter() - T0:6.0f} s] {msg}', flush=True)
mol = gto.M(atom=getattr(molecules, a.mol.upper(), None) or a.mol, basis='def2-tzvpd', charge=a.charge, verbose=0)
def make():
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = {'default': (99, 590), 'P': (150, 590), 'S': (150, 590), 'Cl': (150, 590)}
    mf.nlcgrids.atom_grid = (50, 194)
    mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    return mf
mf = make(); mf.two_level = True
t = time.perf_counter(); e = mf.kernel()
say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads; {mol.natm} atoms, {mol.nao} AOs, '
    f'{len(mf.with_solvent.surface["grid_coords"])} surface points')
say(f'two-level SCF: {time.perf_counter() - t:.1f} s, {mf.cycles} Fock builds, conv {mf.converged}, E {e:.10f}')
t = time.perf_counter(); g = mf.Gradients().kernel(); say(f'gradient: {time.perf_counter() - t:.1f} s, max |g| {abs(g).max():.2e}')
def freqs(h):
    return numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
def hessian(label, setup):
    hobj = mf.Hessian(); setup(hobj)
    for name in ('partial_hess_elec', 'solve_mo1'):
        def wrap(fn, name):
            def w(*args, **kw):
                t = time.perf_counter(); r = fn(*args, **kw); say(f'  {name}: {time.perf_counter() - t:.1f} s'); return r
            return w
        setattr(hobj, name, wrap(getattr(hobj, name), name))
    solv = mf.with_solvent; hess0 = solv.hess
    def hess_timed(dm):
        t = time.perf_counter(); r = hess0(dm); say(f'  solvent Hessian: {time.perf_counter() - t:.1f} s'); return r
    solv.hess = hess_timed
    t = time.perf_counter(); h = hobj.kernel(); solv.hess = hess0
    say(f'Hessian, {label}: {time.perf_counter() - t:.1f} s')
    return h
mf.set_second_grids(2); mf.set_second_nlcgrids(1)
def approx(hobj):
    hobj.sgx_response = True
h1 = hessian('response on second_grids 2 / second_nlcgrids 1 with sgx_response level 1', approx)
if a.save:
    numpy.save(a.save, h1)
f1 = freqs(h1)
if a.compare:
    hc = numpy.load(a.compare)
    say(f'against the saved Hessian: max|dH| {abs(h1 - hc).max():.2e}, max|df| {abs(f1 - freqs(hc)).max():.4f} cm^-1')
say('frequencies (cm^-1): ' + ' '.join('%.1f' % x for x in f1[:12]) + (' ...' if len(f1) > 12 else ''))
if not a.skip_exact:
    mf.second_grids = None; mf.second_nlcgrids = None
    def exact(hobj):
        hobj.cphf_residual_tol = 1e-6
    h0 = hessian('exact response', exact)
    f0 = freqs(h0)
    say(f'approximate against exact response: max|dH| {abs(h1 - h0).max():.2e}, max|df| {abs(f1 - f0).max():.3f} cm^-1, mean {abs(f1 - f0).mean():.3f}')
