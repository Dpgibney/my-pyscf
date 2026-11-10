'''End to end on caffeine wB97M-V/def2-TZVPD, grids (99,590)/(50,194), through the public API of the new options.
Prints a line as each step ends. Usage: e2e_caf.py scf|rijcosx|hess'''
import sys, time, numpy
import pyscf
from pyscf import gto, dft, df, lib
from pyscf.hessian import thermo, rhf as rhf_hess, rks as rks_hess
from molecules import CAFFEINE
T0 = time.perf_counter()
def say(msg):
    print(f'[{time.perf_counter() - T0:6.0f} s] {msg}', flush=True)
E_REF = -680.3497717994
S = sys.argv[2]
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
def make():
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    return mf
mode = sys.argv[1]
say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads, mode {mode}')
if mode == 'scf':
    mf = make(); mf.two_level = True
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    say(f'two-level SCF: {dt:.1f} s, {mf.cycles} Fock builds of the model, conv {mf.converged}, E {e:.10f}, E - E(plain SCF) {e - E_REF:+.2e}')
    numpy.savez(S + '/caf_two_level_scf.npz', mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy, e_tot=e)
    t = time.perf_counter(); g = mf.Gradients().kernel(); dt = time.perf_counter() - t
    g0 = numpy.load(S + '/caf_exact_grad.npy')
    say(f'exact gradient: {dt:.1f} s (156 s before); max|g - g(plain SCF, old code)| {abs(g - g0).max():.2e}')
elif mode == 'rijcosx':
    mf = make().RIJCOSX(auxbasis=df.aug_etb(mol))
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    say(f'RIJCOSX(aug_etb) SCF, production grids: {dt:.1f} s, {mf.cycles} cycles, E - E(exact) {e - E_REF:+.2e}')
    t = time.perf_counter(); g = mf.Gradients().kernel(); dt = time.perf_counter() - t
    g0 = numpy.load(S + '/caf_exact_grad.npy')
    say(f'RIJCOSX(aug_etb) gradient: {dt:.1f} s; max|g - g(exact)| {abs(g - g0).max():.2e}')
    mf = make().RIJCOSX(auxbasis=df.aug_etb(mol)); mf.two_level = True
    t = time.perf_counter(); e2 = mf.kernel(); dt = time.perf_counter() - t
    say(f'the same with two_level (coarse grids in the iterations): {dt:.1f} s, {mf.cycles} builds on the production grids, E - E(plain RIJCOSX) {e2 - e:+.2e}')
else:
    z = numpy.load(S + '/caf_exact_scf.npz')
    mf = make()
    mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot, mf.converged = z['mo_coeff'], z['mo_occ'], z['mo_energy'], float(z['e_tot']), True
    mf.set_second_grids(2); mf.set_second_nlcgrids(0)
    hobj = mf.Hessian(); hobj.sgx_response = True
    for owner, name in ((hobj, 'partial_hess_elec'), (hobj, 'solve_mo1')):
        def wrap(fn, name):
            def w(*a, **k):
                t = time.perf_counter(); r = fn(*a, **k); say(f'{name}: {time.perf_counter() - t:.1f} s'); return r
            return w
        setattr(owner, name, wrap(getattr(owner, name), name))
    for mod, name in ((rks_hess, '_get_vxc_deriv1'), (rks_hess, '_get_vnlc_deriv1'), (rks_hess, '_make_h1_split')):
        def wrap(fn, name):
            def w(*a, **k):
                t = time.perf_counter(); r = fn(*a, **k); say(f'{name}: {time.perf_counter() - t:.1f} s'); return r
            return w
        setattr(mod, name, wrap(getattr(mod, name), name))
    rhf_hess._MAKE_H1_SPLIT[rks_hess.make_h1] = rks_hess._make_h1_split
    t = time.perf_counter(); h = hobj.kernel(); dt = time.perf_counter() - t
    say(f'Hessian (split H1, second grids 2 / VV10 level 0, sgx_response level 1, one-pass K): {dt:.1f} s (3176 s exact before)')
    h0 = numpy.load(S + '/hv_caf_ref.npy')
    f = numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
    f0 = numpy.asarray(thermo.harmonic_analysis(mol, h0, imaginary_freq=False)['freq_wavenumber'])
    say(f'against the exact Hessian of the review: max|dH| {abs(h - h0).max():.2e}, max|df| {abs(f - f0).max():.3f} cm-1, mean|df| {abs(f - f0).mean():.3f}')
say('done')
