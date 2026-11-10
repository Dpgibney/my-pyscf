import time, numpy, sys
from pyscf import gto, dft, lib
from pyscf.hessian import thermo
mol = gto.M(atom=sys.argv[1], basis='def2-tzvpd', charge=-1, verbose=0)
mf = dft.RKS(mol, xc='wb97m-v')
mf.grids.atom_grid = {'default': (99, 590), 'P': (150, 590)}; mf.nlcgrids.atom_grid = (50, 194)
mf = mf.SMD(); mf.with_solvent.solvent = 'water'; mf.conv_tol = 1e-11
mf.kernel()
def freqs(h):
    return numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
def run(sg=None, snl=None, **kw):
    mf.second_grids = None; mf.second_nlcgrids = None
    if sg is not None: mf.set_second_grids(sg)
    if snl is not None: mf.set_second_nlcgrids(snl)
    h = mf.Hessian(); h.set(**kw)
    t = time.perf_counter(); out = h.kernel(); return out, time.perf_counter() - t
h0, t0 = run(cphf_residual_tol=1e-7)
f0 = freqs(h0)
print('exact response %.0f s; frequencies' % t0, numpy.round(f0, 1), flush=True)
for label, args, kw in (('residual tol 1e-4', (), dict(cphf_residual_tol=1e-4)),
                        ('second_grids 2', (2,), dict(cphf_residual_tol=1e-6)),
                        ('second_nlcgrids 0', (None, 0), dict(cphf_residual_tol=1e-6)),
                        ('second_nlcgrids 1', (None, 1), dict(cphf_residual_tol=1e-6)),
                        ('sgx_response level 1', (), dict(sgx_response=True)),
                        ('sgx_response level 2', (), dict(sgx_response=True, sgx_response_level=2)),
                        ('sgx_response level 1, tol 1e-6', (), dict(sgx_response=True, cphf_residual_tol=1e-6)),
                        ('all three (grids 2, nlc 0, sgx 1)', (2, 0), dict(sgx_response=True))):
    h, t = run(*args, **kw)
    df = freqs(h) - f0
    k = abs(df).argmax()
    print(f'{label:34s} {t:5.0f} s  max|dH| {abs(h - h0).max():.1e}  max|df| {abs(df).max():.3f} cm-1 (mode at {f0[k]:.0f}), mean {abs(df).mean():.3f}; '
          f'real modes above 300: max {abs(df[f0 > 300]).max():.3f}', flush=True)
