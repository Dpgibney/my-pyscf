'''UHF/UKS Hessians with the residual-based CPHF solver (ucphf.solve_residual) and the seminumerical response
(sgx_response): against the residual solver at 1e-9 (the stock Krylov solver stops short for DFT), and a closed-shell molecule as UKS against RKS.'''
import time, numpy
from pyscf import gto, dft, scf
from pyscf.hessian import thermo
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
def freqs(mol, h):
    return numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
def hess(mf, **kw):
    h = mf.Hessian()
    for k, v in kw.items():
        setattr(h, k, v)
    t = time.perf_counter(); out = h.kernel(); return out, time.perf_counter() - t
cation = gto.M(atom=atom, basis='def2-svpd', charge=1, spin=1, verbose=0)
neutral = gto.M(atom=atom, basis='def2-svpd', verbose=0)
for name, make in (('UHF', lambda m: scf.UHF(m)), ('UKS pbe', lambda m: dft.UKS(m, xc='pbe')),
                   ('UKS b3lyp', lambda m: dft.UKS(m, xc='b3lyp')), ('UKS wb97m-v', lambda m: dft.UKS(m, xc='wb97m-v'))):
    mf = make(cation); mf.conv_tol = 1e-11
    if 'wb97m' in name:
        mf.nlcgrids.level = 1
    mf.kernel()
    href, tref = hess(mf, cphf_residual_tol=1e-9)
    f0 = freqs(cation, href)
    for label, kw in (('stock Krylov solver', {}),
                      ('residual tol 1e-6', dict(cphf_residual_tol=1e-6)), ('residual tol 1e-4', dict(cphf_residual_tol=1e-4)),
                      ('sgx_response level 1', dict(sgx_response=True)),
                      ('sgx_response level 2', dict(sgx_response=True, sgx_response_level=2)),
                      ('sgx_response level 1, aug_etb', dict(sgx_response=True, sgx_response_auxbasis='aug_etb'))):
        if 'sgx' in label and 'pbe' in name:
            continue
        h, t = hess(mf, **kw)
        print(f'{name:12s} {label:30s} max|dH| {abs(h - href).max():.1e}  max|df| {abs(freqs(cation, h) - f0).max():.3f} cm-1  {t:.1f} s (residual tol 1e-9: {tref:.1f} s)', flush=True)
# closed shell: UKS with sgx_response against RKS with it
for xc in ('b3lyp', 'wb97m-v'):
    out = []
    for cls in (dft.RKS, dft.UKS):
        mf = cls(neutral, xc=xc); mf.conv_tol = 1e-11
        if 'wb97m' in xc:
            mf.nlcgrids.level = 1
        mf.kernel()
        out.append(hess(mf, sgx_response=True, cphf_residual_tol=1e-7)[0])
    print(f'closed shell {xc}: max|H(UKS, sgx_response) - H(RKS, sgx_response)| {abs(out[0] - out[1]).max():.1e}', flush=True)
