'''The default CPHF solver of Hessians: cphf.solve_residual to cphf_residual_tol = 1e-4 (the default now) against
PySCF's Krylov solver to conv_tol_cpscf (cphf_residual_tol = None, the default before), both against the residual
solver at 1e-9. Exact, density-fitted and solvated models, closed and open shells; the time is solve_mo1's.'''
import time, numpy
from pyscf import gto, dft, scf
from pyscf.hessian import thermo
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
neutral = gto.M(atom=atom, basis='def2-svpd', verbose=0)
cation = gto.M(atom=atom, basis='def2-svpd', charge=1, spin=1, verbose=0)
def freqs(mol, h):
    return numpy.asarray(thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber'])
def hess(mf, **kw):
    hobj = mf.Hessian()
    for k, v in kw.items():
        setattr(hobj, k, v)
    solve0 = hobj.solve_mo1; dt = [0.]
    def solve(*a, **k):
        t = time.perf_counter(); r = solve0(*a, **k); dt[0] += time.perf_counter() - t; return r
    hobj.solve_mo1 = solve
    return hobj.kernel(), dt[0]
def ks(cls, xc):
    def make(mol):
        mf = cls(mol, xc=xc)
        if 'wb97m' in xc:
            mf.nlcgrids.level = 1
        return mf
    return make
cases = [('RHF', neutral, scf.RHF), ('RKS pbe', neutral, ks(dft.RKS, 'pbe')), ('RKS b3lyp', neutral, ks(dft.RKS, 'b3lyp')),
         ('RKS wb97m-v', neutral, ks(dft.RKS, 'wb97m-v')),
         ('RKS b3lyp, density fitting', neutral, lambda m: ks(dft.RKS, 'b3lyp')(m).density_fit()),
         ('RKS b3lyp, PCM', neutral, lambda m: ks(dft.RKS, 'b3lyp')(m).PCM()),
         ('RKS wb97m-v, SMD', neutral, lambda m: ks(dft.RKS, 'wb97m-v')(m).SMD()),
         ('UHF', cation, scf.UHF), ('UKS pbe', cation, ks(dft.UKS, 'pbe')), ('UKS b3lyp', cation, ks(dft.UKS, 'b3lyp')),
         ('UKS wb97m-v', cation, ks(dft.UKS, 'wb97m-v')),
         ('UKS b3lyp, density fitting', cation, lambda m: ks(dft.UKS, 'b3lyp')(m).density_fit()),
         ('UKS b3lyp, PCM', cation, lambda m: ks(dft.UKS, 'b3lyp')(m).PCM())]
print(f'{"":28s} {"Krylov, conv_tol_cpscf 1e-8 (before)":>44s} {"residual 1e-4 (default now)":>44s}')
print(f'{"":28s} {"max|dH|  max|df| cm-1   CPHF":>44s} {"max|dH|  max|df| cm-1   CPHF":>44s}')
for name, mol, make in cases:
    mf = make(mol); mf.conv_tol = 1e-11; mf.kernel()
    href, _ = hess(mf, cphf_residual_tol=1e-9)
    f0 = freqs(mol, href)
    out = []
    for kw in (dict(cphf_residual_tol=None), {}):
        h, t = hess(mf, **kw)
        out.append(f'{abs(h - href).max():10.1e} {abs(freqs(mol, h) - f0).max():10.3f} {t:8.1f} s')
    print(f'{name:28s} {out[0]:>44s} {out[1]:>44s}', flush=True)
