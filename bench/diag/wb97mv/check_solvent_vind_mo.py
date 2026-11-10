'''The CPHF response of a solvated model from the factors (the solvent wrapper's vind_mo: the gas-phase vind_mo plus
the solvent's response) against the AO matrices, which it took for every term before: Hessians and CPHF times, RKS
and UKS, PCM and SMD, exact response and sgx_response. Usage: check_solvent_vind_mo.py [XYZ charge basis]'''
import sys, time, numpy
from pyscf import gto, dft, scf
from pyscf.solvent import _attach_solvent
atom = 'O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8'
if len(sys.argv) > 1:
    mol = gto.M(atom=sys.argv[1], charge=int(sys.argv[2]), basis=sys.argv[3], verbose=0)
    cases = [('RKS wb97m-v SMD', lambda: dft.RKS(mol, xc='wb97m-v').SMD())]
else:
    mol = gto.M(atom=atom, basis='def2-svpd', verbose=0)
    mol1 = gto.M(atom=atom, basis='def2-svpd', charge=1, spin=1, verbose=0)
    cases = [('RHF PCM', lambda: scf.RHF(mol).PCM()), ('RKS pbe PCM', lambda: dft.RKS(mol, xc='pbe').PCM()),
             ('RKS wb97m-v SMD', lambda: dft.RKS(mol, xc='wb97m-v').SMD()),
             ('UKS b3lyp PCM', lambda: dft.UKS(mol1, xc='b3lyp').PCM()), ('UKS wb97m-v SMD', lambda: dft.UKS(mol1, xc='wb97m-v').SMD())]
make_vind_mo = _attach_solvent._solvent_vind_mo
for name, make in cases:
    mf = make(); mf.conv_tol = 1e-11
    if getattr(mf, 'nlcgrids', None) is not None:
        mf.nlcgrids.level = 1
    mf.kernel()
    for label, kw in (('exact response', dict(cphf_residual_tol=1e-8)), ('sgx_response', dict(sgx_response=True, cphf_residual_tol=1e-8))):
        out = []
        for factors in (True, False):
            _attach_solvent._solvent_vind_mo = make_vind_mo if factors else (lambda *args: None)
            hobj = mf.Hessian(); hobj.set(**kw)
            tc = [0.]
            solve = hobj.solve_mo1
            def timed(*args, **kwargs):
                t = time.perf_counter(); r = solve(*args, **kwargs); tc[0] = time.perf_counter() - t; return r
            hobj.solve_mo1 = timed
            out.append((hobj.kernel(), tc[0]))
        _attach_solvent._solvent_vind_mo = make_vind_mo
        print(f'{name:18s} {label:15s} max|H(factors) - H(AO matrices)| {abs(out[0][0] - out[1][0]).max():.1e}; CPHF {out[1][1]:.1f} -> {out[0][1]:.1f} s', flush=True)
