import time, numpy, sys
from pyscf import gto, dft, lib
from pyscf.sgx import sgx_jk
print('mixed range supported:', sgx_jk.mixed_range_supported())
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svpd', verbose=0)
mol1 = gto.M(atom=mol.atom, basis='def2-svpd', charge=1, spin=1, verbose=0)
# K matrices of a density, one pass against two
mf = dft.RKS(mol, xc='wb97x').RIJCOSX(); mf.with_df.build(level=1)
dm = mf.get_init_guess()
w = mf.with_df
for ptol in (1e-9, None):
    w.sgx_tol_pair = ptol; w._rsh_df = {}; w.build(level=1)
    k = w.get_jk(dm, 1, None, False, True)[1]; klr = w.get_jk(dm, 1, None, False, True, omega=.3)[1]
    kmix = w.get_jk(dm, 1, None, False, True, omega=.3, mix=(.15, .85))[1]
    ref = .15 * k + .85 * klr
    print(f'K, pair_tol {ptol}: max|K_mix - (0.15 K + 0.85 K_LR)| {abs(kmix - ref).max():.1e} of {abs(ref).max():.2f}; E_K diff {-.25*numpy.einsum("ij,ji", dm, kmix - ref):+.1e}')
for name, make in [('RKS wb97m-v', lambda: dft.RKS(mol, xc='wb97m-v')), ('RKS cam-b3lyp', lambda: dft.RKS(mol, xc='cam-b3lyp')),
                   ('UKS wb97x', lambda: dft.UKS(mol1, xc='wb97x')), ('RKS b3lyp', lambda: dft.RKS(mol, xc='b3lyp')),
                   ('RKS hse06', lambda: dft.RKS(mol, xc='hse06'))]:
    out = []
    for flag in (False, True):
        sgx_jk.MIXED_RANGE = flag
        mf = make().RIJCOSX(); mf.conv_tol = 1e-10
        if mf.do_nlc(): mf.nlcgrids.level = 0
        t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
        g = mf.nuc_grad_method().kernel()
        out.append((e, mf.cycles, dt, g, len(mf.with_df._rsh_df)))
    print(f'{name:14s} E two passes {out[0][0]:.10f} one pass {out[1][0]:.10f} diff {out[1][0]-out[0][0]:+.1e}; cycles {out[0][1]} {out[1][1]}; '
          f't {out[0][2]:.2f} {out[1][2]:.2f}; max|dg| {abs(out[0][3]-out[1][3]).max():.1e}', flush=True)
