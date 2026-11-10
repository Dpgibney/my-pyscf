import time, numpy
from pyscf import gto, scf, dft
from pyscf.sgx import sgx
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47', basis='def2-svp', verbose=0)
mol1 = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47', basis='def2-svp', charge=1, spin=1, verbose=0)
for name, make in [('RKS b3lyp', lambda: dft.RKS(mol, xc='b3lyp')), ('RKS wb97m-v', lambda: dft.RKS(mol, xc='wb97m-v')),
                   ('RKS hse06', lambda: dft.RKS(mol, xc='hse06')), ('RKS lrc-wpbe', lambda: dft.RKS(mol, xc='lrc-wpbe')),
                   ('RKS pbe', lambda: dft.RKS(mol, xc='pbe')), ('UKS wb97x', lambda: dft.UKS(mol1, xc='wb97x')),
                   ('ROKS b3lyp', lambda: dft.ROKS(mol1, xc='b3lyp')), ('RHF', lambda: scf.RHF(mol)), ('UHF', lambda: scf.UHF(mol1))]:
    out = []
    for k_only in (True, False):
        mf = make().RIJCOSX()
        mf._energy_on_grid_k_only = k_only
        mf.conv_tol = 1e-10
        t = time.perf_counter(); e = mf.kernel(); out.append((e, time.perf_counter() - t))
    mf = make().RIJCOSX(); mf.with_df.grids_level_energy = None; mf.conv_tol = 1e-10; e0 = mf.kernel()
    print(f'{name:12s} k-only {out[0][0]:.10f} full {out[1][0]:.10f} diff {out[0][0]-out[1][0]:+.1e}  change from the SCF grid {out[0][0]-e0:+.1e}  t {out[0][1]:.2f} {out[1][1]:.2f}')
