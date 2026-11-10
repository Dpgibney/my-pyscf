import time, numpy, sys
from pyscf import gto, scf, dft, lib
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svpd', verbose=0)
mol1 = gto.M(atom=mol.atom, basis='def2-svpd', charge=1, spin=1, verbose=0)
cases = [('RKS wb97m-v', lambda: dft.RKS(mol, xc='wb97m-v')), ('RKS b3lyp', lambda: dft.RKS(mol, xc='b3lyp')),
         ('RKS pbe', lambda: dft.RKS(mol, xc='pbe')), ('RHF', lambda: scf.RHF(mol)),
         ('UKS wb97x', lambda: dft.UKS(mol1, xc='wb97x')), ('UHF', lambda: scf.UHF(mol1)), ('ROKS b3lyp', lambda: dft.ROKS(mol1, xc='b3lyp')),
         ('RKS b3lyp df', lambda: dft.RKS(mol, xc='b3lyp').density_fit()),
         ('RKS wb97m-v PCM', lambda: dft.RKS(mol, xc='wb97m-v').PCM()),
         ('RKS wb97m-v RIJCOSX', lambda: dft.RKS(mol, xc='wb97m-v').RIJCOSX())]
for name, make in cases:
    def setup():
        mf = make()
        if hasattr(mf, 'grids'): mf.grids.atom_grid = (99, 590)
        if hasattr(mf, 'nlcgrids'): mf.nlcgrids.atom_grid = (50, 194)
        return mf
    try:
        mf = setup(); t = time.perf_counter(); e0 = mf.kernel(); t0 = time.perf_counter() - t; c0 = mf.cycles
        g0 = mf.nuc_grad_method().kernel() if 'ROKS' not in name else None
        mf = setup(); mf.two_level = True; mf.verbose = 4 if '-v' == sys.argv[-1] else 0
        t = time.perf_counter(); e1 = mf.kernel(); t1 = time.perf_counter() - t
        g1 = mf.nuc_grad_method().kernel() if 'ROKS' not in name else None
        dg = '' if g0 is None else f' max|dg| {abs(g1 - g0).max():.1e}'
        print(f'{name:22s} dE {e1 - e0:+.1e} conv {mf.converged} builds {c0 + 2} -> {mf.cycles}  t {t0:.1f} -> {t1:.1f}{dg}', flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(name, 'FAILED', type(e).__name__, e, flush=True)
