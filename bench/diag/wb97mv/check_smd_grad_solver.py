'''solvent/grad/smd.py grad_solver by blocks of rows against the version with the full [n,n,3] arrays (upstream's, the
file given as argv[1]: git -C pyscf show a72443310:pyscf/solvent/grad/smd.py > old.py); and PCM's IEF-PCM and SS(V)PE gradients against finite differences of the energy.'''
import sys, importlib.util, numpy
from pyscf import gto, dft, scf
from pyscf.solvent.grad import smd as smd_grad
spec = importlib.util.spec_from_file_location('smd_grad_old', sys.argv[1]); old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
mols = {'h2po4-': ('bench/diag/wb97mv/h2po4_opt.xyz', -1), 'glycine': ('bench/diag/wb97mv/glycine_opt.xyz', 0)}
for name, (xyz, charge) in mols.items():
    for solvent in ('water', 'toluene'):
        mol = gto.M(atom=xyz, basis='def2-svpd', charge=charge, verbose=0)
        mf = dft.RKS(mol, xc='b3lyp').SMD(); mf.with_solvent.solvent = solvent
        mf.kernel(); dm = mf.make_rdm1()
        a = smd_grad.grad_solver(mf.with_solvent, dm); b = old.grad_solver(mf.with_solvent, dm)
        g = mf.Gradients().kernel()
        print(f'{name:8s} SMD {solvent:8s} {len(mf.with_solvent.surface["charge_exp"]):5d} points: grad_solver max|new - old| '
              f'{abs(a - b).max():.1e} (max |term| {abs(b).max():.1e}); total gradient max {abs(g).max():.2e}', flush=True)
mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47; N 2.5 0.3 0.1; H 2.6 1.2 0.5; H 3.3 0.1 -0.5; H 2.4 -0.4 0.8',
            basis='def2-svp', verbose=0)
for method in ('IEF-PCM', 'SS(V)PE', 'C-PCM'):
    def energy(m):
        mf = scf.RHF(m).PCM(); mf.with_solvent.method = method; mf.conv_tol = 1e-12
        return mf, mf.kernel()
    mf, e0 = energy(mol)
    g = mf.Gradients().kernel()
    err = 0
    for ia, x in ((0, 2), (3, 0), (5, 1)):
        es = []
        for d in (1e-3, -1e-3):
            c = mol.atom_coords().copy(); c[ia, x] += d
            es.append(energy(mol.set_geom_(c, unit='Bohr', inplace=False))[1])
        err = max(err, abs((es[0] - es[1]) / 2e-3 - g[ia, x]))
    print(f'RHF {method:8s}: analytical gradient against finite differences of the energy, 3 components: {err:.1e}', flush=True)
