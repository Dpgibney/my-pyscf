'''make_h1's int2e_ip1 J/K (and long-range K) of sample atoms of the 57-atom doublet at two screening thresholds
(MAKE_H1_TOL), and the long-range K second derivatives with the two-density bound.
Usage: h1jk_big.py XYZ NPZ [--atoms 0,30] [--tols 1e-14,1e-12] [--lr-hess TOL]'''
import sys, time, numpy
from pyscf import gto, dft, lib
from pyscf.hessian import rhf as rhf_hess, uks as uks_hess
args = [x for x in sys.argv[1:] if not x.startswith('--')]
opts = dict(x[2:].split('=') for x in sys.argv[1:] if x.startswith('--'))
mol = gto.M(atom=args[0], basis='def2-tzvp', spin=1, verbose=0)
z = numpy.load(args[1]); mo, occ = z['mo_coeff'], z['mo_occ']
dms = numpy.array([mo[s][:, occ[s] > 0] @ mo[s][:, occ[s] > 0].T for s in range(2)])
dm0a, dm0b = dms
dm = dm0a + dm0b
omega, alpha, hyb = dft.UKS(mol, xc='wb97m-v')._numint.rsh_and_hybrid_coeff('wb97m-v', spin=mol.spin)
aoslices = mol.aoslice_by_atom()
nbf = numpy.array([aoslices[i][3] - aoslices[i][2] for i in range(mol.natm)])
print(f'{mol.natm} atoms, {mol.nao} AOs, {lib.num_threads()} threads; AOs per atom: '
      f'{dict(zip(*[x.tolist() for x in numpy.unique(nbf, return_counts=True)]))}', flush=True)
atoms = [int(x) for x in opts.get('atoms', '0,30').split(',') if x]
ref = {}
for tol in [float(x) for x in opts.get('tols', '1e-14,1e-12').split(',')]:
    rhf_hess.MAKE_H1_TOL = tol
    t = time.perf_counter()
    vhfopt = rhf_hess._make_h1_vhfopt(mol, [dm, dm0a, dm0b], 'jk')
    with mol.with_range_coulomb(omega):
        vhfopt_lr = rhf_hess._make_h1_vhfopt(mol, [dm0a, dm0b], 'k')
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        t = time.perf_counter()
        v = uks_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                             ['ji->s2kl', -dm[:,p0:p1], 'lk->s1ij', -dm,
                              'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                              'jk->s1il', -dm0a, 'jk->s1il', -dm0b],
                             shls_slice=shls_slice, vhfopt=vhfopt)
        t1 = time.perf_counter()
        with mol.with_range_coulomb(omega):
            vlr = uks_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                   ['li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                                    'jk->s1il', -dm0a, 'jk->s1il', -dm0b],
                                   shls_slice=shls_slice, vhfopt=vhfopt_lr)
        t2 = time.perf_counter()
        key = ia
        if key not in ref:
            ref[key] = (v, vlr)
        d = max(abs(x - y).max() for x, y in zip(list(v) + list(vlr), list(ref[key][0]) + list(ref[key][1])))
        print(f'tol {tol:.0e}: J/K of atom {ia} ({mol.atom_symbol(ia)}, {p1 - p0} AOs): {t1 - t:.1f} s + long-range K '
              f'{t2 - t1:.1f} s; max difference of the matrices from the first threshold {d:.1e}', flush=True)
if 'lr-hess' in opts:
    tol = float(opts['lr-hess'])
    stats = []
    t = time.perf_counter()
    with mol.with_range_coulomb(omega):
        ej, ek = rhf_hess.hess_jk_energy(mol, None, dms, 2, tol=tol, stats=stats)
    print(f'long-range K second derivatives, two density elements, tol {tol:.0e}: {time.perf_counter() - t:.1f} s, '
          f'{stats[0]:.3e} orbits, {stats[1]:.3e} integral evaluations (max |ek| {abs(ek).max():.2f})', flush=True)
    numpy.save('dd_big_lr_ek.npy', ek)
