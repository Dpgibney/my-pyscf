'''The analytical Hessian with and without grid_response against central differences of the gradient with
grid_response (Richardson extrapolation of the steps 5e-4 and 1e-3 Bohr), RKS or UKS.
Usage: hess_grid_response_fd.py ATOMS XC1,XC2,.. NRAD,NANG BASIS [NLC_NRAD,NLC_NANG] [CHARGE] [SPIN]'''
import sys, time, numpy
from pyscf import gto, dft, grad, hessian
atom, xcs, grid, basis = sys.argv[1], sys.argv[2].split(','), tuple(int(x) for x in sys.argv[3].split(',')), sys.argv[4]
nlcgrid = tuple(int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else '20,50').split(','))
charge = int(sys.argv[6]) if len(sys.argv) > 6 else 0
spin = int(sys.argv[7]) if len(sys.argv) > 7 else 0
mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
for xc in xcs:
    t0 = time.time()
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc); mf.grids.atom_grid = grid; mf.nlcgrids.atom_grid = nlcgrid
    mf.conv_tol = 1e-13; mf.conv_tol_grad = 1e-10; mf.kernel()
    res = {}
    for gr in (False, True):
        hobj = mf.Hessian(); hobj.grid_response = gr; hobj.cphf_residual_tol = 1e-8
        t = time.time(); res[gr] = hobj.kernel(); res[(gr, 't')] = time.time() - t
    g = mf.Gradients(); g.grid_response = True
    gs = g.as_scanner(); x0 = mol.atom_coords(); n = mol.natm
    def fd(st):
        h = numpy.zeros((n, n, 3, 3))
        for i in range(n):
            for d in range(3):
                r = []
                for s in (st, -st):
                    x = x0.copy(); x[i, d] += s
                    r.append(gs(mol.set_geom_(x, unit='Bohr', inplace=False))[1])
                h[i, :, d, :] = (r[0] - r[1]) / (2*st)
        return .5 * (h + h.transpose(1, 0, 3, 2))
    f1, f2 = fd(5e-4), fd(1e-3)
    ref = (4 * f1 - f2) / 3
    print(f'{xc:10s} grids {grid}/{nlcgrid}: max |H| {abs(ref).max():.3f}; analytical - FD(Richardson) max: '
          f'grid_response=False {abs(res[False]-ref).max():.2e}, True {abs(res[True]-ref).max():.2e} '
          f'(FD 5e-4 vs 1e-3: {abs(f1-f2).max():.1e}); sum rule {abs(res[True].sum(axis=0)).max():.1e}; '
          f'Hessian {res[(False,"t")]:.1f} / {res[(True,"t")]:.1f} s; {time.time()-t0:.0f} s', flush=True)
