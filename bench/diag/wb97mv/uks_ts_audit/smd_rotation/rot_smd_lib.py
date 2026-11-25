import glob, numpy
from pyscf import gto, dft
from pyscf.hessian import thermo

def geom(d):
    L = open(glob.glob(d + '/*.xyz')[0]).read().split('\n')[2:5]
    return [l.split()[0] for l in L], numpy.array([[float(x) for x in l.split()[1:4]] for l in L])

def freqs(sym, x, smd, lebedev=None):
    mol = gto.M(atom=[(s, tuple(r)) for s, r in zip(sym, x)], basis='def2-tzvp', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    n = 0
    if smd:
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
        if lebedev:
            mf.with_solvent.lebedev_order = lebedev
    mf.two_level = True
    mf.set_second_grids(2)
    mf.set_second_nlcgrids(1)
    mf.kernel()
    if smd:
        n = len(mf.with_solvent.surface['grid_coords'])
    g = mf.Gradients().kernel()
    hobj = mf.Hessian()
    hobj.grid_response = True
    h = hobj.kernel()
    f = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
    return mf.e_tot, abs(g).max(), f, n
