# The frequencies of one HCN <-> HNC saddle-point geometry (the CPU example's with SMD water) as the molecule is rotated
# and shifted rigidly, and at the second run's geometry: the example's model (wB97M-V/def2-TZVP, (99,590)/(50,194),
# two-level SCF, grid_response), gas phase and SMD. A rigid motion changes nothing physically: the spread is the
# orientation dependence of the cavity's surface points and of the XC grids.
import sys, glob, numpy
from pyscf import gto, dft
from pyscf.hessian import thermo
from scipy.spatial.transform import Rotation

def geom(d):
    L = open(glob.glob(d + '/*.xyz')[0]).read().split('\n')[2:5]
    return [l.split()[0] for l in L], numpy.array([[float(x) for x in l.split()[1:4]] for l in L])

def freqs(sym, x, smd, lebedev=None):
    mol = gto.M(atom=[(s, tuple(r)) for s, r in zip(sym, x)], basis='def2-tzvp', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    if smd:
        mf = mf.SMD(); mf.with_solvent.solvent = 'water'
        if lebedev:
            mf.with_solvent.lebedev_order = lebedev
    mf.two_level = True
    mf.set_second_grids(2)
    mf.set_second_nlcgrids(1)
    mf.kernel()
    g = mf.Gradients().kernel()
    hobj = mf.Hessian()
    hobj.grid_response = True
    h = hobj.kernel()
    f = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
    return mf.e_tot, abs(g).max(), f

sym, x0 = geom('cpu_ts/hcn_smd')
_, x1 = geom('cpu_ts/hcn_svp_smd')
rots = [('as it ended', numpy.eye(3), 0.)]
for k, seed in enumerate((1, 2, 3)):
    rots.append(('rotation %d' % (k+1), Rotation.random(random_state=seed).as_matrix(), 0.))
rots.append(('shifted (0.3,0.2,0.1) A', numpy.eye(3), numpy.array([.3, .2, .1])))
for smd in (True, False):
    print('SMD water' if smd else 'gas phase', flush=True)
    for label, R, t in rots:
        e, g, f = freqs(sym, x0.dot(R.T) + t, smd)
        print('  %-26s E %.9f  max |g| %.1e  freq %s' % (label, e, g, ' '.join('%.2f' % v for v in f)), flush=True)
    if smd:
        e, g, f = freqs(sym, x1, smd)
        print('  %-26s E %.9f  max |g| %.1e  freq %s' % ('the other run\'s geometry', e, g, ' '.join('%.2f' % v for v in f)), flush=True)
