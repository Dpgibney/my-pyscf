import time, numpy, sys
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad
from molecules import CAFFEINE, n_alkane
def run(name, mol, mf, xc, grid, nlc):
    ni = mf._numint
    dm = mf.make_rdm1()
    dm = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
    g = dft.gen_grid.Grids(mol)
    if isinstance(grid, tuple): g.atom_grid = grid
    else: g.level = grid
    g.build(with_non0tab=True)
    gn = dft.gen_grid.Grids(mol)
    if isinstance(nlc, tuple): gn.atom_grid = nlc
    else: gn.level = nlc
    gn.build(with_non0tab=True)
    for rep in range(2):
        t = time.perf_counter(); v = rks_grad.get_vxc(ni, mol, g, xc, dm, max_memory=8000)[1]; t1 = time.perf_counter() - t
        t = time.perf_counter(); q = rks_grad.get_vxc_dm(ni, mol, g, xc, mf.mo_coeff, mf.mo_occ, 8000); t2 = time.perf_counter() - t
    ref = numpy.einsum('xij,ij->xi', v, dm)
    print(f'{name} {xc} XC grid {g.weights.size}: matrices {t1:.2f} s, with the density {t2:.2f} s, max diff {abs(q - ref).max():.1e} (max {abs(ref).max():.2f})', flush=True)
    if mf.do_nlc():
        for rep in range(2):
            t = time.perf_counter(); v = rks_grad.get_nlc_vxc(ni, mol, gn, xc, dm, max_memory=8000)[1]; t1 = time.perf_counter() - t
            t = time.perf_counter(); q = rks_grad.get_nlc_vxc_dm(ni, mol, gn, xc, dm, mf.mo_coeff, mf.mo_occ, 8000); t2 = time.perf_counter() - t
        ref = numpy.einsum('xij,ij->xi', v, dm)
        print(f'{name} {xc} VV10 grid {gn.weights.size}: matrices {t1:.2f} s, with the density {t2:.2f} s, max diff {abs(q - ref).max():.1e} (max {abs(ref).max():.1e})', flush=True)
z = numpy.load(sys.argv[1])
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v'); mf.mo_coeff, mf.mo_occ = z['mo_coeff'], z['mo_occ']
run('caffeine/def2-TZVPD', mol, mf, 'wb97m-v', (99, 590), (50, 194))
run('caffeine/def2-TZVPD', mol, mf, 'wb97m-v', 3, 1)
mol = gto.M(atom=n_alkane(20), basis='def2-tzvp', verbose=0)
mf = dft.RKS(mol, xc='pbe').RIJCOSX(); mf.kernel()
for xc in ('pbe', 'r2scan', 'wb97m-v'):
    m = dft.RKS(mol, xc=xc); m.mo_coeff, m.mo_occ = mf.mo_coeff, mf.mo_occ
    run('n-C20H42/def2-TZVP', mol, m, xc, 3, 1)
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='pbe').RIJCOSX(); mf.kernel()
for xc in ('lda,vwn', 'pbe', 'r2scan'):
    m = dft.RKS(mol, xc=xc); m.mo_coeff, m.mo_occ = mf.mo_coeff, mf.mo_occ
    run('caffeine/def2-SVP', mol, m, xc, 3, 1)
