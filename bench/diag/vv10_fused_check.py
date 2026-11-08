'''nr_nlc_vxc on the fused kernels against its block_loop path (FUSED_XC off
for the VV10 call only): nelec, E_nlc and the matrix, for a tagged dm (the
orbital path), an untagged one (the dm path) and a non-symmetric one (hermi=0).'''
import sys, time
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint

GEOM = {
    'water': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    'caffeine': '''C 1.5463 -1.0266 -0.0000; N 0.4183 -1.9087 -0.0000; C -0.8530 -1.4230 -0.0000;
N -1.0617 -0.0848 -0.0000; C 0.0539 0.8233 0.0000; C 1.3320 0.3418 0.0000; N 2.3170 1.2826 0.0000;
C 1.6372 2.4142 0.0000; N 0.2957 2.2239 0.0000; C -0.5841 3.3637 0.0000; C -2.4141 0.4612 -0.0000;
C 0.6315 -3.3466 -0.0000; O 2.6929 -1.4471 -0.0000; O -1.7783 -2.2165 -0.0000; H 2.0500 3.4097 0.0000;
H -1.6192 3.0193 0.0000; H -0.3958 3.9646 0.8908; H -0.3958 3.9646 -0.8908; H -2.3874 1.5522 -0.0000;
H -2.9447 0.1153 0.8913; H -2.9447 0.1153 -0.8913; H 1.6998 -3.5603 -0.0000; H 0.1440 -3.7741 0.8889;
H 0.1440 -3.7741 -0.8889''',
}

def nlc(ni, mol, grids, dm, hermi, fused):
    saved = numint.FUSED_XC
    numint.FUSED_XC = fused
    try:
        t0 = time.perf_counter()
        r = ni.nr_nlc_vxc(mol, grids, 'wb97m-v', dm, hermi=hermi, max_memory=4000)
        return r, time.perf_counter() - t0
    finally:
        numint.FUSED_XC = saved

for name, basis in (('water', 'def2-svp'), ('water', 'def2-qzvp'), ('caffeine', 'def2-svp'),
                    ('caffeine', 'def2-tzvp')):
    mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.nlcgrids.level = 1
    dm0 = mf.get_init_guess()
    mf.initialize_grids(mol, dm0)
    grids = mf.nlcgrids
    ni = mf._numint
    fused = numint._fused_xc(ni, mol, grids, mol.nao, 'GGA', 4000, (dm0,))
    mo_e, mo = numpy.linalg.eigh(mf.get_hcore() + mf.get_veff(mol, dm0) * 0)
    mo_occ = numpy.zeros(mol.nao); mo_occ[:mol.nelectron//2] = 2
    dm_tag = lib.tag_array(mf.make_rdm1(mo, mo_occ), mo_coeff=mo, mo_occ=mo_occ)
    rng = numpy.random.default_rng(1)
    dm_ns = numpy.asarray(dm_tag) + 1e-3 * rng.standard_normal((mol.nao, mol.nao))
    print(f'{name}/{basis} nao {mol.nao} nlc grid {grids.weights.size} fused {fused is not None}')
    for label, dm, hermi in (('tagged', dm_tag, 1), ('untagged', numpy.asarray(dm_tag), 1),
                             ('non-symmetric', dm_ns, 0)):
        (n0, e0, v0), t0 = nlc(ni, mol, grids, dm, hermi, False)
        (n1, e1, v1), t1 = nlc(ni, mol, grids, dm, hermi, True)
        print(f'  {label:14s} nelec {n1:.8f} ({n1-n0:+.1e})  E_nlc {e1:.12f} ({e1-e0:+.1e})  '
              f'max|dv| {abs(v1-v0).max():.1e} (|v| {abs(v0).max():.2e})  '
              f'{t0:.2f} -> {t1:.2f} s')
    sys.stdout.flush()
