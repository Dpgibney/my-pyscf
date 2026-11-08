'''Hessians and TD excitation energies through every branch of the RHF/RKS response (the J/K of each range
separation: B3LYP, HSE06 short range only, LRC-wPBE long range only, CAM-B3LYP and wB97M-V both; singlets and
triplets; HF), the UHF/UKS Hessians, and a B3LYP Hessian with the release of the semi-direct ERIs forced
(direct SCF, max_memory 1 MB for the Hessian), saved to compare two trees. CHECK_RELEASE=1 also asserts that the
ERIs are gone after that Hessian and that the next get_jk stores them again (trees from c6570dbac on).
Compare two outputs with numpy (the keys are the cases).

Usage: hess_resp_ref.py OUT.npz'''
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, scf, dft, lib  # noqa: E402

water = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
out = {}


def mol_(atom, charge=0, spin=0):
    return gto.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)


def run(name, fn):
    t = time.perf_counter()
    out[name] = numpy.asarray(fn())
    print(f'{name:28s} {time.perf_counter() - t:6.1f} s  max|x| {abs(out[name]).max():.6f}', flush=True)


def hess(mf, release=False, direct=False):
    mf.conv_tol = 1e-11
    if direct or release:
        # direct SCF (with its semi-direct ERIs) although the ERIs fit in memory
        mf._is_mem_enough = lambda: False
    mf.kernel()
    if direct or release:
        assert getattr(mf._opt[None], 'semi_direct_eri', None) is not None
    if release:
        # the Hessian's arrays then do not fit beside the semi-direct ERIs
        mf.max_memory = 1
    h = mf.Hessian().kernel()
    if release and os.environ.get('CHECK_RELEASE') == '1':
        sd = getattr(mf._opt[None], 'semi_direct_eri', 'absent')
        assert mf.semi_direct and sd == 'absent', (mf.semi_direct, sd)
        mf.max_memory = 4000
        mf.get_jk(mf.mol, mf.make_rdm1())
        assert getattr(mf._opt[None], 'semi_direct_eri', None) is not None
    return h


run('hf_water', lambda: hess(scf.RHF(mol_(water))))
run('uhf_water+', lambda: hess(scf.UHF(mol_(water, 1, 1))))
run('b3lyp_water', lambda: hess(dft.RKS(mol_(water), xc='b3lyp')))
run('b3lyp_water_direct', lambda: hess(dft.RKS(mol_(water), xc='b3lyp'), direct=True))
run('b3lyp_water_release', lambda: hess(dft.RKS(mol_(water), xc='b3lyp'), release=True))
run('uks_b3lyp_water+', lambda: hess(dft.UKS(mol_(water, 1, 1), xc='b3lyp')))
run('hse06_water', lambda: hess(dft.RKS(mol_(water), xc='hse06')))
run('lrcwpbe_water', lambda: hess(dft.RKS(mol_(water), xc='lrc-wpbe')))
run('camb3lyp_water', lambda: hess(dft.RKS(mol_(water), xc='camb3lyp')))


def wb97mv(atom):
    mf = dft.RKS(mol_(atom), xc='wb97m-v')
    mf.nlcgrids.level = 1
    return mf


run('wb97mv_water', lambda: hess(wb97mv(water)))
run('wb97mv_propane', lambda: hess(wb97mv(n_alkane(3))))


def td(mf, kind, singlet, exclude_nlc=True):
    mf.conv_tol = 1e-11
    mf.kernel()
    t = getattr(mf, kind)()
    t.nstates = 4
    t.singlet = singlet
    t.conv_tol = 1e-10
    if hasattr(t, 'exclude_nlc'):
        t.exclude_nlc = exclude_nlc
    t.kernel()
    return t.e


for kind in ('TDA', 'TDDFT'):
    for singlet in (True, False):
        s = 'S' if singlet else 'T'
        run(f'tdhf_{kind}_{s}', lambda: td(scf.RHF(mol_(water)), 'TDA' if kind == 'TDA' else 'TDHF',
                                             singlet))
        for xc in ('b3lyp', 'hse06', 'lrc-wpbe', 'camb3lyp'):
            run(f'{xc}_{kind}_{s}', lambda: td(dft.RKS(mol_(water), xc=xc), kind, singlet))
    run(f'wb97mv_{kind}_S_nlc', lambda: td(wb97mv(water), kind, True, exclude_nlc=False))
numpy.savez(sys.argv[1], **out)
