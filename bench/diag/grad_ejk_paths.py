"""Nuclear gradients with grad.rhf.JK_ENERGY on and off (It20): every SCF type whose grad_elec takes
the J/K energy gradient, and types that must keep the J/K matrices (density fitting, DFT+U)."""
import os
import sys
import time
import numpy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import WATER, BENZENE  # noqa: E402
from pyscf import gto, scf, dft, lib  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402

water = gto.M(atom=WATER, basis='def2-svp', verbose=0)
water_p = gto.M(atom=WATER, basis='def2-svp', charge=1, spin=1, verbose=0)
benzene = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)


def ks(cls, mol, xc, **kw):
    mf = cls(mol, xc=xc)
    for k, v in kw.items():
        setattr(mf, k, v)
    return mf


cases = [
    ('RHF water', scf.RHF(water), {}),
    ('UHF water+', scf.UHF(water_p), {}),
    ('ROHF water+', scf.ROHF(water_p), {}),
    ('RKS PBE water', ks(dft.RKS, water, 'PBE'), {}),
    ('RKS B3LYP water', ks(dft.RKS, water, 'B3LYP'), {}),
    ('RKS B3LYP water grid_response', ks(dft.RKS, water, 'B3LYP'), {'grid_response': True}),
    ('RKS CAM-B3LYP water', ks(dft.RKS, water, 'CAM-B3LYP'), {}),
    ('RKS wB97X water', ks(dft.RKS, water, 'wB97X'), {}),
    ('RKS r2SCAN water', ks(dft.RKS, water, 'r2SCAN'), {}),
    ('UKS PBE water+', ks(dft.UKS, water_p, 'PBE'), {}),
    ('UKS B3LYP water+', ks(dft.UKS, water_p, 'B3LYP'), {}),
    ('UKS wB97X water+', ks(dft.UKS, water_p, 'wB97X'), {}),
    ('ROKS B3LYP water+', ks(dft.ROKS, water_p, 'B3LYP'), {}),
    ('ROKS CAM-B3LYP water+', ks(dft.ROKS, water_p, 'CAM-B3LYP'), {}),
    ('RKS B3LYP water atmlst', ks(dft.RKS, water, 'B3LYP'), {'atmlst': [2, 0]}),
    ('DF-RKS B3LYP water (matrices)', ks(dft.RKS, water, 'B3LYP').density_fit(), {}),
    ('RKS B3LYP benzene', ks(dft.RKS, benzene, 'B3LYP'), {}),
]
for name, mf, kw in cases:
    mf.conv_tol = 1e-11
    mf.kernel()
    g = mf.nuc_grad_method()
    atmlst = kw.pop('atmlst', None)
    for k, v in kw.items():
        setattr(g, k, v)
    out = []
    for flag in (False, True, False, True):
        with lib.temporary_env(rhf_grad, JK_ENERGY=flag):
            t0 = time.perf_counter()
            de = g.kernel(atmlst=atmlst)
            out.append((de.copy(), time.perf_counter() - t0))
    split = rhf_grad._VEFF_SPLIT.get(getattr(g.get_veff, '__func__', None)) is not None
    d = abs(out[1][0] - out[0][0]).max()
    rep = max(abs(out[2][0] - out[0][0]).max(), abs(out[3][0] - out[1][0]).max())
    print(f'{name:32s} split {split!s:5s} max|d| {d:.1e} (repeat {rep:.0e})  sum {abs(out[1][0].sum(axis=0)).max():.1e}'
          f'  t old {min(out[0][1], out[2][1]):.2f} new {min(out[1][1], out[3][1]):.2f} s', flush=True)
