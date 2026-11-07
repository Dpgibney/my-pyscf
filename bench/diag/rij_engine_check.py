"""The J-engine of df_jk.get_j (rij_engine.c) against the integral-direct RI-J through int3c2e (df_jk.RIJ_ENGINE =
False): max |vj - vj_ref| for a random symmetric density and for the density of a few SCF cycles, over bases with
general contractions (cc-pVDZ), Cartesian functions, f shells with g auxiliary functions and a heavier element, and
the time of each. Run with PYTHONPATH set to the tree.
Usage: rij_engine_check.py [--quick]"""
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
from pyscf import gto, dft, df  # noqa: E402
from pyscf.df import df_jk  # noqa: E402

CASES = [
    ('water', WATER, 'cc-pvdz', 'cc-pvdz-jkfit', False),
    ('water cart', WATER, 'cc-pvdz', 'cc-pvdz-jkfit', True),
    ('water', WATER, 'def2-qzvp', 'def2-universal-jfit', False),
    ('benzene', BENZENE, 'def2-svp', 'def2-universal-jfit', False),
    ('CH3Br', 'C 0 0 0; Br 0 0 1.94; H 1.03 0 -0.36; H -0.51 0.89 -0.36; H -0.51 -0.89 -0.36',
     'def2-tzvp', 'def2-universal-jfit', False),
    ('caffeine', CAFFEINE, 'def2-tzvp', 'def2-universal-jfit', False),
]
if '--quick' in sys.argv:
    CASES = CASES[:4]

worst = 0
for name, atom, basis, aux, cart in CASES:
    mol = gto.M(atom=atom, basis=basis, cart=cart, verbose=0)
    rng = numpy.random.default_rng(1)
    a = rng.standard_normal((mol.nao, mol.nao))
    dm_rand = (a + a.T) * .05
    mf = dft.RKS(mol, xc='b3lyp').density_fit(auxbasis=aux)
    mf.max_cycle = 4
    mf.kernel()
    dm_scf = mf.make_rdm1()
    dms = numpy.asarray([dm_scf, dm_rand])
    res = {}
    for engine in (False, True):
        df_jk.RIJ_ENGINE = engine
        dfobj = df.DF(mol, auxbasis=aux)
        df_jk.get_j(dfobj, dms[0], hermi=1)
        t = time.perf_counter()
        vj = df_jk.get_j(dfobj, dms, hermi=1)
        res[engine] = (vj, time.perf_counter() - t)
    (v0, t0), (v1, t1) = res[False], res[True]
    err = [abs(v1[i] - v0[i]).max() for i in range(2)]
    scale = abs(v0[0]).max()
    worst = max(worst, max(err) / scale)
    print(f'{name:10s} {basis:10s} {aux:20s} nao {mol.nao:4d} naux {dfobj.auxmol.nao:5d}: max |dvj| SCF dm '
          f'{err[0]:.2e}, random dm {err[1]:.2e} (|vj| {scale:.1f}); int3c2e {t0:.3f} s, J-engine {t1:.3f} s '
          f'({t0 / t1:.2f}x)', flush=True)
print(f'worst relative difference {worst:.2e}')
