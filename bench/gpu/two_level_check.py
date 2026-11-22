"""The two-level SCF of GPU4PySCF (mf.two_level = True) against the plain SCF of the same model (node1): energies and
nuclear gradients for a set of small cases (restricted and unrestricted, pure, global and range-separated hybrids,
VV10, density fitting, PCM and SMD), each converged to --conv-tol. The two-level solution is the model's own, so the
energies agree to the convergence threshold and the gradients to its square root times the gradient's sensitivity.
Prints one line per case and a summary; --out FILE appends a JSON line per case.
Usage: two_level_check.py [--basis def2-svp] [--conv-tol 1e-10] [--cases NAME,...] [--mol water|benzene|caffeine]
                          [--out FILE]"""
import argparse
import json
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, WATER  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks, uks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--cases', default=None)
ap.add_argument('--mol', default='water')
ap.add_argument('--out', default=None)
args = ap.parse_args()
ATOM = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE}[args.mol]

# name: (xc, charge, spin, unrestricted, density fitting, solvent, attributes)
CASES = {
    'rks_pbe': ('pbe', 0, 0, False, False, None, {}),
    'rks_b3lyp': ('b3lyp', 0, 0, False, False, None, {}),
    'rks_r2scan': ('r2scan', 0, 0, False, False, None, {}),
    'rks_wb97x': ('wb97x', 0, 0, False, False, None, {}),
    'rks_wb97mv': ('wb97m-v', 0, 0, False, False, None, {}),
    'rks_wb97mv_prodgrids': ('wb97m-v', 0, 0, False, False, None,
                             {'grids.atom_grid': (99, 590), 'nlcgrids.atom_grid': (50, 194)}),
    'rks_wb97mv_smd': ('wb97m-v', 0, 0, False, False, 'smd', {}),
    'rks_b3lyp_cpcm': ('b3lyp', 0, 0, False, False, 'cpcm', {}),
    'rks_b3lyp_iefpcm': ('b3lyp', 0, 0, False, False, 'iefpcm', {}),
    'rks_b3lyp_df': ('b3lyp', 0, 0, False, True, None, {}),
    'rks_wb97mv_df_smd': ('wb97m-v', 0, 0, False, True, 'smd', {}),
    'rks_b3lyp_levelshift': ('b3lyp', 0, 0, False, False, None, {'level_shift': 0.2}),
    'uks_b3lyp_cation': ('b3lyp', 1, 1, True, False, None, {}),
    'uks_wb97mv_cation_smd': ('wb97m-v', 1, 1, True, False, 'smd', {}),
    'uks_pbe_cation': ('pbe', 1, 1, True, False, None, {}),
    'uks_wb97mv_closed': ('wb97m-v', 0, 0, True, False, None, {}),
}
names = args.cases.split(',') if args.cases else list(CASES)


def set_attr(obj, name, value):
    *path, last = name.split('.')
    for p in path:
        obj = getattr(obj, p)
    setattr(obj, last, value)


def build(name, two_level):
    xc, charge, spin, unrestricted, df, solvent, attrs = CASES[name]
    mol = gto.M(atom=ATOM, basis=args.basis, charge=charge, spin=spin, verbose=0)
    mf = (uks.UKS if unrestricted else rks.RKS)(mol, xc=xc)
    for k, v in attrs.items():
        set_attr(mf, k, v)
    if df:
        mf = mf.density_fit()
    if solvent == 'smd':
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    elif solvent:
        mf = mf.PCM()
        mf.with_solvent.method = {'cpcm': 'C-PCM', 'iefpcm': 'IEF-PCM'}[solvent]
        mf.with_solvent.eps = 78.3553
    mf.conv_tol = args.conv_tol
    mf.two_level = two_level
    return mf


def run(name, two_level):
    mf = build(name, two_level)
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cupy.cuda.Device().synchronize()
    t_scf = time.perf_counter() - t0
    g = mf.Gradients().kernel()
    g = cupy.asnumpy(g) if isinstance(g, cupy.ndarray) else numpy.asarray(g)
    return {'e': float(e), 'converged': bool(mf.converged), 'cycles': int(mf.cycles), 't_scf': t_scf, 'grad': g}


print(f'tree {os.path.dirname(gpu4pyscf.__file__)}, {args.mol} {args.basis}, conv_tol {args.conv_tol:g}')
print(f"{'case':24s} {'E plain':>18s} {'dE two-level':>13s} {'max |dgrad|':>12s} {'cycles':>6s} {'refs':>4s} "
      f"{'t plain':>8s} {'t two':>8s}")
worst_e = worst_g = 0.
bad = []
for name in names:
    try:
        a = run(name, False)
        b = run(name, True)
    except Exception as err:   # report and go on with the other cases
        print(f'{name:24s} FAILED: {type(err).__name__}: {str(err)[:100]}')
        bad.append(name)
        continue
    de = b['e'] - a['e']
    dg = float(abs(b['grad'] - a['grad']).max())
    ok = a['converged'] and b['converged']
    worst_e = max(worst_e, abs(de))
    worst_g = max(worst_g, dg)
    if not ok:
        bad.append(name)
    print(f"{name:24s} {a['e']:18.10f} {de:13.2e} {dg:12.2e} {a['cycles']:6d} {b['cycles']:4d} "
          f"{a['t_scf']:8.2f} {b['t_scf']:8.2f}{'' if ok else '  NOT CONVERGED'}", flush=True)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps({'case': name, 'mol': args.mol, 'basis': args.basis, 'conv_tol': args.conv_tol,
                                'e_plain': a['e'], 'dE': de, 'max_dgrad': dg, 'cycles': a['cycles'],
                                'refs': b['cycles'], 't_plain': round(a['t_scf'], 3),
                                't_two': round(b['t_scf'], 3), 'converged': ok}) + '\n')
print(f'largest |dE| {worst_e:.2e}, largest |dgrad| {worst_g:.2e}, failed or unconverged: {bad or "none"}')
