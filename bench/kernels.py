"""Fixed-input kernel tests: identical inputs through the XC/J/K kernels in a candidate build and in the
unmodified reference build, compared array by array. Catches drift before it reaches SCF energies.

Usage: kernels.py CAND [--ref SPEC] [--threads N] [--rtol R]
  e.g. kernels.py new=pyscf
(internally each build runs ``kernels.py --dump OUT.npz`` in its own pinned environment)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def dump(path):
    from pyscf import gto, dft, scf
    from pyscf.dft import numint
    from molecules import CAFFEINE
    mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    dm = dft.RKS(mol).get_init_guess(key='minao')
    ni = numint.NumInt()
    out = {'grid_weights': grids.weights}
    for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN'):
        nelec, exc, vmat = ni.nr_rks(mol, grids, xc, dm)
        out[f'vxc_rks_{xc}'] = vmat
        out[f'exc_rks_{xc}'] = np.array([nelec, exc])
    nelec, exc, vmat = ni.nr_uks(mol, grids, 'PBE', (dm * .6, dm * .4))
    out['vxc_uks_PBE'] = vmat
    out['exc_uks_PBE'] = np.array([*nelec, exc])
    dm1 = np.random.default_rng(7).standard_normal((3,) + dm.shape) * 1e-2
    for xc in ('PBE', 'B3LYP'):
        out[f'fxc_rks_{xc}'] = ni.nr_rks_fxc(mol, grids, xc, dm, dm1, hermi=0)
    ao = numint.eval_ao(mol, grids.coords[:20000], deriv=1)
    out['rho_gga'] = numint.eval_rho(mol, ao, dm, xctype='GGA')
    out['rho_mgga'] = numint.eval_rho(mol, ao, dm, xctype='MGGA', with_lapl=False)
    out['vj'], out['vk'] = scf.hf.get_jk(mol, dm)
    out['vj_df'] = dft.RKS(mol).density_fit().get_j(mol, dm)
    np.savez(path, **out)


def main():
    from common import RESULTS, SCRATCH, Config, repo_shas, run_script
    ap = argparse.ArgumentParser()
    ap.add_argument('cand')
    ap.add_argument('--ref', default='ref=pyscf-base')
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--rtol', type=float, default=1e-11)
    args = ap.parse_args()
    cand, ref = Config.parse(args.cand), Config.parse(args.ref)

    env = '-'.join(f'{k}={v}' for k, v in sorted(ref.env.items())) or 'default'
    variant = ref.variant().replace(' ', '_').replace('=', '-')
    ref_path = RESULTS / 'ref' / (f'kernels__{ref.tree}@{repo_shas().get(ref.tree)}__{variant}'
                                  f'__{env}__t{args.threads}.npz')
    if not ref_path.exists():
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        run_script('kernels.py', ['--dump', str(ref_path)], ref, args.threads)
    cand_path = SCRATCH / f'kernels_{cand.label}.npz'
    run_script('kernels.py', ['--dump', str(cand_path)], cand, args.threads)

    r, c = np.load(ref_path), np.load(cand_path)
    ok = True
    print(f'candidate {cand.label}: {cand.tree} {cand.env} [{cand.variant()}]\n'
          f'reference {ref.label}: {ref.tree} {ref.env} [{ref.variant()}]  rtol={args.rtol:.0e}')
    for key in r.files:
        a, b = c[key], r[key]
        scale = max(1., float(np.max(np.abs(b))))
        rel = float(np.max(np.abs(a - b))) / scale if a.shape == b.shape else np.inf
        good = rel <= args.rtol
        ok &= good
        print(f'  {"ok  " if good else "FAIL"} {key:18s} max|d|/scale = {rel:.2e}')
    print('ALL PASS' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--dump']:
        dump(sys.argv[2])
    else:
        main()
