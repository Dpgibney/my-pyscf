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
VERSION = 5  # bump when dump() changes, so that cached reference dumps are regenerated


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
    for xc in ('LDA,VWN', 'PBE', 'r2SCAN'):
        nelec, exc, vmat = ni.nr_uks(mol, grids, xc, (dm * .6, dm * .4))
        out[f'vxc_uks_{xc}'] = vmat
        out[f'exc_uks_{xc}'] = np.array([*nelec, exc])
    rng = np.random.default_rng(7)
    dm1 = rng.standard_normal((3,) + dm.shape) * 1e-2
    for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN'):
        out[f'fxc_rks_{xc}'] = ni.nr_rks_fxc(mol, grids, xc, dm, dm1, hermi=0)
    ao = numint.eval_ao(mol, grids.coords[:20000], deriv=1)
    dm_asym = dm + rng.standard_normal(dm.shape) * 1e-2   # exercises the hermi=0 paths
    for tag, d in (('', dm), ('_asym', dm_asym)):
        out[f'rho_lda{tag}'] = numint.eval_rho(mol, ao[0], d, xctype='LDA')
        out[f'rho_gga{tag}'] = numint.eval_rho(mol, ao, d, xctype='GGA')
        out[f'rho_mgga{tag}'] = numint.eval_rho(mol, ao, d, xctype='MGGA', with_lapl=False)
    out['vj'], out['vk'] = scf.hf.get_jk(mol, dm)
    out['vj_df'] = dft.RKS(mol).density_fit().get_j(mol, dm)
    # J/K of several dms (TDDFT/CPHF trial densities, UKS), from ERIs in memory and direct with the
    # SCF screening. Separate generator, so that the inputs of the other entries stay as they were.
    from pyscf.scf import _vhf
    rng2 = np.random.default_rng(11)
    dm_t = rng2.standard_normal((5,) + dm.shape) * 1e-2   # non-symmetric
    dm_s = np.array([dm * .6, dm * .4])
    eri = mol.intor('int2e', aosym='s8')
    out['vj_incore_t'], out['vk_incore_t'] = _vhf.incore(eri, dm_t, hermi=0)
    out['vj_incore_s'], out['vk_incore_s'] = _vhf.incore(eri, dm_s, hermi=1)
    out['vj_incore_1'], out['vk_incore_1'] = _vhf.incore(eri, dm, hermi=1)   # SCF: one symmetric dm
    del eri
    vhfopt = scf.RHF(mol).init_direct_scf()
    out['vj_direct_t'], out['vk_direct_t'] = scf.hf.get_jk(mol, dm_t, 0, vhfopt)
    out['vj_direct_s'], out['vk_direct_s'] = scf.hf.get_jk(mol, dm_s, 1, vhfopt)

    # A molecule large enough for the screened (sparse) grid kernels of block_loop
    from molecules import n_alkane
    mol = gto.M(atom=n_alkane(20), basis='def2-svp', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 2
    grids.build(with_non0tab=True)
    assert numint._sparse_enough(grids.non0tab)
    dm = dft.RKS(mol).get_init_guess(key='minao')
    for xc in ('LDA,VWN', 'PBE', 'r2SCAN'):
        nelec, exc, vmat = ni.nr_rks(mol, grids, xc, dm)
        out[f'sparse_vxc_rks_{xc}'] = vmat
        out[f'sparse_exc_rks_{xc}'] = np.array([nelec, exc])
    for xc in ('PBE', 'r2SCAN'):
        nelec, exc, vmat = ni.nr_uks(mol, grids, xc, (dm * .6, dm * .4))
        out[f'sparse_vxc_uks_{xc}'] = vmat
        out[f'sparse_exc_uks_{xc}'] = np.array([*nelec, exc])
    dm1 = rng.standard_normal((2,) + dm.shape) * 1e-2
    for xc in ('LDA,VWN', 'PBE', 'r2SCAN'):
        out[f'sparse_fxc_rks_{xc}'] = ni.nr_rks_fxc(mol, grids, xc, dm, dm1, hermi=0)
    # MO-tagged density as in SCF iterations (LDA and meta-GGA take eval_rho2 here) and the XC
    # term of the nuclear gradient
    from pyscf.grad import rks as rks_grad
    mf = dft.RKS(mol)
    mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
    dm_mo = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
    for xc in ('LDA,VWN', 'PBE', 'r2SCAN'):
        nelec, exc, vmat = ni.nr_rks(mol, grids, xc, dm_mo)
        out[f'sparse_vxc_rks_mo_{xc}'] = vmat
        out[f'sparse_exc_rks_mo_{xc}'] = np.array([nelec, exc])
        out[f'sparse_grad_vxc_{xc}'] = rks_grad.get_vxc(ni, mol, grids, xc, dm_mo)[1]
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
    ref_path = RESULTS / 'ref' / (f'kernels{VERSION}__{ref.tree}@{repo_shas().get(ref.tree)}__{variant}'
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
        print(f'  {"ok  " if good else "FAIL"} {key:24s} max|d|/scale = {rel:.2e}')
    print('ALL PASS' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--dump']:
        dump(sys.argv[2])
    else:
        main()
