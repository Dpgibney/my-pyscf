#!/bin/bash
# cmp_final.sh DIR: the nine UKS and nine RKS Hessians of hess_ref.py with the workspace's tree against those of
# another tree, DIR/href_frozen.npz and DIR/href_frozen_rks.npz (made there with
# `HESS_REF_RKS=0|1 PYTHONPATH=<tree> python hess_ref.py href_frozen[_rks].npz`): with the old screening of the
# two-electron derivative integrals (everything else new), and with the defaults. CMP_THREADS threads per run (2 runs
# at a time); with another thread count than the references' some SCFs end 1e-12 apart, and their Hessians 1e-8.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "$here/../../../.." && pwd)"
cd "$1" || exit 1
for dd in 0 1; do
  for r in 0 1; do
    sfx=$([ $r = 1 ] && echo _rks)
    HESS_REF_DD=$dd HESS_REF_RKS=$r PYTHONPATH=$W/pyscf OMP_NUM_THREADS=${CMP_THREADS:-2} $W/.venv/bin/python "$here/hess_ref.py" href_final${dd}$sfx.npz > href_final${dd}$sfx.out 2>&1 &
  done
  wait
done
$W/.venv/bin/python - <<'PY'
import numpy as np
for dd in (0, 1):
    print('old screening of the two-electron derivative integrals' if dd == 0 else 'defaults (two-density bound, 1e-13)')
    for sfx in ('', '_rks'):
        a = np.load(f'href_frozen{sfx}.npz'); b = np.load(f'href_final{dd}{sfx}.npz')
        for k in a.files:
            if k.endswith('_e'): continue
            print(f'  {sfx or "_uks":5s} {k:24s} dE {b[k+"_e"]-a[k+"_e"]:+.1e}  max |dH| {abs(a[k]-b[k]).max():.1e}')
PY
