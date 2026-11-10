import copy
from pyscf import gto, dft
from molecules import CAFFEINE
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
for label, setup in [('level 3 (default)', lambda mf: None),
                     ('atom_grid (99,590), nlc (50,194)', lambda mf: (setattr(mf.grids, 'atom_grid', (99, 590)),
                                                                     setattr(mf.nlcgrids, 'atom_grid', (50, 194))))]:
    mf = dft.RKS(mol, xc='wb97m-v')
    setup(mf)
    out = []
    for key in ('grids', 'nlcgrids'):
        # the lines of SCF.init_guess_by_pre_scf
        grids = copy.copy(getattr(mf, key)).reset(mol)
        grids.__dict__.pop('_fused_ao_data', None)
        grids.level = min(grids.level, mf.pre_scf_grids_level)
        grids.build()
        full = getattr(mf, key)
        full.build()
        out.append(f'{key}: pre-SCF {grids.weights.size} points, SCF {full.weights.size}')
    print(label, '->', '; '.join(out))
