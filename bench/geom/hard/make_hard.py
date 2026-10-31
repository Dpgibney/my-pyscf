"""Geometries of harder SCF cases (It21): small-gap conjugated systems (RDKit ETKDG + MMFF94), and an
octahedral [Fe(H2O)6]2+ built by hand. Run with .venv-tools/bin/python."""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

here = os.path.dirname(os.path.abspath(__file__))
for name, smi in {'pentacene': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
                  'coronene': 'c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67',
                  'beta_carotene': 'CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C=C/C=C(\\C)/C=C/C=C(\\C)/C=C/C2=C(CCCC2(C)C)C)/C)/C',
                  }.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, randomSeed=7) == 0, name
    AllChem.MMFFOptimizeMolecule(m, maxIters=5000)
    conf = m.GetConformer()
    lines = [f'{m.GetNumAtoms()}', f'0 0 {smi}']
    for a in m.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f'{a.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}')
    open(os.path.join(here, f'{name}.xyz'), 'w').write('\n'.join(lines) + '\n')
    print(name, m.GetNumAtoms())

# [Fe(H2O)6]2+, high spin (2S = 4): Fe-O 2.12 A, O-H 0.97 A, H-O-H 105 deg, water planes along the axes
lines = []
r_feo, r_oh, ang = 2.12, 0.97, np.radians(105.)
atoms = [('Fe', np.zeros(3))]
axes = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
perp = [np.array(v, float) for v in ((0, 1, 0), (0, 0, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0), (0, 1, 0))]
for a, p in zip(axes, perp):
    o = a * r_feo
    atoms.append(('O', o))
    for sgn in (1, -1):
        h = o + r_oh * (np.cos(ang / 2) * a + sgn * np.sin(ang / 2) * p)
        atoms.append(('H', h))
lines = [f'{len(atoms)}', '2 4 [Fe(H2O)6]2+ high spin']
lines += [f'{s:2s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}' for s, x in atoms]
open(os.path.join(here, 'fe_h2o6.xyz'), 'w').write('\n'.join(lines) + '\n')
print('fe_h2o6', len(atoms))
