"""Geometries of the SCF guess and DIIS test set (It21): RDKit ETKDG embedding, then MMFF94 optimization.
Run with the tools venv:  .venv-tools/bin/python bench/geom/make_geoms.py
Writes bench/geom/<name>.xyz (comment line: charge spin SMILES)."""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

here = os.path.dirname(os.path.abspath(__file__))
# name: (SMILES, charge, 2S)
MOLECULES = {
    'aspirin': ('CC(=O)OC1=CC=CC=C1C(=O)O', 0, 0),
    'paracetamol': ('CC(=O)NC1=CC=C(C=C1)O', 0, 0),
    'ibuprofen': ('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 0, 0),
    'nicotine': ('CN1CCC[C@H]1C2=CN=CC=C2', 0, 0),
    'serotonin': ('C1=CC2=C(C=C1O)C(=CN2)CCN', 0, 0),
    'dopamine': ('C1=CC(=C(C=C1CCN)O)O', 0, 0),
    'glucose': ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O', 0, 0),
    'tryptophan': ('C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N', 0, 0),
    'adenine': ('C1=NC2=NC=NC(=C2N1)N', 0, 0),
    'naphthalene': ('c1ccc2ccccc2c1', 0, 0),
    'anthracene': ('c1ccc2cc3ccccc3cc2c1', 0, 0),
    'pyrene': ('c1cc2ccc3cccc4ccc(c1)c2c34', 0, 0),
    'azobenzene': ('c1ccc(cc1)/N=N/c2ccccc2', 0, 0),
    'porphine': ('C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5', 0, 0),
    'terthiophene': ('c1ccc(s1)-c2ccc(s2)-c3cccs3', 0, 0),
    'penicillin_g': ('CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C', 0, 0),
    'ala_ala': ('C[C@@H](C(=O)N[C@@H](C)C(=O)O)N', 0, 0),
    'retinal': ('CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C=O)/C)/C', 0, 0),
    'testosterone': ('C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C', 0, 0),
    'sucrose': ('OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O', 0, 0),
    'cholesterol': ('C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C',
                    0, 0),
    'benzoate': ('[O-]C(=O)c1ccccc1', -1, 0),
    'serotonin_h+': ('C1=CC2=C(C=C1O)C(=CN2)CC[NH3+]', 1, 0),
    'phenoxyl': ('[O]c1ccccc1', 0, 1),
    'benzyl': ('[CH2]c1ccccc1', 0, 1),
    'tempo': ('CC1(C)CCCC(C)(C)N1[O]', 0, 1),
    'naphthalene+': ('c1ccc2ccccc2c1', 1, 1),
    'diphenylcarbene_t': ('[C](c1ccccc1)c2ccccc2', 0, 2),
}

for name, (smi, charge, spin) in MOLECULES.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, randomSeed=7) == 0, name
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    conf = m.GetConformer()
    lines = [f'{m.GetNumAtoms()}', f'{charge} {spin} {smi}']
    for a in m.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f'{a.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}')
    with open(os.path.join(here, f'{name}.xyz'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{name:18s} {m.GetNumAtoms():3d} atoms  charge {charge:+d}  2S {spin}')
