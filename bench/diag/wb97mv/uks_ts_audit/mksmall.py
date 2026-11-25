import sys, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
def gen(smi, name, drop_oh):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=3); AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    x = m.GetConformer().GetPositions()
    skip = -1
    if drop_oh:
        o = m.GetSubstructMatches(Chem.MolFromSmarts('[OX2H]'))[0][0]
        skip = [n.GetIdx() for n in m.GetAtomWithIdx(o).GetNeighbors() if n.GetAtomicNum() == 1][0]
    rows = [(at.GetSymbol(), x[at.GetIdx()]) for at in m.GetAtoms() if at.GetIdx() != skip]
    with open(name, 'w') as f:
        f.write('%d\n%s\n' % (len(rows), smi + (' minus the OH hydrogen' if drop_oh else '')))
        for s, r in rows: f.write('%-2s %12.6f %12.6f %12.6f\n' % (s, *r))
    print(name, len(rows))
gen('COc1ccc(O)cc1', 'meophenol.xyz', False)
gen('COc1ccc(O)cc1', 'meophenoxyl.xyz', True)
