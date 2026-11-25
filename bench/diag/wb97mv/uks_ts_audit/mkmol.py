'''FMNH2 (1,5-dihydroflavin mononucleotide, neutral) and HNCO embedded separately (ETKDG + MMFF), HNCO placed 3.2 A
above the pyrimidine ring, the complex relaxed with MMFF, then the N1 hydrogen removed: the neutral flavin
semiquinone radical FMNH. with an HNCO beside it: 57 atoms, doublet.'''
import sys
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

def embed(smi, n):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 7
    AllChem.EmbedMultipleConfs(m, n, ps)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=4000)
    cid = int(np.argmin([r[1] for r in res]))
    out = Chem.Mol(m, confId=cid)
    return out, np.array(m.GetConformer(cid).GetPositions())

fl, xf = embed('Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)C1=C(N2)C(=O)NC(=O)N1', 30)
hn, xh = embed('N=C=O', 1)
patt = Chem.MolFromSmarts('[NX3;H1;R](C(=O))C=C')
n1 = fl.GetSubstructMatches(patt)[0][0]
ring = [a for r in fl.GetRingInfo().AtomRings() if n1 in r for a in r]
c = xf[ring].mean(0)
u, s, vt = np.linalg.svd(xf[ring] - c)
normal = vt[2]
# the side of the ring with fewer atoms nearby
side = max((1, -1), key=lambda sg: np.linalg.norm(xf - (c + sg * 3.2 * normal), axis=1).min())
xh = xh - xh.mean(0) + c + side * 3.2 * normal
m = Chem.CombineMols(fl, hn)
conf = Chem.Conformer(m.GetNumAtoms())
for i, p in enumerate(np.vstack([xf, xh])):
    conf.SetAtomPosition(i, Point3D(*p))
m.RemoveAllConformers(); m.AddConformer(conf)
Chem.SanitizeMol(m)
print('MMFF of the complex:', AllChem.MMFFOptimizeMolecule(m, maxIters=4000))
x = np.array(m.GetConformer().GetPositions())
hrm = [a.GetIdx() for a in m.GetAtomWithIdx(n1).GetNeighbors() if a.GetAtomicNum() == 1][0]
keep = [i for i in range(m.GetNumAtoms()) if i != hrm]
sym = [m.GetAtomWithIdx(i).GetSymbol() for i in keep]
x = x[keep]
with open(sys.argv[1], 'w') as f:
    f.write('%d\nFMNH radical (N5-H flavin mononucleotide semiquinone, neutral) + HNCO; MMFF geometry of FMNH2 + HNCO, the N1 hydrogen removed\n' % len(keep))
    for s_, p in zip(sym, x):
        f.write('%-2s %12.6f %12.6f %12.6f\n' % (s_, *p))
print(Counter(sym), len(sym))
d = np.linalg.norm(x[:,None] - x[None], axis=2) + np.eye(len(x)) * 9
nf = fl.GetNumAtoms() - 1
print('shortest distance %.3f A; flavin-HNCO shortest %.2f A; extent %s A' % (d.min(), d[:nf, nf:].min(), np.round(x.max(0) - x.min(0), 1)))
