"""PCM results of a tree, saved (--save FILE) or compared with saved ones (--ref FILE), for checking a change to
pyscf.solvent against the code before it: SCF energies and cycles for C-PCM, COSMO, IEF-PCM and SS(V)PE
(formaldehyde/def2-SVP, B3LYP; a UKS cation; a Cartesian basis), their gradients, RHF Hessians for C-PCM and IEF-PCM,
TDA excitation energies with equilibrium solvation (PCM's response, _B_dot_x), and SMD's electrostatic energy and
Fock term at a fixed density. Run with PYTHONPATH set to the tree.
Usage: pcm_check.py --save ref.json | --ref ref.json"""
import argparse
import json
import numpy
from pyscf import gto, scf, dft
from pyscf.solvent import smd

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--save')
g.add_argument('--ref')
args = ap.parse_args()

ATOM = 'C 0 0 -0.53; O 0 0 0.67; H 0 0.94 -1.12; H 0 -0.94 -1.12'
mol = gto.M(atom=ATOM, basis='def2-svp', verbose=0)
res = {}


def scf_run(mf, key):
    cycles = []
    mf.callback = lambda envs: cycles.append(envs['cycle'])
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, key
    res[key + ' E'] = float(mf.e_tot)
    res[key + ' cycles'] = len(cycles)
    return mf


for method in ('C-PCM', 'COSMO', 'IEF-PCM', 'SS(V)PE'):
    mf = dft.RKS(mol, xc='b3lyp').PCM()
    mf.with_solvent.method = method
    scf_run(mf, f'RKS {method}')
    res[f'RKS {method} grad'] = mf.nuc_grad_method().kernel().tolist()
    if method in ('C-PCM', 'IEF-PCM'):
        td = mf.TDA(equilibrium_solvation=True)
        td.conv_tol = 1e-10
        res[f'RKS {method} TDA'] = td.kernel(nstates=3)[0].tolist()
        mfh = scf.RHF(mol).PCM()
        mfh.with_solvent.method = method
        scf_run(mfh, f'RHF {method}')
        res[f'RHF {method} hess'] = mfh.Hessian().kernel().tolist()

cation = gto.M(atom=ATOM, basis='def2-svp', charge=1, spin=1, verbose=0)
mf = dft.UKS(cation, xc='b3lyp').PCM()
scf_run(mf, 'UKS cation C-PCM')
res['UKS cation C-PCM grad'] = mf.nuc_grad_method().kernel().tolist()

cart = gto.M(atom=ATOM, basis='def2-svp', cart=True, verbose=0)
scf_run(dft.RKS(cart, xc='b3lyp').PCM(), 'RKS cart C-PCM')

# SMD's electrostatics only (its CDS term needs PySCF's optional SMD module): build() and the inherited _get_vind
dm = dft.RKS(mol, xc='b3lyp').run(conv_tol=1e-12).make_rdm1()
smdobj = smd.SMD(mol)
smdobj.solvent = 'water'
smdobj.build()
e_smd, v_smd = smdobj._get_vind(dm[None])
res['SMD electrostatic E'] = float(e_smd)
res['SMD electrostatic vmat'] = v_smd.tolist()

if args.save:
    with open(args.save, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'saved {len(res)} results to {args.save}')
else:
    with open(args.ref) as f:
        ref = json.load(f)
    worst = {}
    for key, val in ref.items():
        if key.endswith('cycles'):
            ok = res[key] == val
            print(f'{key:28s} {res[key]} against {val} {"" if ok else "  DIFFERENT"}')
            continue
        d = abs(numpy.asarray(res[key]) - numpy.asarray(val)).max()
        kind = key.split()[-1]
        worst[kind] = max(worst.get(kind, 0), d)
        print(f'{key:28s} max |diff| {d:.2e}')
    print('largest differences: ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
