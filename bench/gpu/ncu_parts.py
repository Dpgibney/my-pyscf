"""One J, K, J+K or XC build of GPU4PySCF under Nsight Compute (node1). The SCF runs first, unprofiled; one build at
the converged density is then bracketed with cudaProfilerStart/Stop, so run it as
    ncu --profile-from-start off --csv --page raw --metrics ... python ncu_parts.py --part k ...
For the J/K builders it records, in launch order, the angular-momentum class of each call of the integral launcher
(RYS_build_k, RYS_build_jk, MD_build_j) and writes them to --labels, so ncu's per-launch rows can be grouped by
class (ncu_summary.py).
Part jgrad is the J engine's gradient of a pure functional (jk_energy_per_atom with k_factor 0: MD_build_j_ip1,
part 24). Part grad is the gradient's two-electron energy derivatives (jk_energy_per_atom with B3LYP's factors,
RYS_per_atom_jk_ip1); part ejk2 the Hessian's second derivatives (_partial_ejk_ip2 with B3LYP's factors,
RYS_per_atom_jk_ip2_type12 and _type3, labelled 't12' and 't3'). --dm DM.npy (jk_at_dm.py --save-dm) takes a saved density in the original AO basis instead of
running the SCF.
Part cphfjk is the Hessian CPHF's one-pass J+K (_VHFOpt.get_jk, RYS_build_jk, as hessian.rhf._get_jk_mo runs it) at
the response densities of --dms DMS.npz (cphf_jk_dms.py); --dm is still needed for the SCF-free setup.
Parts xc (the SCF's nr_rks) and gradxc (the gradient's XC term, grad.rks.get_exc) take the SCF's density with its
orbitals attached, as the SCF and the gradient pass it (the density pass then works from the orbitals), so they
always run the SCF; --dm is ignored there.
Usage: ncu_parts.py --part k|j|jk|xc|gradxc|grad|jgrad|ejk2|cphfjk [--xc b3lyp] [--basis def2-svp] [--mol caffeine]
                    [--labels FILE] [--dm DM.npy] [--dms DMS.npz]"""
import argparse
import ctypes
import json
import os
import sys
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.scf import jk, j_engine  # noqa: E402
from gpu4pyscf.grad import rhf as rhf_grad, rks as rks_grad  # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--part', required=True, choices=['k', 'j', 'jk', 'xc', 'gradxc', 'grad', 'jgrad', 'ejk2', 'cphfjk'])
ap.add_argument('--dms', default=None, help='part cphfjk: the response densities (cphf_jk_dms.py)')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--labels', default=None)
ap.add_argument('--dm', default=None)
args = ap.parse_args()

atom = n_alkane(int(args.mol[1:])) if args.mol[0] == 'c' and args.mol[1:].isdigit() else \
    {'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol]
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
if args.dm and args.part not in ('xc', 'gradxc'):
    dm = cupy.asarray(numpy.load(args.dm))
else:
    mf.kernel()
    dm = mf.make_rdm1()   # with mo_coeff and mo_occ attached
    if args.part not in ('xc', 'gradxc'):
        dm = cupy.asarray(dm)
g = mf.nuc_grad_method()
if args.part == 'cphfjk':
    # the Hessian CPHF's one-pass J+K (hessian.rhf._get_jk_mo) at its response densities
    kopt = jk._VHFOpt(mol, mf.direct_scf_tol).build()
    cphf_dms = kopt.apply_coeff_C_mat_CT(cupy.asarray(numpy.load(args.dms)['dms']))
build = {'k': lambda: mf.get_k(mol, dm, 1), 'j': lambda: mf.get_j(mol, dm, 1),
         'jk': lambda: mf.get_jk(mol, dm, 1),
         'xc': lambda: mf._numint.nr_rks(mol, mf.grids, mf.xc, dm),
         'gradxc': lambda: rks_grad.get_exc(mf._numint, mol, mf.grids, mf.xc, dm),
         'grad': lambda: g.jk_energy_per_atom(dm, 1., .2),
         'jgrad': lambda: g.jk_energy_per_atom(dm, 1., 0.),
         'ejk2': lambda: rhf_hess._partial_ejk_ip2(mol, dm, j_factor=1., k_factor=.2),
         'cphfjk': lambda: kopt.get_jk(cphf_dms, 1, 0)}[args.part]
build()  # screening data, module loading, memory pool

labels = []


def record(clib, name, opt, suffix=''):
    '''Wrap the launcher clib.name so that each call appends its class (and suffix) to labels'''
    orig = getattr(clib, name)
    smol = opt.sorted_mol
    loc = numpy.append(0, numpy.cumsum(smol.l_ctr_counts))
    symb = [lib.param.ANGULAR[l] for l in smol.uniq_l_ctr[:, 0]]

    def call(*a):
        shls = next(x for x in a if isinstance(x, ctypes.Array) and len(x) == 8)
        g = [int(numpy.searchsorted(loc, shls[2 * n], side='right')) - 1 for n in range(4)]
        labels.append({'launcher': name, 'class': f'({symb[g[0]]}{symb[g[1]]}|{symb[g[2]]}{symb[g[3]]}){suffix}',
                       'groups': g})
        return orig(*a)
    setattr(clib, name, call)


opt_k = next((v for v in mf._opt_gpu.values() if v is not None), None)
opt_j = next((v for v in mf._opt_jengine.values() if v is not None), None)
if args.part == 'k':
    record(jk.libvhf_rys, 'RYS_build_k', opt_k)
elif args.part == 'jk':
    record(jk.libvhf_rys, 'RYS_build_jk', opt_k)
elif args.part == 'cphfjk':
    record(jk.libvhf_rys, 'RYS_build_jk', kopt)
elif args.part == 'j':
    record(j_engine.libvhf_md, 'MD_build_j', opt_j)
elif args.part == 'jgrad':
    record(j_engine.libvhf_md, 'MD_build_j_ip1', opt_j)
elif args.part == 'grad':
    record(rhf_grad.libvhf_rys, 'RYS_per_atom_jk_ip1', opt_k)
elif args.part == 'ejk2':
    opt_h = rhf_hess._VHFOpt(mol).build()   # the screening _partial_ejk_ip2 builds for itself
    record(rhf_hess.libvhf_rys, 'RYS_per_atom_jk_ip2_type12', opt_h, ' t12')
    record(rhf_hess.libvhf_rys, 'RYS_per_atom_jk_ip2_type3', opt_h, ' t3')

cupy.cuda.Device().synchronize()
cupy.cuda.runtime.profilerStart()
build()
cupy.cuda.Device().synchronize()
cupy.cuda.runtime.profilerStop()
print(f'{args.mol} {args.basis} {args.xc} part {args.part}: nao {mol.nao}, {len(labels)} launcher calls',
      file=sys.stderr)
if args.labels:
    with open(args.labels, 'w') as f:
        json.dump(labels, f)
