'''The solvent's terms of an SMD (or PCM) Hessian by part, at one converged density (density-fitted PBE: the terms depend
only on the density, the basis and the surface): the solvent Hessian (nuclear, q.V and solver terms, SMD's CDS term by
finite differences), the solvent's part of H1 (analytical_grad_vmat), and the solvent's response of 3*natm CPHF-like
densities. Nested calls are indented. Usage: smd_hess_prof.py MOL [--basis def2-tzvpd] [--save FILE] [--compare FILE]'''
import argparse, time, numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.solvent.hessian import pcm as pcm_hess, smd as smd_hess
import molecules
p = argparse.ArgumentParser()
p.add_argument('mol'); p.add_argument('--basis', default='def2-tzvpd')
p.add_argument('--charge', type=int, default=0); p.add_argument('--save'); p.add_argument('--compare'); p.add_argument('--method', default='SMD')
a = p.parse_args()
T0 = time.perf_counter(); depth = [0]
def say(msg):
    print(f'[{time.perf_counter() - T0:6.0f} s] {msg}', flush=True)
def wrap(mod, name):
    fn = getattr(mod, name)
    def w(*args, **kw):
        depth[0] += 1; t = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            depth[0] -= 1; say('  ' * depth[0] + f'{name}: {time.perf_counter() - t:.1f} s')
    setattr(mod, name, w)
atom = getattr(molecules, a.mol.upper(), None) or a.mol
mol = gto.M(atom=atom, basis=a.basis, charge=a.charge, verbose=0)
mf = dft.RKS(mol, xc='pbe').density_fit()
mf = mf.SMD() if a.method == 'SMD' else mf.PCM()
if a.method == 'SMD':
    mf.with_solvent.solvent = 'water'
t = time.perf_counter(); e = mf.kernel()
pcm = mf.with_solvent
say(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads; {mol.natm} atoms, {mol.nao} AOs, '
    f'{len(pcm.surface["grid_coords"])} surface points; SCF {time.perf_counter() - t:.1f} s, E {e:.10f}')
for name in ('analytical_hess_nuc', 'analytical_hess_qv', 'analytical_hess_solver', 'analytical_grad_vmat',
             'get_dqsym_dx', 'get_dqsym_dx_fix_vgrids', 'get_dqsym_dx_fix_K_R', 'get_dvgrids',
             '_d2A_d2Sii_dot', '_d2_contract'):
    wrap(pcm_hess, name)
wrap(smd_hess, 'get_cds')
dm = mf.make_rdm1()
out = {}
t = time.perf_counter(); out['grad'] = pcm.grad(dm); say(f'solvent gradient: {time.perf_counter() - t:.1f} s')
t = time.perf_counter(); out['hess'] = pcm.hess(dm); say(f'solvent Hessian: {time.perf_counter() - t:.1f} s')
t = time.perf_counter(); out['dv'] = pcm_hess.analytical_grad_vmat(pcm, dm); say(f'solvent part of H1: {time.perf_counter() - t:.1f} s')
nset = 3 * mol.natm
rng = numpy.random.default_rng(1)
dms = rng.standard_normal((nset, mol.nao, mol.nao)) * 0.01   # not from the orbitals: their phases change between runs
dms = dms + dms.transpose(0, 2, 1)
t = time.perf_counter(); out['resp'] = pcm._B_dot_x(dms); say(f'solvent response of {nset} densities: {time.perf_counter() - t:.1f} s')
if a.save:
    numpy.savez(a.save, **out)
if a.compare:
    z = numpy.load(a.compare)
    for k in out:
        if k not in z:
            continue
        say(f'{k}: max|new - saved| {abs(out[k] - z[k]).max():.2e} (max|saved| {abs(z[k]).max():.2e})')
