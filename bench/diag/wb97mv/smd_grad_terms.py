'''The solvent terms of an SMD gradient alone, from a saved density matrix (PREFIX_dm.npy of prod_scf_grad.py): the
surface, the reaction field of the density, then grad_qv, grad_solver, grad_nuc and the CDS term, each with its wall
time, the resident memory and its peak since the previous line.
Usage: smd_grad_terms.py XYZ DM.npy [--charge 0] [--solvent water] [--basis def2-tzvpd] [--save OUT.npy]'''
import argparse, time, numpy
import pyscf
from pyscf import gto, dft, lib
from pyscf.solvent import smd
from pyscf.solvent.grad import pcm as pcm_grad, smd as smd_grad
p = argparse.ArgumentParser()
p.add_argument('xyz'); p.add_argument('dm'); p.add_argument('--charge', type=int, default=0)
p.add_argument('--solvent', default='water'); p.add_argument('--basis', default='def2-tzvpd'); p.add_argument('--save')
a = p.parse_args()
T0 = time.perf_counter()


def memory():
    d = {}
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(('VmRSS', 'VmHWM')):
                d[line[:5]] = int(line.split()[1]) / 1048576
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
    except OSError:
        pass
    return d['VmRSS'], d['VmHWM']


def step(label, fn):
    t = time.perf_counter()
    r = fn()
    rss, peak = memory()
    print(f'>>> [{time.perf_counter() - T0:5.0f} s] {label}: {time.perf_counter() - t:.1f} s; resident {rss:.1f} GB, '
          f'peak in this step {peak:.1f} GB', flush=True)
    return r


mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, verbose=0)
dm = numpy.load(a.dm)
assert dm.shape == (mol.nao, mol.nao)
mf = dft.RKS(mol, xc='wb97m-v').SMD()
solv = mf.with_solvent
solv.solvent = a.solvent
print(f'pyscf {pyscf.__file__}, {lib.num_threads()} threads; {mol.natm} atoms, {mol.nao} AOs, tr(DS) '
      f'{numpy.einsum("ij,ji", dm, mol.intor("int1e_ovlp")):.6f}', flush=True)
memory()
step('surface', solv.build)
n = len(solv.surface['charge_exp'])
print(f'    {n} surface points, {n * n * 8 / 1e9:.2f} GB for each [n,n] matrix', flush=True)
step('reaction field of the density (potential on the surface, factorization of K)', lambda: solv._get_vind(dm))
print(f'    E_diel = {float(solv._intermediates["e_solvent"] if "e_solvent" in solv._intermediates else solv.e):.12f} Eh',
      flush=True) if False else None
de_qv = step('grad_qv (derivative integrals with the surface charges)', lambda: pcm_grad.grad_qv(solv, dm))
de_solver = step('grad_solver (derivatives of the surface matrices)', lambda: smd_grad.grad_solver(solv, dm))
de_nuc = step('grad_nuc', lambda: pcm_grad.grad_nuc(solv, dm))
de_cds = step('CDS term', lambda: smd.get_cds_legacy(solv)[1])
de = de_qv + de_solver + de_nuc + de_cds
print(f'>>> solvent gradient: max |g| {abs(de).max():.3e} Eh/Bohr; by term: qv {abs(de_qv).max():.2e}, solver '
      f'{abs(de_solver).max():.2e}, nuc {abs(de_nuc).max():.2e}, CDS {abs(de_cds).max():.2e}; total {time.perf_counter() - T0:.0f} s',
      flush=True)
if a.save:
    numpy.save(a.save, numpy.array([de_qv, de_solver, de_nuc, de_cds]))
