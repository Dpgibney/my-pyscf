'''A transition-state search with geomeTRIC (transition=True) on the production workflow: wB97M-V, grids (99,590) and
(50,194), RIJCOSX (RI-J in df.aug_etb, seminumerical K) or exact J and K, the two-level SCF, optionally SMD. Then at the
saddle point found: a fresh calculation (the scanner's last energy and gradient against a new SCF object's), and a
Hessian (finite differences of the analytical gradient for RIJCOSX, which has no analytical Hessian; the analytical one
for exact J/K) with its harmonic frequencies, which should hold exactly one imaginary mode.
Usage: ts_rijcosx.py {hcn,sn2} [--rijcosx] [--solvent water|none] [--basis def2-tzvpd] [--exact-hessian] [--save PREFIX]
  hcn  HCN <-> HNC isomerization (neutral, 3 atoms)
  sn2  Cl- + CH3Cl -> ClCH3 + Cl- (charge -1, D3h saddle point; started off centre)
  --exact-hessian  with --rijcosx, also the exact-J/K analytical Hessian at the RIJCOSX saddle point'''
import argparse, time, numpy
import pyscf
from pyscf import gto, dft, df, lib, grad, hessian  # noqa: F401  (grad, hessian: Gradients() and Hessian())
from pyscf.hessian import thermo
from pyscf.geomopt import geometric_solver
import geometric

GEOMS = {
    # H bridging C and N, angle H-C-N 65 degrees
    'hcn': (0, '''C 0.000 0.000 0.000
                  N 1.190 0.000 0.000
                  H 0.486 1.042 0.000'''),
    # Cl-C-Cl along z with C 0.10 A off centre
    'sn2': (-1, '''C  0.000  0.000  0.100
                   Cl 0.000  0.000  2.320
                   Cl 0.000  0.000 -2.320
                   H  1.070  0.000  0.100
                   H -0.535  0.927  0.100
                   H -0.535 -0.927  0.100'''),
}
p = argparse.ArgumentParser()
p.add_argument('system', choices=GEOMS); p.add_argument('--rijcosx', action='store_true')
p.add_argument('--solvent', default='water'); p.add_argument('--basis', default='def2-tzvpd')
p.add_argument('--exact-hessian', action='store_true'); p.add_argument('--save')
a = p.parse_args()
T0 = time.perf_counter()


def say(msg):
    print(f'>>> [{time.perf_counter() - T0:6.0f} s] {msg}', flush=True)


def make_mf(mol, rijcosx):
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    if rijcosx:
        mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
    if a.solvent != 'none':
        mf = mf.SMD()
        mf.with_solvent.solvent = a.solvent
    mf.two_level = True
    return mf


def bonds(mol):
    '''the distances below 2.6 A, A'''
    r = mol.atom_coords(unit='A')
    out = []
    for i in range(mol.natm):
        for j in range(i):
            d = numpy.linalg.norm(r[i] - r[j])
            if d < 2.6:
                out.append(f'{mol.atom_symbol(i)}{i}-{mol.atom_symbol(j)}{j} {d:.4f}')
    return ', '.join(out)


def frequencies(mol, h, label):
    '''harmonic analysis of the [natm,natm,3,3] Hessian h; returns the wavenumbers, imaginary ones negative'''
    res = thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    w = res['freq_wavenumber']
    nimag = int((w < 0).sum())
    mode = res['norm_mode'][0]          # the lowest mode, [natm,3]
    amp = numpy.linalg.norm(mode, axis=1)
    say(f'{label}: {nimag} imaginary mode(s); wavenumbers (cm-1) ' + ' '.join(f'{x:.1f}' for x in w)
        + f'; lowest mode by atom (|displacement|) ' + ' '.join(f'{mol.atom_symbol(i)}{i} {amp[i]:.2f}'
                                                              for i in range(mol.natm)))
    return w, mode


charge, atoms = GEOMS[a.system]
mol = gto.M(atom=atoms, basis=a.basis, charge=charge, verbose=4)
mf = make_mf(mol, a.rijcosx)
method = 'RIJCOSX (aug_etb)' if a.rijcosx else 'exact J and K'
say(f'pyscf {pyscf.__file__}, geomeTRIC {geometric.__version__}, {lib.num_threads()} threads; {a.system}: '
    f'{mol.natm} atoms, {mol.nao} AOs, charge {charge}; wB97M-V/{a.basis}, {method}, two-level SCF, '
    f'solvent {a.solvent}; start {bonds(mol)}')

LAST = {}


def callback(env):
    LAST.update(coords=env['coords'].copy(), e=env['energy'], g=env['gradients'].copy(),
                conv=env['g_scanner'].converged)
    say(f'gradient {env["self"].cycle}: E = {env["energy"]:.10f}, max |g| {abs(env["gradients"]).max():.2e}, '
        f'SCF converged {env["g_scanner"].converged}')


# hessian=True: the analytical Hessian at the start for exact J/K. For RIJCOSX it is not available, PySCF warns and
# geomeTRIC computes it by finite differences of the gradient (its default for transition=True), which is what a
# user who asks for it gets.
t = time.perf_counter()
conv, mol_ts = geometric_solver.kernel(mf, transition=True, hessian=True, callback=callback, maxsteps=60)
t_opt = time.perf_counter() - t
say(f'TS optimization: converged {conv}, {t_opt:.0f} s, {mf.mol is mol and "method.mol unchanged"}; '
    f'{bonds(mol_ts)}')
assert numpy.allclose(LAST['coords'], mol_ts.atom_coords()), 'the last gradient was not at the final geometry'

# The scanner's last result against a new SCF object at the saddle point
mol_ts.verbose = 4
mf2 = make_mf(mol_ts, a.rijcosx)
e2 = mf2.kernel()
g2 = mf2.Gradients().kernel()
say(f'fresh calculation at the saddle point: E = {e2:.10f} (scanner {LAST["e"]:.10f}, difference {e2 - LAST["e"]:.1e}); '
    f'max |g| {abs(g2).max():.2e} Eh/Bohr, gradient difference from the scanner {abs(g2 - LAST["g"]).max():.1e}')

t = time.perf_counter()
if a.rijcosx:
    # central differences of the gradient, the step geomeTRIC uses (1e-3 Bohr), the SCF converged tighter
    mf2.conv_tol = 1e-11
    gs = mf2.Gradients().as_scanner()
    x0 = mol_ts.atom_coords()
    n = mol_ts.natm
    h = numpy.zeros((n, 3, n, 3))
    step = 1e-3
    for i in range(n):
        for k in range(3):
            gp = []
            for s in (step, -step):
                x = x0.copy()
                x[i, k] += s
                gp.append(gs(mol_ts.set_geom_(x, unit='Bohr', inplace=False))[1])
            h[i, k] = (gp[0] - gp[1]) / (2 * step)
        say(f'finite-difference Hessian: atom {i + 1} of {n} done')
    asym = abs(h - h.transpose(2, 3, 0, 1)).max()
    h = .5 * (h + h.transpose(2, 3, 0, 1))
    h = h.transpose(0, 2, 1, 3)          # [natm,natm,3,3]
    say(f'finite-difference Hessian: {time.perf_counter() - t:.0f} s, largest asymmetry {asym:.1e} Eh/Bohr^2')
    w, mode = frequencies(mol_ts, h, 'RIJCOSX finite-difference Hessian at the RIJCOSX saddle point')
    if a.exact_hessian:
        t = time.perf_counter()
        mfx = make_mf(mol_ts, False)
        mfx.kernel()
        gx = mfx.Gradients().kernel()
        hx = mfx.Hessian().kernel()
        say(f'exact-J/K analytical Hessian at the RIJCOSX saddle point: {time.perf_counter() - t:.0f} s; exact max |g| '
            f'there {abs(gx).max():.2e}; largest difference from the RIJCOSX Hessian {abs(hx - h).max():.1e} Eh/Bohr^2')
        wx, _ = frequencies(mol_ts, hx, 'exact-J/K analytical Hessian at the RIJCOSX saddle point')
else:
    h = mf2.Hessian().kernel()
    say(f'analytical Hessian: {time.perf_counter() - t:.0f} s')
    w, mode = frequencies(mol_ts, h, 'exact-J/K analytical Hessian at the exact saddle point')

if a.save:
    numpy.savez(a.save + '.npz', coords=mol_ts.atom_coords(), e=e2, g=g2, hess=h, freq=w, mode=mode, conv=conv,
                t_opt=t_opt)
    with open(a.save + '.xyz', 'w') as f:
        f.write(f'{mol_ts.natm}\n{a.system} saddle point, {method}, E = {e2:.10f}\n')
        for i, r in enumerate(mol_ts.atom_coords(unit='A')):
            f.write(f'{mol_ts.atom_symbol(i):2s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n')
say(f'total {time.perf_counter() - T0:.0f} s')
