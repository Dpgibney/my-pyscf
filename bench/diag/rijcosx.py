"""RIJCOSX-like SCFs in PySCF against exact and RIJK: one SCF of --mol/--basis/--xc with --method
  exact    the tree's J/K (direct, screened ERIs),
  rijk     density-fitted J and K (mf.density_fit, --auxbasis or PySCF's predefined JK-fit basis),
  rijcosx  density-fitted J and seminumerical K with P-junction screening (sgx.sgx_fit(mf, pjs=True)), set up as
           ORCA's RIJCOSX by default: J fitted in def2-universal-jfit (ORCA's def2/J), the SGX grid level 0 in the
           first cycles and 1 after the switch (ORCA's GRIDX1 and GRIDX2), then the exchange energy of the converged
           density once more on level 2 (GRIDX3) as ORCA does (--final-level -1 skips it).
Prints the SCF's time split into J, K, XC and the rest, its cycles and energy, and a line per K build (grid points,
full or incremental, seconds). Memory and disk: a thread samples the process's RSS and the bytes allocated under
the run's own scratch directory (PySCF's TMPDIR, made under --scratch, which should be on a disk: /tmp is a tmpfs
here, whose files take RAM) every 0.1 s, by phase (setup, SCF, final step); the kernel's peak RSS (ru_maxrss) is
reported too. --ref FILE: a JSON file of exact energies by case, to print the error against exact and record this
run's energy (an exact run writes it). --json FILE appends the record. Run with PYTHONPATH set to the tree.
Usage: rijcosx.py [--mol caffeine|c20h42|c26h54|benzene] [--basis def2-tzvp] [--xc b3lyp] [--method rijcosx]
       [--auxbasis B] [--levels I F] [--final-level L] [--etol E] [--vtol V] [--rebuild N] [--xc-level L]
       [--scratch DIR] [--ref FILE] [--json FILE]"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402


class Monitor(threading.Thread):
    '''RSS of this process and the bytes allocated under a directory, sampled every dt seconds, peaks by phase'''
    def __init__(self, path, dt=0.1):
        super().__init__(daemon=True)
        self.path = path
        self.dt = dt
        self.phase = 'setup'
        self.peak = {}
        self.done = threading.Event()
        self.page = os.sysconf('SC_PAGE_SIZE')

    def disk(self):
        n = 0
        for root, _, files in os.walk(self.path):
            for f in files:
                try:
                    n += os.stat(os.path.join(root, f)).st_blocks * 512
                except OSError:
                    pass
        return n

    def sample(self):
        with open('/proc/self/statm') as f:
            rss = int(f.read().split()[1]) * self.page
        disk = self.disk()
        r, d = self.peak.get(self.phase, (0, 0))
        self.peak[self.phase] = (max(r, rss), max(d, disk))

    def run(self):
        while not self.done.wait(self.dt):
            self.sample()

    def set_phase(self, phase):
        self.sample()
        self.phase = phase

    def stop(self):
        self.sample()
        self.done.set()
        self.join()

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--method', choices=['exact', 'rijk', 'rijcosx'], default='rijcosx')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--levels', type=int, nargs=2, default=(0, 1), metavar=('I', 'F'))
ap.add_argument('--final-level', type=int, default=2)
ap.add_argument('--etol', default='auto')
ap.add_argument('--vtol', default='auto')
ap.add_argument('--rebuild', type=int, default=5)
ap.add_argument('--pair-tol', type=float, default=None)
ap.add_argument('--switch-thrd', type=float, default=None)
ap.add_argument('--xc-level', type=int, default=None)
ap.add_argument('--conv-tol', type=float, default=1e-9)
ap.add_argument('--ref', default=None)
ap.add_argument('--json', default=None)
ap.add_argument('--tag', default='')
ap.add_argument('--scratch', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                    'scratch', 'rijcosx-tmp'))
args = ap.parse_args()

os.makedirs(args.scratch, exist_ok=True)
tmpdir = tempfile.mkdtemp(prefix='run-', dir=args.scratch)
lib.param.TMPDIR = tmpdir
mon = Monitor(tmpdir)
mon.start()


def molecule(name):
    if name == 'caffeine':
        return CAFFEINE
    if name == 'benzene':
        return BENZENE
    if name.startswith('c') and name.endswith('h' + str(2 * int(name[1:name.index('h')]) + 2)):
        return n_alkane(int(name[1:name.index('h')]))
    raise ValueError(name)


mol = gto.M(atom=molecule(args.mol), basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = args.conv_tol
if args.xc_level is not None:
    mf.grids.level = args.xc_level

T = {'j': 0., 'k': 0., 'jk': 0., 'veff': 0., 'xc': 0.}
in_final = [False]
kbuilds = []
jbuilds = []
cycle_t = []

if args.method == 'exact':
    get_jk0 = mf.get_jk

    def get_jk(*a, **k):
        t = time.perf_counter()
        try:
            return get_jk0(*a, **k)
        finally:
            T['jk'] += time.perf_counter() - t
    mf.get_jk = get_jk
elif args.method == 'rijk':
    mf = mf.density_fit(auxbasis=args.auxbasis)
    from pyscf.df import df_jk
    get_jk0 = df_jk.get_jk

    def get_jk(*a, **k):
        t = time.perf_counter()
        try:
            return get_jk0(*a, **k)
        finally:
            T['jk'] += time.perf_counter() - t
    df_jk.get_jk = get_jk
else:
    from pyscf import sgx
    from pyscf.sgx import sgx_jk
    from pyscf.df import df_jk
    auxbasis = args.auxbasis or ('def2-universal-jfit' if args.basis.lower().startswith('def2') else None)
    mf = sgx.sgx_fit(mf, auxbasis=auxbasis, pjs=True)
    sdf = mf.with_df
    sdf.grids_level_i, sdf.grids_level_f = args.levels
    if args.switch_thrd is not None:
        sdf.grids_switch_thrd = args.switch_thrd
    sdf.sgx_tol_energy = args.etol if args.etol == 'auto' else (None if args.etol == 'none' else float(args.etol))
    sdf.sgx_tol_potential = args.vtol if args.vtol == 'auto' else (None if args.vtol == 'none' else float(args.vtol))
    mf.rebuild_nsteps = args.rebuild
    if args.pair_tol is not None:
        sdf.sgx_tol_pair = args.pair_tol
    get_j0 = df_jk.get_j
    get_k0 = sgx_jk.get_k_only

    def get_j(*a, **k):
        t = time.perf_counter()
        try:
            return get_j0(*a, **k)
        finally:
            dt = time.perf_counter() - t
            T['j'] += dt
            jbuilds.append(dt)

    def get_k_only(s, dm, *a, **k):
        incr = s._full_dm is not None
        t = time.perf_counter()
        try:
            return get_k0(s, dm, *a, **k)
        finally:
            dt = time.perf_counter() - t
            if not in_final[0]:
                T['k'] += dt
                kbuilds.append((s.grids.weights.size, 'incr' if incr else 'full', dt))
    df_jk.get_j = get_j
    sgx_jk.get_k_only = get_k_only

last = {}
get_veff0 = mf.get_veff
nr_rks0 = mf._numint.nr_rks


def nr_rks(*a, **k):
    t = time.perf_counter()
    try:
        return nr_rks0(*a, **k)
    finally:
        T['xc'] += time.perf_counter() - t


mf._numint.nr_rks = nr_rks


def get_veff(*a, **k):
    t = time.perf_counter()
    v = get_veff0(*a, **k)
    T['veff'] += time.perf_counter() - t
    cycle_t.append(time.perf_counter())
    last['vhf'] = v
    return v


mf.get_veff = get_veff
mon.set_phase('scf')
t0 = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t0
dm = mf.make_rdm1()
e_tot = mf.e_tot

e_final = None
t_final = 0.
if args.method == 'rijcosx' and args.final_level >= 0:
    # ORCA's last step: the exchange energy of the converged density on the larger grid, once
    mon.set_phase('final')
    in_final[0] = True
    t1 = time.perf_counter()
    sdf = mf.with_df
    sdf.build(level=args.final_level)
    hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[2]
    vk = sdf.get_jk(dm, hermi=1, with_j=False, with_k=True, direct_scf_tol=mf.direct_scf_tol)[1]
    e_k_scf = -.25 * numpy.einsum('ij,ji', dm, last['vhf'].vk).real
    e_k_fin = -.25 * hyb * numpy.einsum('ij,ji', dm, vk).real
    e_final = e_tot - e_k_scf + e_k_fin
    t_final = time.perf_counter() - t1
    kbuilds.append((sdf.grids.weights.size, 'final step (grid + K)', t_final))

mon.stop()
shutil.rmtree(tmpdir, ignore_errors=True)
maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
mem = {ph: dict(rss_gb=r / 1e9, disk_gb=d / 1e9) for ph, (r, d) in mon.peak.items()}
jk = T['jk'] if args.method != 'rijcosx' else T['j'] + T['k']
xc_rest = T['veff'] - jk - T['xc']
case = f'{args.mol}/{args.basis}/{args.xc}' + (f'/xc{args.xc_level}' if args.xc_level is not None else '')
ref = {}
if args.ref and os.path.exists(args.ref):
    with open(args.ref) as f:
        ref = json.load(f)
if args.method == 'exact' and args.ref:
    ref[case] = e_tot
    with open(args.ref, 'w') as f:
        json.dump(ref, f, indent=1)
e_best = e_final if e_final is not None else e_tot
err = e_best - ref[case] if case in ref and args.method != 'exact' else None

label = args.method
if args.method == 'rijcosx':
    label += f' aux={auxbasis} levels={args.levels[0]}->{args.levels[1]} final={args.final_level} ' \
             f'etol={args.etol} vtol={args.vtol} rebuild={args.rebuild}' + \
             (f' pair_tol={args.pair_tol:.0e}' if args.pair_tol is not None else '')
elif args.method == 'rijk':
    label += f' aux={mf.with_df.auxbasis}'
print(f'{case} nao {mol.nao} {lib.num_threads()} threads {label} {args.tag}')
print(f'  SCF {t_scf:.1f} s, {mf.cycles} cycles, converged {mf.converged}, E {e_tot:.10f}'
      + (f'; final level {args.final_level}: {e_final:.10f} (+{t_final:.1f} s)' if e_final is not None else ''))
if args.method == 'rijcosx':
    print(f'  J {T["j"]:.1f} s, K {T["k"]:.1f} s, XC {T["xc"]:.1f} s, rest of get_veff {xc_rest:.1f} s, outside '
          f'get_veff {t_scf - T["veff"]:.1f} s')
    print('    J builds: ' + ' '.join(f'{dt:.2f}' for dt in jbuilds))
    for n, kind, dt in kbuilds:
        print(f'    K build: {n:8d} points {kind:5s} {dt:7.2f} s' if kind != 'final step (grid + K)' else f'    final step (level-{args.final_level} grid and K): {dt:.2f} s')
else:
    print(f'  J/K {jk:.1f} s, XC {T["xc"]:.1f} s, rest of get_veff {xc_rest:.1f} s, outside get_veff '
          f'{t_scf - T["veff"]:.1f} s')
if err is not None:
    print(f'  error against exact: {err * 1e3:+.4f} mEh')
print(f'  memory: peak RSS {maxrss / 1e9:.2f} GB (kernel); sampled peaks by phase: '
      + ', '.join(f'{ph} RSS {m["rss_gb"]:.2f} GB disk {m["disk_gb"]:.2f} GB' for ph, m in mem.items())
      + f' (max_memory {mf.max_memory} MB)')
if args.json:
    rec = dict(case=case, method=args.method, label=label, tag=args.tag, nao=mol.nao, threads=lib.num_threads(),
               t_scf=t_scf, t_final=t_final, cycles=mf.cycles, converged=bool(mf.converged), e_tot=e_tot,
               e_final=e_final, err=err, t_j=T['j'], t_k=T['k'], t_jk=jk, t_xc=T['xc'], t_rest=xc_rest,
               t_outside=t_scf - T['veff'], kbuilds=kbuilds, jbuilds=jbuilds, maxrss_gb=maxrss / 1e9, mem=mem,
               max_memory=mf.max_memory)
    with open(args.json, 'a') as f:
        f.write(json.dumps(rec) + '\n')
