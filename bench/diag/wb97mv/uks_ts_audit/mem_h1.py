'''Peak resident memory and time of the steps of make_h1 after the XC terms, at production size: the solvent's
Fock derivative (analytical_grad_vmat) and the int2e_ip1 J/K of sample atoms. SCF state from an npz.
Usage: mem_h1.py XYZ SPIN NPZ [--atoms 0,30] [--no-solvent] [--solvent-hess]'''
import sys, time, threading, numpy
from pyscf import gto, dft, lib
from pyscf.hessian import rhf as rhf_hess, uks as uks_hess
args = [x for x in sys.argv[1:] if not x.startswith('--')]
opts = [x for x in sys.argv[1:] if x.startswith('--')]
T0 = time.perf_counter()
PEAK = [0.]
import traceback
MAIN = threading.main_thread().ident
LAST = [0.]
def sampler():
    # at every new high of the resident memory (by 0.5 GB), the lines the main thread is on
    while True:
        now = lib.current_memory()[0]
        PEAK[0] = max(PEAK[0], now)
        if now > LAST[0] + 500:
            LAST[0] = now
            frames = traceback.extract_stack(sys._current_frames()[MAIN])[-4:]
            print(f'    rss {now / 1e3:5.2f} GB at ' + ' < '.join(f'{f.filename.split("/")[-1]}:{f.lineno} {f.name}' for f in reversed(frames)), flush=True)
        time.sleep(.05)
threading.Thread(target=sampler, daemon=True).start()
def say(msg):
    now = lib.current_memory()[0]
    print(f'>>> [{time.perf_counter() - T0:6.0f} s, now {now / 1e3:5.2f} GB, peak since last {max(PEAK[0], now) / 1e3:5.2f} GB] {msg}', flush=True)
    PEAK[0] = 0.
mol = gto.M(atom=args[0], basis='def2-tzvp', spin=int(args[1]), verbose=0)
mf = dft.UKS(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
mf = mf.SMD(); mf.with_solvent.solvent = 'water'
z = numpy.load(args[2])
mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
dms = mf.make_rdm1(mo_coeff, mo_occ)
dm = dms[0] + dms[1]
nao = mol.nao
say(f'{mol.natm} atoms, {nao} AOs, {lib.num_threads()} threads, max_memory {mol.max_memory}')
if '--no-solvent' not in opts:
    from pyscf.solvent.hessian import pcm as pcm_hess
    pcmobj = mf.with_solvent
    pcmobj.build()
    say(f'solvent build: {pcmobj.surface["grid_coords"].shape[0]} surface points')
    pcmobj._get_vind(dm)
    say('solvent _get_vind')
    for name in ('get_dqsym_dx_fix_vgrids', 'get_dqsym_dx_fix_K_R', '_make_dv_parts', '_int1e_grids_deriv', '_int1e_grids_dot'):
        fn = getattr(pcm_hess, name)
        def wrapped(*a, _fn=fn, _name=name, **k):
            t = time.perf_counter(); out = _fn(*a, **k)
            if time.perf_counter() - t > 2:
                say(f'    {_name}: {time.perf_counter() - t:.1f} s')
            return out
        setattr(pcm_hess, name, wrapped)
    t = time.perf_counter()
    dv = pcm_hess.analytical_grad_vmat(pcmobj, dm, atmlst=range(mol.natm))
    say(f'analytical_grad_vmat: {time.perf_counter() - t:.1f} s, output {dv.nbytes / 1e9:.2f} GB')
    dv = None
    if '--solvent-hess' in opts:
        for name in ('analytical_hess_nuc', 'analytical_hess_qv', 'analytical_hess_solver'):
            t = time.perf_counter()
            getattr(pcm_hess, name)(pcmobj, dm)
            say(f'{name}: {time.perf_counter() - t:.1f} s')
atoms = [0, 30]
for o in opts:
    if o.startswith('--atoms='):
        atoms = [int(x) for x in o[8:].split(',')] if o[8:] else []
if atoms:
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    dm0a, dm0b = dms
    aoslices = mol.aoslice_by_atom()
    t = time.perf_counter()
    vhfopt = rhf_hess._make_h1_vhfopt(mol, [dm, dm0a, dm0b], 'jk')
    with mol.with_range_coulomb(omega):
        vhfopt_lr = rhf_hess._make_h1_vhfopt(mol, [dm0a, dm0b], 'k')
    say(f'vhfopt: {time.perf_counter() - t:.1f} s')
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        t = time.perf_counter()
        uks_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                         ['ji->s2kl', -dm[:,p0:p1], 'lk->s1ij', -dm,
                          'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                          'jk->s1il', -dm0a, 'jk->s1il', -dm0b],
                         shls_slice=shls_slice, vhfopt=vhfopt)
        t1 = time.perf_counter()
        with mol.with_range_coulomb(omega):
            uks_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                             ['li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                              'jk->s1il', -dm0a, 'jk->s1il', -dm0b],
                             shls_slice=shls_slice, vhfopt=vhfopt_lr)
        say(f'J/K of atom {ia} ({mol.atom_symbol(ia)}, {p1 - p0} AOs): {t1 - t:.1f} s + long-range K {time.perf_counter() - t1:.1f} s')
    nbf = numpy.array([aoslices[i][3] - aoslices[i][2] for i in range(mol.natm)])
    print('AOs per atom:', dict(zip(*numpy.unique(nbf, return_counts=True))), flush=True)
