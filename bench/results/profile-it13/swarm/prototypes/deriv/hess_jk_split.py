'''Time the 2e derivative J/K calls of the RKS-hybrid Hessian (hessian/rhf.py _partial_hess_ejk, make_h1)
one family at a time, single thread, and test the k/j-range restriction to the atoms that
_partial_hess_ejk actually reads (atmlst[:i0+1]).
usage: hess_jk_split.py MOL TEST [ATOMS]
  MOL: benzene | caffeine | water
  TEST: ip1ip2 | ipvip1 | ipip1 | ip1
  ATOMS: comma list of atom ids (default: all)'''
import sys, time, ctypes
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, BENZENE, WATER
from pyscf import gto, scf, lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import _vhf

molname, test = sys.argv[1], sys.argv[2]
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'water': WATER}[molname]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
nao = mol.nao
atoms = list(range(mol.natm)) if len(sys.argv) < 4 else [int(a) for a in sys.argv[3].split(',')]

# a realistic dm0: a few RHF cycles from minao (the screening only needs its magnitudes)
mf = scf.RHF(mol)
mf.max_cycle = 3 if molname == 'caffeine' else 6
mf.conv_check = False
t0 = time.perf_counter()
mf.kernel()
mocc = mf.mo_coeff[:,mf.mo_occ>0]
dm0 = numpy.dot(mocc, mocc.T) * 2
print(f'{molname} nao {nao} natm {mol.natm} nbas {mol.nbas}; dm0 from {mf.cycles} RHF cycles '
      f'in {time.perf_counter()-t0:.1f} s', flush=True)
aoslices = mol.aoslice_by_atom()

def dm_cond_of(dm):
    '''max |dm| per shell pair (CVHFnr_dm_cond1) of the full dm0; bounds every column slice of it'''
    ao_loc = mol.ao_loc_nr()
    out = numpy.empty((mol.nbas, mol.nbas))
    dm = numpy.asarray(dm, order='C')
    _vhf.libcvhf.CVHFnr_dm_cond1(out.ctypes, dm.ctypes, ctypes.c_int(1), ao_loc.ctypes,
                                 mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes,
                                 ctypes.c_int(mol.nbas), mol._env.ctypes)
    return out

def screened_opt(key, vhf_intor):
    '''_make_vhfopt as upstream intended it (its hasattr test misses the _sph suffix, so it returns
    None), with dm_cond from the full dm0 so that dms of different shapes are fine'''
    opt = _vhf._VHFOpt(mol, 'int2e_'+key, 'CVHF'+key+'_prescreen', dmcondname=None)
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q_cond = numpy.empty((2, nbas, nbas))
    libcvhf = _vhf.libcvhf
    with mol.with_integral_screen(opt.direct_scf_tol**2):
        fq = libcvhf.CVHFnr_int2e_pp_q_cond if vhf_intor == 'int2e_ip1ip2' else libcvhf.CVHFnr_int2e_pppp_q_cond
        fq(getattr(libcvhf, mol._add_suffix(vhf_intor)), lib.c_null_ptr(), q_cond[0].ctypes,
           ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes, ctypes.c_int(nbas),
           mol._env.ctypes)
        libcvhf.CVHFnr_int2e_q_cond(getattr(libcvhf, mol._add_suffix('int2e')), lib.c_null_ptr(),
           q_cond[1].ctypes, ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes,
           ctypes.c_int(nbas), mol._env.ctypes)
    opt.q_cond = q_cond
    opt.dm_cond = dm_cond_of(dm0)
    return opt

def kslice(i0):
    '''shells of atoms 0..i0 (atmlst = range(natm), shells ordered by atom)'''
    return 0, aoslices[i0][1], 0, aoslices[i0][3]

if test == 'ip1ip2':
    opt = rhf_hess._make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
    sopt = screened_opt('ip1ip2', 'int2e_ip1ip2')
    tf = tr = ts = 0; err = serr = 0
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        jk = ['ji->s1kl', dm0[:,p0:p1], 'li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
        c0 = time.perf_counter()
        full = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk,
                                shls_slice=(shl0, shl1) + (0, mol.nbas)*3, vhfopt=opt)
        c1 = time.perf_counter()
        k0, k1, q0, q1 = kslice(ia)
        res = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk,
                               shls_slice=(shl0, shl1, 0, mol.nbas, k0, k1, 0, mol.nbas), vhfopt=opt)
        c2 = time.perf_counter()
        scr = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk,
                               shls_slice=(shl0, shl1, 0, mol.nbas, k0, k1, 0, mol.nbas), vhfopt=sopt)
        c3 = time.perf_counter()
        ts += c3 - c2
        serr = max(serr, max(abs(a[:,q0:q1] - b).max() for a, b in zip(full, scr)))
        tf += c1 - c0; tr += c2 - c1
        for a, b in zip(full, res):
            e = abs(a[:,q0:q1] - b).max() / abs(a[:,q0:q1]).max()
            err = max(err, e)
        print(f'  atom {ia:2d} ({mol.atom_symbol(ia)}) nA {p1-p0:3d}: full {c1-c0:.2f} s, '
              f'k<=atom {c2-c1:.2f} s, k<=atom+screened {c3-c2:.2f} s', flush=True)
    print(f'ip1ip2 total: full {tf:.2f} s, restricted {tr:.2f} s ({tr/tf:.2f}); max rel diff {err:.1e}; '
          f'restricted+screened {ts:.2f} s ({ts/tf:.2f}), max abs diff {serr:.1e}')

elif test == 'ipvip1':
    opt = rhf_hess._make_vhfopt(mol, dm0, 'ipvip1', 'int2e_ipvip1ipvip2')
    tf = tr = 0; err = 0
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        jk = ['lk->s1ij', dm0, 'li->s1kj', dm0[:,p0:p1]]
        c0 = time.perf_counter()
        full = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk,
                                shls_slice=(shl0, shl1) + (0, mol.nbas)*3, vhfopt=opt)
        c1 = time.perf_counter()
        j0, j1, q0, q1 = kslice(ia)
        res = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk,
                               shls_slice=(shl0, shl1, j0, j1, 0, mol.nbas, 0, mol.nbas), vhfopt=opt)
        c2 = time.perf_counter()
        tf += c1 - c0; tr += c2 - c1
        # vj [9, nA, nj]: columns j <= atom; vk [9, nk, nj]: columns j <= atom
        e = max(abs(full[0][:,:,q0:q1] - res[0]).max() / abs(full[0][:,:,q0:q1]).max(),
                abs(full[1][:,:,q0:q1] - res[1]).max() / abs(full[1][:,:,q0:q1]).max())
        err = max(err, e)
        print(f'  atom {ia:2d} ({mol.atom_symbol(ia)}) nA {p1-p0:3d}: full {c1-c0:.2f} s, '
              f'j<=atom {c2-c1:.2f} s', flush=True)
    print(f'ipvip1 total: full {tf:.2f} s, restricted {tr:.2f} s ({tr/tf:.2f}); max rel diff {err:.1e}')

elif test == 'ipip1':
    c0 = time.perf_counter()
    rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl', ['lk->s1ij', dm0, 'jk->s1il', dm0],
                     vhfopt=rhf_hess._make_vhfopt(mol, dm0, 'ipip1', 'int2e_ipip1ipip2'))
    print(f'ipip1 (one call, J+K): {time.perf_counter()-c0:.2f} s')

elif test == 'ipip1slice':
    # int2e_ipip1 J+K restricted to the i shells of single atoms (its cost is linear in the i range)
    tt = 0
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        c0 = time.perf_counter()
        rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl', ['lk->s1ij', dm0, 'jk->s1il', dm0],
                         shls_slice=(shl0, shl1) + (0, mol.nbas)*3)
        tt += time.perf_counter() - c0
        print(f'  atom {ia:2d} ({mol.atom_symbol(ia)}): ipip1 i-slice {time.perf_counter()-c0:.2f} s', flush=True)

elif test == 'ip1':
    # make_h1 as in hessian/rks.py (hybrid): per atom, 4 dms, no vhfopt (no Schwarz screening)
    tn = ts = 0; err = 0
    vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen', dmcondname=None)
    vhfopt.q_cond = rhf_grad._calc_q_cond(mol, vhfopt)
    vhfopt.dm_cond = dm_cond_of(dm0)
    for ia in atoms:
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        jk = ['ji->s2kl', -dm0[:,p0:p1], 'lk->s1ij', -dm0, 'li->s1kj', -dm0[:,p0:p1], 'jk->s1il', -dm0]
        c0 = time.perf_counter()
        a = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl', jk, shls_slice=shls_slice)
        c1 = time.perf_counter()
        b = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl', jk, shls_slice=shls_slice, vhfopt=vhfopt)
        c2 = time.perf_counter()
        tn += c1 - c0; ts += c2 - c1
        err = max(err, max(abs(x - y).max() for x, y in zip(a, b)))
        print(f'  atom {ia:2d} ({mol.atom_symbol(ia)}): unscreened {c1-c0:.2f} s, '
              f'grad-style screened {c2-c1:.2f} s', flush=True)
    print(f'make_h1 ip1 total: unscreened {tn:.2f} s, screened {ts:.2f} s ({ts/tn:.2f}); '
          f'max abs diff {err:.1e}')
    if len(atoms) == mol.natm:
        c0 = time.perf_counter()
        rhf_grad.get_jk(mol, dm0)
        print(f'gradient get_jk (one call, all atoms, screened): {time.perf_counter()-c0:.2f} s')
