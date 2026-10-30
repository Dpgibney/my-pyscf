'''Time the post-CPHF loop of hessian/rhf.py:hess_elec (lines 63-86) on caffeine/benzene shapes with
synthetic mo1/mo_e1/h1ao, against a BLAS formulation; check agreement. Single thread.'''
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, BENZENE
from pyscf import gto, lib

def current(mol, mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst):
    # verbatim from hessian/rhf.py hess_elec, minus partial_hess/make_h1/solve_mo1
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    de2 = numpy.zeros((len(atmlst), len(atmlst), 3, 3))
    t = {'s1oo': 0., 'dm1': 0., 'h1': 0., 'dme1': 0., 's1': 0., 'e1': 0.}
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        c0 = time.perf_counter()
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)
        t['s1oo'] += time.perf_counter() - c0
        for j0 in range(i0+1):
            ja = atmlst[j0]
            c0 = time.perf_counter()
            dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            c1 = time.perf_counter()
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            c2 = time.perf_counter()
            dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[mo_occ>0])
            c3 = time.perf_counter()
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            c4 = time.perf_counter()
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2
            c5 = time.perf_counter()
            t['dm1'] += c1-c0; t['h1'] += c2-c1; t['dme1'] += c3-c2; t['s1'] += c4-c3; t['e1'] += c5-c4
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T
    return de2, t

def blas(mol, mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst):
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    e_occ = mo_energy[mo_occ>0]
    natm = len(atmlst)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    x = numpy.asarray([mo1[ja] for ja in atmlst]).reshape(-1,nao,nocc)       # [B*3,nao,nocc]
    dm1 = lib.dot(x.reshape(-1,nocc), mocc.T).reshape(natm*3,nao,nao)
    dme1 = lib.dot((x*e_occ).reshape(-1,nocc), mocc.T).reshape(natm*3,nao,nao)
    h1 = numpy.asarray([h1ao[ia] for ia in atmlst]).reshape(natm*3,-1)
    de2 = 4 * lib.dot(h1, dm1.reshape(natm*3,-1).T)                           # [A*3,B*3]
    dme1s = dme1 + dme1.transpose(0,2,1)
    e1 = numpy.asarray([mo_e1[ja] for ja in atmlst]).reshape(natm*3,-1)
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        # sum_pq s1ao[x,p,q] dme1[y,p,q] with s1ao nonzero on rows/cols of atom A
        s = s1a[:,p0:p1].reshape(3,-1)
        de2[i0*3:i0*3+3] -= 4 * lib.dot(s, dme1s[:,p0:p1].reshape(natm*3,-1).T)
        t = lib.dot(s1a[:,p0:p1].reshape(-1,nao), mocc).reshape(3,p1-p0,nocc)
        s1oo = numpy.einsum('pi,xpj->xij', mocc[p0:p1], t)
        s1oo = s1oo + s1oo.transpose(0,2,1)
        de2[i0*3:i0*3+3] -= 2 * lib.dot(s1oo.reshape(3,-1), e1.T)
    de2 = de2.reshape(natm,3,natm,3).transpose(0,2,1,3).copy()
    for i0 in range(natm):   # as hess_elec: the upper blocks are the transposed lower ones
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T
    return de2

for name, atom in (('benzene', BENZENE), ('caffeine', CAFFEINE)):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    nao = mol.nao
    nocc = mol.nelectron // 2
    rng = numpy.random.default_rng(0)
    s = mol.intor('int1e_ovlp')
    e, c = numpy.linalg.eigh(s)
    mo_coeff = c / numpy.sqrt(e)  # orthonormal MOs
    mo_occ = numpy.zeros(nao); mo_occ[:nocc] = 2
    mo_energy = numpy.sort(rng.standard_normal(nao))
    atmlst = range(mol.natm)
    mo1 = [rng.standard_normal((3,nao,nocc)) for _ in atmlst]
    mo_e1 = [rng.standard_normal((3,nocc,nocc)) for _ in atmlst]
    h1ao = [rng.standard_normal((3,nao,nao)) for _ in atmlst]
    t0 = time.perf_counter()
    ref, tt = current(mol, mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst)
    t1 = time.perf_counter()
    new = blas(mol, mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst)
    t2 = time.perf_counter()
    err = abs(new-ref).max() / abs(ref).max()
    print(f'{name}: nao {nao} nocc {nocc} natm {mol.natm}: current {t1-t0:.3f} s '
          f'({", ".join(f"{k} {v:.3f}" for k, v in tt.items())}); blas {t2-t1:.3f} s; rel diff {err:.2e}',
          flush=True)

# the solve_mo1 back-transformation einsum (rhf.py:345) on caffeine shapes
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
nao = mol.nao; nocc = mol.nelectron // 2
rng = numpy.random.default_rng(1)
mo_coeff = rng.standard_normal((nao,nao)); mo1 = rng.standard_normal((mol.natm*3,nao,nocc))
t0 = time.perf_counter()
a = numpy.einsum('pq,xqi->xpi', mo_coeff, mo1)
t1 = time.perf_counter()
b = lib.einsum('pq,xqi->xpi', mo_coeff, mo1)
t2 = time.perf_counter()
print(f'solve_mo1 back-transform einsum: numpy {t1-t0:.3f} s, lib.einsum {t2-t1:.3f} s, '
      f'diff {abs(a-b).max()/abs(a).max():.1e}')
