'''_partial_hess_ejk (hessian/rhf.py:99-188) with (a) the ip1ip2 k range and the ipvip1 j range cut to
atoms before A (the blocks B < A it reads), and (b) the diagonal 2e blocks from translational
invariance, ej[A,A] = -sum_{B!=A} ej[A,B] (same for ek), so int2e_ipip1 is not computed.
Compared with the tree's function on benzene (RHF, so J and K), single thread.'''
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, WATER
from pyscf import gto, scf, lib
from pyscf.hessian import rhf as rhf_hess
_get_jk = rhf_hess._get_jk

def partial_hess_ejk_pairs(hessobj, mo_energy, mo_coeff, mo_occ, with_k=True):
    mol = hessobj.mol
    atmlst = range(mol.natm)       # needs shells ordered by atom (always true for Mole)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0]) * 2
    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    e1 = numpy.zeros((natm, natm, 3, 3))
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        if i0 > 0:
            # shells and AOs of the atoms before ia
            kshl1, kao1 = shl0, p0
            jk1 = ['ji->s1kl', dm0[:,p0:p1]]
            if with_k:
                jk1 += ['li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
            vjk1 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1,
                           shls_slice=(shl0, shl1, 0, mol.nbas, 0, kshl1, 0, mol.nbas))
            vj1 = vjk1[0]                             # [9, k<A, l]
            if with_k:
                vk1, vk2 = vjk1[1:]                   # [9, k<A, j], [9, k<A, i in A]
                vk1[:,:,p0:p1] += vk2
            jk2 = ['lk->s1ij', dm0]
            if with_k:
                jk2 += ['li->s1kj', dm0[:,p0:p1]]
            vjk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2,
                           shls_slice=(shl0, shl1, 0, kshl1, 0, mol.nbas, 0, mol.nbas))
            # vjk2[0]: [9, i in A, j<A] -> rows j<A, columns of A of vj1
            vj1[:,:,p0:p1] += vjk2[0].transpose(0,2,1) * .5
            vj1 = vj1.reshape(3,3,kao1,nao)
            if with_k:
                vk1 += vjk2[1][:,:,:kao1].transpose(0,2,1)   # [9, k, j<A] -> rows j<A
                vk1 = vk1.reshape(3,3,kao1,nao)
            for j0, ja in enumerate(atmlst[:i0]):
                q0, q1 = aoslices[ja][2:]
                ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1])*4
                if with_k:
                    ek[i0, j0] += numpy.einsum('xypq,pq->xy', vk1[:,:,q0:q1], dm0[q0:q1])
        e1[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2
            e1[i0, j0] += numpy.einsum('xypq,pq->xy', hcore_deriv(ia, ja), dm0)
        for j0 in range(i0):
            e1[j0, i0] = e1[i0, j0].T
            ej[j0, i0] = ej[i0, j0].T
            ek[j0, i0] = ek[i0, j0].T
    # translational invariance of the 2e terms
    for a in range(natm):
        ej[a, a] = -(ej[a].sum(axis=0) - ej[a, a])
        ek[a, a] = -(ek[a].sum(axis=0) - ek[a, a])
    return e1, ej, ek

for name, atom in (('water', WATER), ('benzene', BENZENE)):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    h = mf.Hessian()
    t0 = time.perf_counter()
    ref = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, range(mol.natm))
    t1 = time.perf_counter()
    new = partial_hess_ejk_pairs(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    t2 = time.perf_counter()
    for lab, a, b in zip(('e1', 'ej', 'ek'), ref, new):
        off = ~numpy.eye(mol.natm, dtype=bool)
        print(f'{name} {lab}: max|d| off-diagonal blocks {abs(a[off]-b[off]).max():.1e}, '
              f'diagonal blocks {max(abs(a[i,i]-b[i,i]).max() for i in range(mol.natm)):.1e} '
              f'(max|{lab}| {abs(a).max():.2e})')
    print(f'{name}: tree {t1-t0:.2f} s, pairs+TI {t2-t1:.2f} s', flush=True)
