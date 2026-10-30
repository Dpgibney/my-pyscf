"""_partial_hess_ejk with the int2e_ip1ip2 k range and the int2e_ipvip1 j range cut to the shells of
atoms B <= A (the only rows the ej[A,B], ek[A,B] contractions read). Compares against the original
and times both (1 thread). Usage: hess_tri.py SYSTEM [compare|time]"""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE, WATER
from pyscf import gto, scf, lib
from pyscf.hessian import rhf as rhf_hess

_get_jk = rhf_hess._get_jk
_make_vhfopt = rhf_hess._make_vhfopt


def ejk_tri(mol, mo_energy, mo_coeff, mo_occ, atmlst=None, with_k=True):
    if atmlst is None: atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    jk_diag = ['lk->s1ij', dm0]
    if with_k:
        jk_diag += ['jk->s1il', dm0]
    vjk1_diag = _get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk_diag,
                        vhfopt=_make_vhfopt(mol, dm0, 'ipip1', 'int2e_ipip1ipip2'))
    vj1_diag = vjk1_diag[0].reshape(3,3,nao,nao)
    if with_k:
        vk1_diag = vjk1_diag[1].reshape(3,3,nao,nao)
    ip1ip2_opt = _make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
    ipvip1_opt = _make_vhfopt(mol, dm0, 'ipvip1', 'int2e_ipvip1ipvip2')
    aoslices = mol.aoslice_by_atom()
    ao_loc = mol.ao_loc_nr()
    natm = len(atmlst)
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        # shells of the atoms B <= A of atmlst (a superset if atmlst is not ordered)
        bsh1 = max(aoslices[ja][1] for ja in atmlst[:i0+1])
        nb = ao_loc[bsh1]
        jk1 = ['ji->s1kl', dm0[:,p0:p1]]
        if with_k:
            jk1 += ['li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
        vjk1 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1,
                       shls_slice=(shl0, shl1, 0, mol.nbas, 0, bsh1, 0, mol.nbas),
                       vhfopt=ip1ip2_opt)
        vj1 = vjk1[0]                       # (9, nb, nao)
        if with_k:
            vk1, vk2 = vjk1[1:]             # (9, nb, nao), (9, nb, nA)
            vk1[:,:,p0:p1] += vk2
        jk2 = ['lk->s1ij', dm0]
        if with_k:
            jk2 += ['li->s1kj', dm0[:,p0:p1]]
        vjk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2,
                       shls_slice=(shl0, shl1, 0, bsh1, 0, mol.nbas, 0, mol.nbas),
                       vhfopt=ipvip1_opt)
        vj1[:,:,p0:p1] += vjk2[0].transpose(0,2,1) * .5     # (9, nA, nb)^T
        vj1 = vj1.reshape(3,3,nb,nao)
        if with_k:
            vk1 += vjk2[1].transpose(0,2,1)                 # (9, nao, nb)^T
            vk1 = vk1.reshape(3,3,nb,nao)
        ej[i0, i0] += numpy.einsum('xypq,pq->xy', vj1_diag[:,:,p0:p1], dm0[p0:p1])*2
        if with_k:
            ek[i0, i0] += numpy.einsum('xypq,pq->xy', vk1_diag[:,:,p0:p1], dm0[p0:p1])
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1])*4
            if with_k:
                ek[i0, j0] += numpy.einsum('xypq,pq->xy', vk1[:,:,q0:q1], dm0[q0:q1])
        for j0 in range(i0):
            ej[j0, i0] = ej[i0, j0].T
            ek[j0, i0] = ek[i0, j0].T
    return ej, ek


if __name__ == '__main__':
    system = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'compare'
    atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'water': WATER,
            'ethylene': 'C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237'}[system]
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    hobj = mf.Hessian()
    if mode == 'compare':
        t0 = time.perf_counter()
        e1, ej0, ek0 = rhf_hess._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        t1 = time.perf_counter()
        ej1, ek1 = ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        t2 = time.perf_counter()
        print(f'{system}: original {t1-t0:.2f} s, triangle {t2-t1:.2f} s')
        print('max |ej diff|', abs(ej1-ej0).max(), 'max |ej|', abs(ej0).max())
        print('max |ek diff|', abs(ek1-ek0).max(), 'max |ek|', abs(ek0).max())
        # J only (with_k=False)
        _, ej0b, _ = rhf_hess._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ, with_k=False)
        ej1b, _ = ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ, with_k=False)
        print('with_k=False: max |ej diff|', abs(ej1b-ej0b).max())
        # a permuted atom list
        atl = list(range(mol.natm))[::-1]
        _, ej0c, ek0c = rhf_hess._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
        ej1c, ek1c = ejk_tri(mol, mf.mo_energy, mf.mo_coeff, mf.mo_occ, atmlst=atl)
        print('reversed atmlst: max diff', abs(ej1c-ej0c).max(), abs(ek1c-ek0c).max())
