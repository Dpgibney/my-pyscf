"""It14 check: the Hessian's two-electron blocks (ej, ek, and the long-range K of range-separated
hybrids) and hess_elec's mo1 terms against copies of the code they replace (`perf` d88076cb8).

Run at 1 thread for exactness (the direct J/K driver sums thread-private blocks in arbitrary order):
    bench/diag/run_in.py G=pyscf 1 bench/diag/hess_ejk_check.py [water|ethylene|benzene ...]
"""
import os
import sys
import time

import numpy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, WATER  # noqa: E402
from pyscf import dft, gto, lib, scf  # noqa: E402
from pyscf.hessian import rhf as rhf_hess  # noqa: E402
from pyscf.hessian import uhf as uhf_hess  # noqa: E402

ETHYLENE = 'C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237'
_get_jk = rhf_hess._get_jk


def ref_rhf_ejk(mol, dm0, atmlst, with_k=True):
    '''ej, ek of hessian/rhf.py _partial_hess_ejk before It14'''
    nao = mol.nao
    jk_diag = ['lk->s1ij', dm0]
    if with_k:
        jk_diag += ['jk->s1il', dm0]
    vjk1_diag = _get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk_diag)
    vj1_diag = vjk1_diag[0].reshape(3,3,nao,nao)
    if with_k:
        vk1_diag = vjk1_diag[1].reshape(3,3,nao,nao)
    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        jk1 = ['ji->s1kl', dm0[:,p0:p1]]
        if with_k:
            jk1 += ['li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
        vjk1 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=shls_slice)
        vj1 = vjk1[0]
        if with_k:
            vk1, vk2 = vjk1[1:]
            vk1[:,:,p0:p1] += vk2
        jk2 = ['lk->s1ij', dm0]
        if with_k:
            jk2 += ['li->s1kj', dm0[:,p0:p1]]
        vjk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, shls_slice=shls_slice)
        vj1[:,:,p0:p1] += vjk2[0].transpose(0,2,1) * .5
        vj1 = vj1.reshape(3,3,nao,nao)
        if with_k:
            vk1 += vjk2[1].transpose(0,2,1)
            vk1 = vk1.reshape(3,3,nao,nao)
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


def ref_uhf_ejk(mol, dm0a, dm0b, atmlst, with_k=True):
    '''ej, ek of hessian/uhf.py _partial_hess_ejk before It14'''
    nao = mol.nao
    dm0 = dm0a + dm0b
    jk_diag = ['lk->s1ij', dm0]
    if with_k:
        jk_diag += ['jk->s1il', dm0a, 'jk->s1il', dm0b]
    vjk1_diag = _get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk_diag)
    vj1_diag = vjk1_diag[0].reshape(3,3,nao,nao)
    if with_k:
        vk1a_diag = vjk1_diag[1].reshape(3,3,nao,nao)
        vk1b_diag = vjk1_diag[2].reshape(3,3,nao,nao)
    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        jk1 = ['ji->s1kl', dm0[:,p0:p1]]
        if with_k:
            jk1 += ['li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1],
                    'lj->s1ki', dm0a, 'lj->s1ki', dm0b]
        vjk1 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=shls_slice)
        vj1 = vjk1[0]
        if with_k:
            vk1a, vk1b, vk2a, vk2b = vjk1[1:]
            vk1a[:,:,p0:p1] += vk2a
            vk1b[:,:,p0:p1] += vk2b
        jk2 = ['lk->s1ij', dm0]
        if with_k:
            jk2 += ['li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1]]
        vjk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, shls_slice=shls_slice)
        vj1[:,:,p0:p1] += vjk2[0].transpose(0,2,1) * .5
        if with_k:
            vk1a += vjk2[1].transpose(0,2,1)
            vk1b += vjk2[2].transpose(0,2,1)
        vj1 = vj1.reshape(3,3,nao,nao)
        if with_k:
            vk1a = vk1a.reshape(3,3,nao,nao)
            vk1b = vk1b.reshape(3,3,nao,nao)
        ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1_diag[:,:,p0:p1], dm0[p0:p1])*2
        if with_k:
            ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1a_diag[:,:,p0:p1], dm0a[p0:p1])*2
            ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1b_diag[:,:,p0:p1], dm0b[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1])*4
            if with_k:
                ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1a[:,:,q0:q1], dm0a[q0:q1])*2
                ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1b[:,:,q0:q1], dm0b[q0:q1])*2
        for j0 in range(i0):
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T
    return ej, ek


def ref_lr_k(mol, dms, omega, atmlst):
    '''The long-range K blocks that hessian/rks.py and uks.py added to de2 before It14, divided by
    -(alpha-hyb): with dms = [dm0] the RKS ones, with [dm0a, dm0b] the UKS ones.'''
    nao = mol.nao
    fac = 1 if len(dms) == 1 else 2   # rks: (alpha-hyb)*.5 * vk, *2; uks: (alpha-hyb) * vk, *2
    with mol.with_range_coulomb(omega):
        vk_diag = _get_jk(mol, 'int2e_ipip1', 9, 's2kl', sum((['jk->s1il', dm] for dm in dms), []))
    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    out = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        veff = [numpy.zeros((3,3,nao,nao)) for dm in dms]
        with mol.with_range_coulomb(omega):
            v = _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                        sum((['li->s1kj', dm[:,p0:p1]] for dm in dms), [])
                        + sum((['lj->s1ki', dm] for dm in dms), []), shls_slice=shls_slice)
        n = len(dms)
        for s in range(n):
            veff[s] += v[s].reshape(3,3,nao,nao)
            veff[s][:,:,:,p0:p1] += v[n+s].reshape(3,3,nao,p1-p0)
        with mol.with_range_coulomb(omega):
            v = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                        sum((['li->s1kj', dm[:,p0:p1]] for dm in dms), []), shls_slice=shls_slice)
        for s in range(n):
            veff[s] += v[s].transpose(0,2,1).reshape(3,3,nao,nao)
        for s, dm in enumerate(dms):
            out[i0,i0] += numpy.einsum('xypq,pq->xy', vk_diag[s].reshape(3,3,nao,nao)[:,:,p0:p1],
                                       dm[p0:p1]) * fac
            for j0, ja in enumerate(atmlst[:i0+1]):
                q0, q1 = aoslices[ja][2:]
                out[i0,j0] += numpy.einsum('xypq,pq->xy', veff[s][:,:,q0:q1], dm[q0:q1]) * fac
        for j0 in range(i0):
            out[j0,i0] = out[i0,j0].T
    return out


def ref_hess_elec_mo1(mol, atmlst, h1ao, mo1, mo_e1, mocc, mo_eocc):
    '''hess_elec's mo1 terms (hessian/rhf.py before It14) without the factor 4 for RHF, i.e. the
    terms of one spin of hessian/uhf.py divided by 2.'''
    nao = mocc.shape[0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    de2 = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)
        for j0 in range(i0+1):
            ja = atmlst[j0]
            dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1)
            dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_eocc)
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1)
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * .5
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T
    return de2


def report(label, ref, new):
    off = ~numpy.eye(ref.shape[0], dtype=bool)
    d_off = abs(ref[off] - new[off]).max() if off.any() else 0.
    d_diag = max(abs(ref[i,i] - new[i,i]).max() for i in range(ref.shape[0]))
    print(f'  {label:34s} max|d| off-diagonal blocks {d_off:.1e}, diagonal blocks {d_diag:.1e} '
          f'(max|ref| {abs(ref).max():.2e})', flush=True)


def timed(f, *args, **kwargs):
    t0 = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t0


def check(name, atom):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    natm = mol.natm
    full = list(range(natm))
    lists = {'all atoms': full, 'reversed': full[::-1], 'subset [2,0]': [2, 0]}
    print(f'{name}: natm {natm}, nao {mol.nao}, {lib.num_threads()} thread(s)', flush=True)

    mf = scf.RHF(mol).run(conv_tol=1e-10)
    mocc = mf.mo_coeff[:,mf.mo_occ>0]
    dm0 = mocc.dot(mocc.T) * 2
    hobj = mf.Hessian()
    for label, atmlst in lists.items():
        for with_k in (True, False):
            (ej0, ek0), t0 = timed(ref_rhf_ejk, mol, dm0, atmlst, with_k)
            (e1, ej1, ek1), t1 = timed(rhf_hess._partial_hess_ejk, hobj, mf.mo_energy, mf.mo_coeff,
                                       mf.mo_occ, atmlst, with_k=with_k)
            print(f' RHF {label}, with_k={with_k}: before {t0:.2f} s, now {t1:.2f} s')
            report('ej', ej0, ej1)
            if with_k:
                report('ek', ek0, ek1)
        if name == 'benzene':
            break

    h1ao = hobj.make_h1(mf.mo_coeff, mf.mo_occ, None, full)
    mo1, mo_e1 = hobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, None, full)
    eocc = mf.mo_energy[mf.mo_occ>0]
    for label, atmlst in lists.items():
        ref, t0 = timed(ref_hess_elec_mo1, mol, atmlst, h1ao, mo1, mo_e1, mocc, eocc)
        new, t1 = timed(rhf_hess._hess_elec_mo1, mol, atmlst, h1ao, mo1, mo_e1, mocc, eocc)
        nat = len(atmlst)
        new = numpy.array([[new[i,j] if j <= i else new[j,i].T for j in range(nat)] for i in range(nat)])
        print(f' hess_elec mo1 terms, {label}: before {t0:.2f} s, now {t1:.3f} s')
        report('h1/s1/e1 terms', ref, new)

    if name == 'benzene':
        return
    mol1 = gto.M(atom=atom, basis='def2-svp', charge=1, spin=1, verbose=0)
    mfu = scf.UHF(mol1).run(conv_tol=1e-10)
    moa = mfu.mo_coeff[0][:,mfu.mo_occ[0]>0]
    mob = mfu.mo_coeff[1][:,mfu.mo_occ[1]>0]
    dma, dmb = moa.dot(moa.T), mob.dot(mob.T)
    hobju = mfu.Hessian()
    for label, atmlst in lists.items():
        (ej0, ek0), t0 = timed(ref_uhf_ejk, mol1, dma, dmb, atmlst)
        (e1, ej1, ek1), t1 = timed(uhf_hess._partial_hess_ejk, hobju, mfu.mo_energy, mfu.mo_coeff,
                                   mfu.mo_occ, atmlst)
        print(f' UHF {label}: before {t0:.2f} s, now {t1:.2f} s')
        report('ej', ej0, ej1)
        report('ek', ek0, ek1)

    omega = .33
    for label, atmlst in lists.items():
        ref = ref_lr_k(mol, [dm0], omega, atmlst)
        new = rhf_hess._partial_hess_2e(mol, None, [dm0], 1, atmlst, with_j=False, omega=omega)[1]
        print(f' RKS long-range K, {label}')
        report('ek(omega)', ref, new)
        ref = ref_lr_k(mol1, [dma, dmb], omega, atmlst)
        new = rhf_hess._partial_hess_2e(mol1, None, [dma, dmb], 2, atmlst, with_j=False,
                                        omega=omega)[1]
        print(f' UKS long-range K, {label}')
        report('ek(omega)', ref, new)


if __name__ == '__main__':
    systems = {'water': WATER, 'ethylene': ETHYLENE, 'benzene': BENZENE}
    for name in sys.argv[1:] or ['water', 'ethylene']:
        check(name, systems[name])
