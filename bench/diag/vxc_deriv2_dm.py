'''Prototype of hessian.rks._get_vxc_deriv2_dm, tested against the original _get_vxc_deriv2.'''
import numpy
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad
from pyscf.hessian.rks import _make_dR_rho1, _d1d2_dot_

# second-derivative AO components: _D2[x][y] = index of d_x d_y ao
_D2 = ((4, 5, 6), (5, 7, 8), (6, 8, 9))


def _atom_pair_sum(aoslices, x, y):
    '''[a,g] = sum_{p in atom a} x[g,p] * y[g,p] for F-ordered x, y of shape (ngrids,nao)'''
    xt, yt = x.T, y.T
    return numpy.array([numpy.einsum('pg,pg->g', xt[p0:p1], yt[p0:p1])
                        for p0, p1 in aoslices[:,2:]])


def _get_vxc_deriv2_dm(hessobj, mo_coeff, mo_occ, max_memory):
    '''e[A,B,x,y] = sum_{p in B} sum_q vxc[A,x,y,p,q] dm0[p,q] for vxc = _get_vxc_deriv2(...).

    The per-atom terms of vxc are weighted AO products whose contraction with dm0 reduces
    to sums over the grid of atom-resolved pair densities (ao_k * ao_dm0)[g, p in B]. These
    are evaluated once per grid block for all atoms instead of 18 Ng*nao^2 GEMMs per atom.
    '''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    vxc_dm = numpy.zeros((natm,natm,3,3))
    ipip = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            # t[B,d,g] = sum_{p in B} (d_d ao)[g,p] ao_dm0[g,p]
            t = numpy.array([_atom_pair_sum(aoslices, ao[1+d], ao_dm0)
                             for d in range(3)]).transpose(1,0,2)
            for ia in range(natm):
                p0, p1 = aoslices[ia][2:]
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                vxc_dm[ia] += numpy.einsum('sg,btg->bst', wf * rho1, t)
            ao_dm0 = aow = None

    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 2
        ncomp = 4 if xctype == 'GGA' else 5
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'GGA':
                rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            else:
                rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            if xctype == 'MGGA':
                wv[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
            if xctype == 'MGGA':
                aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
                _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[4], ao[5], ao[6]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[5], ao[7], ao[8]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[6], ao[8], ao[9]], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            # t[B,d,c,g]: the dm0 contraction of the c-th density component of the
            # per-atom terms, sum_{p in B} (A[d,c] * ao_dm0[0] + d_d ao * ao_dm0[c])[g,p]
            # with A[d,0] = d_d ao and A[d,c] = d_d d_(c-1) ao; c=4 (MGGA) is the tau term
            # sum_x sum_{p in B} (d_x d_d ao * ao_dm0[1+x])[g,p]
            ngrids = weight.size
            t = numpy.empty((natm,3,ncomp,ngrids))
            for d in range(3):
                t[:,d,0] = _atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
                for c in range(1, 4):
                    t[:,d,c] = _atom_pair_sum(aoslices, ao[_D2[d][c-1]], ao_dm0[0])
                    t[:,d,c] += _atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
                if xctype == 'MGGA':
                    t[:,d,4] = sum(_atom_pair_sum(aoslices, ao[_D2[x][d]], ao_dm0[1+x])
                                   for x in range(3))
            for ia in range(natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                if xctype == 'MGGA':
                    wv[:,4] *= .5  # for the factor 1/2 in tau
                vxc_dm[ia] += numpy.einsum('scg,btcg->bst', wv, t)
            ao_dm0 = aow = t = None

    # The atom-independent ipip terms enter vxc[A] through the columns of atom A
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            vxc_dm[ia,ja] += numpy.einsum('xypq,pq->xy', ipip[:,:,q0:q1,p0:p1], dm0[q0:q1,p0:p1])
            if xctype != 'LDA':
                vxc_dm[ia,ja] += numpy.einsum('yxqp,pq->xy', ipip[:,:,p0:p1,q0:q1], dm0[q0:q1,p0:p1])
    return vxc_dm


if __name__ == '__main__':
    import sys, time
    sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
    from molecules import BENZENE, WATER
    from pyscf import gto, dft
    from pyscf.hessian import rks as rks_hess
    for atom, basis in [(WATER, 'def2-svp'), (BENZENE, 'def2-svp')]:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        for xc in ['LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN']:
            mf = dft.RKS(mol, xc=xc); mf.chkfile = None; mf.conv_tol = 1e-10; mf.kernel()
            hobj = mf.Hessian()
            t0 = time.perf_counter()
            vxc = rks_hess._get_vxc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, 4000)
            t1 = time.perf_counter()
            new = _get_vxc_deriv2_dm(hobj, mf.mo_coeff, mf.mo_occ, 4000)
            t2 = time.perf_counter()
            dm0 = mf.make_rdm1()
            aoslices = mol.aoslice_by_atom()
            ref = numpy.zeros_like(new)
            for ia in range(mol.natm):
                for ja in range(mol.natm):
                    q0, q1 = aoslices[ja][2:]
                    ref[ia,ja] = numpy.einsum('xypq,pq->xy', vxc[ia][:,:,q0:q1], dm0[q0:q1])
            err = abs(new - ref).max() / abs(ref).max()
            print(f'{mol.natm:2d} atoms {xc:8s} rel err {err:.1e}   original {t1-t0:6.2f}s  new {t2-t1:6.2f}s', flush=True)


def _get_vxc_deriv2_dm_uks(hessobj, mo_coeff, mo_occ, max_memory):
    '''UKS analogue: (e_a, e_b) with e_s[A,B,x,y] = sum_{p in B} sum_q vxc_s[A,x,y,p,q] dm0_s[p,q].'''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff[0].shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0a, dm0b = mf.make_rdm1(mo_coeff, mo_occ)

    vxc_dm = numpy.zeros((2,natm,natm,3,3))
    ipipa = numpy.zeros((3,3,nao,nao))
    ipipb = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc[:,0]
            aow = numpy.einsum('xpi,p->xpi', ao[1:4], wv[0])
            _d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = numpy.einsum('xpi,p->xpi', ao[1:4], wv[1])
            _d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            wf = weight * fxc[:,0,:,0]
            t = [numpy.array([_atom_pair_sum(aoslices, ao[1+d], ao_dm0)
                              for d in range(3)]).transpose(1,0,2)
                 for ao_dm0 in (ao_dm0a, ao_dm0b)]
            for ia in range(natm):
                p0, p1 = aoslices[ia][2:]
                rho1a = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0a[:,p0:p1]) * 2
                rho1b = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0b[:,p0:p1]) * 2
                wv  = wf[0,:,None] * rho1a
                wv += wf[1,:,None] * rho1b
                for s in range(2):
                    vxc_dm[s,ia] += numpy.einsum('sg,btg->bst', wv[s], t[s])
            ao_dm0a = ao_dm0b = aow = None

    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 2
        ncomp = 4 if xctype == 'GGA' else 5
        nrho = 4 if xctype == 'GGA' else 10
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:nrho], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:nrho], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            if xctype == 'MGGA':
                wv[:,4] *= .25
            for s, ipip in enumerate((ipipa, ipipb)):
                aow = rks_grad._make_dR_dao_w(ao, wv[s])
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                if xctype == 'MGGA':
                    aow = [numint._scale_ao(ao[i], wv[s,4]) for i in range(4, 10)]
                    _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[4], ao[5], ao[6]], mask, ao_loc, False)
                    _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[5], ao[7], ao[8]], mask, ao_loc, False)
                    _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[6], ao[8], ao[9]], mask, ao_loc, False)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            ngrids = weight.size
            t = []
            for ao_dm0 in (ao_dm0a, ao_dm0b):
                ts = numpy.empty((natm,3,ncomp,ngrids))
                for d in range(3):
                    ts[:,d,0] = _atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
                    for c in range(1, 4):
                        ts[:,d,c] = _atom_pair_sum(aoslices, ao[_D2[d][c-1]], ao_dm0[0])
                        ts[:,d,c] += _atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
                    if xctype == 'MGGA':
                        ts[:,d,4] = sum(_atom_pair_sum(aoslices, ao[_D2[x][d]], ao_dm0[1+x])
                                        for x in range(3))
                t.append(ts)
            for ia in range(natm):
                dR_rho1a = _make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = _make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                if xctype == 'MGGA':
                    wv[:,:,4] *= .5
                for s in range(2):
                    vxc_dm[s,ia] += numpy.einsum('scg,btcg->bst', wv[s], t[s])
            ao_dm0a = ao_dm0b = aow = t = None

    for s, (ipip, dm0) in enumerate(((ipipa, dm0a), (ipipb, dm0b))):
        for ia in range(natm):
            p0, p1 = aoslices[ia][2:]
            for ja in range(natm):
                q0, q1 = aoslices[ja][2:]
                vxc_dm[s,ia,ja] += numpy.einsum('xypq,pq->xy', ipip[:,:,q0:q1,p0:p1], dm0[q0:q1,p0:p1])
                if xctype != 'LDA':
                    vxc_dm[s,ia,ja] += numpy.einsum('yxqp,pq->xy', ipip[:,:,p0:p1,q0:q1], dm0[q0:q1,p0:p1])
    return vxc_dm[0], vxc_dm[1]


def _test_uks():
    import sys, time
    sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
    from molecules import WATER
    from pyscf import gto, dft
    from pyscf.hessian import uks as uks_hess
    mol = gto.M(atom=WATER, basis='def2-svp', charge=1, spin=1, verbose=0)
    for xc in ['LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN']:
        mf = dft.UKS(mol, xc=xc); mf.chkfile = None; mf.conv_tol = 1e-10; mf.kernel()
        hobj = mf.Hessian()
        t0 = time.perf_counter()
        vxca, vxcb = uks_hess._get_vxc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, 4000)
        t1 = time.perf_counter()
        new = _get_vxc_deriv2_dm_uks(hobj, mf.mo_coeff, mf.mo_occ, 4000)
        t2 = time.perf_counter()
        aoslices = mol.aoslice_by_atom()
        err = 0
        for vxc, dm0, e in zip((vxca, vxcb), mf.make_rdm1(), new):
            ref = numpy.zeros_like(e)
            for ia in range(mol.natm):
                for ja in range(mol.natm):
                    q0, q1 = aoslices[ja][2:]
                    ref[ia,ja] = numpy.einsum('xypq,pq->xy', vxc[ia][:,:,q0:q1], dm0[q0:q1])
            err = max(err, abs(e - ref).max() / abs(ref).max())
        print(f'UKS H2O+ {xc:8s} rel err {err:.1e}   original {t1-t0:6.2f}s  new {t2-t1:6.2f}s', flush=True)
