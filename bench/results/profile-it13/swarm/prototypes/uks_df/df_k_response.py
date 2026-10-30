'''DF K of TDA-shaped response dms (dm_k = Cvir X_k^T Cocc^T, hermi=0) through df_jk.get_jk,
against a factored evaluation K_k = sum_P (B_P L_k)(B_P Cocc)^T with L_k = Cvir X_k^T, from the
same _cderi. 1 thread.'''
import sys, time
import numpy
from pyscf import gto, dft, lib
from pyscf.df import df_jk

which = sys.argv[1] if len(sys.argv) > 1 else 'benzene'
if which == 'benzene':
    atom = '''C 0.000 1.396 0.000; C 1.209 0.698 0.000; C 1.209 -0.698 0.000;
C 0.000 -1.396 0.000; C -1.209 -0.698 0.000; C -1.209 0.698 0.000;
H 0.000 2.479 0.000; H 2.147 1.240 0.000; H 2.147 -1.240 0.000;
H 0.000 -2.479 0.000; H -2.147 -1.240 0.000; H -2.147 1.240 0.000'''
else:
    atom = '''C 0.000 0.710 0.000; C 0.000 -0.710 0.000; C 1.242 1.403 0.000; C 1.242 -1.403 0.000;
C -1.242 1.403 0.000; C -1.242 -1.403 0.000; C 2.433 0.705 0.000; C 2.433 -0.705 0.000;
C -2.433 0.705 0.000; C -2.433 -0.705 0.000; H 1.244 2.490 0.000; H 1.244 -2.490 0.000;
H -1.244 2.490 0.000; H -1.244 -2.490 0.000; H 3.377 1.245 0.000; H 3.377 -1.245 0.000;
H -3.377 1.245 0.000; H -3.377 -1.245 0.000'''
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='b3lyp').density_fit()
dfobj = mf.with_df
dfobj.build()
nao = mol.nao
naux = dfobj.get_naoaux()
nocc = mol.nelectron // 2
nvir = nao - nocc
print(f'{which}: nao {nao}, naux {naux}, nocc {nocc}, threads {lib.num_threads()}')

rng = numpy.random.default_rng(0)
c = numpy.linalg.qr(rng.standard_normal((nao, nao)))[0]   # stand-in orthonormal orbitals
orbo, orbv = c[:, :nocc], c[:, nocc:]
nset = 10
xs = rng.standard_normal((nset, nocc, nvir))
dms = lib.einsum('xov,pv,qo->xpq', xs, orbv, orbo)   # as tdscf/rhf.py vind (TDA)

def best(f, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r

t_jk, (vj, vk) = best(lambda: df_jk.get_jk(dfobj, dms, hermi=0))
t_j, _ = best(lambda: df_jk.get_jk(dfobj, dms, hermi=0, with_k=False))
print(f'df_jk.get_jk, {nset} dms: J+K {t_jk:.3f} s ({t_jk/nset*1e3:.0f} ms/dm), J only {t_j:.3f} s')

def k_factored():
    # K_k[m,n] = sum_P sum_{ls} B_P[m,l] dm_k[l,s] B_P[s,n], dm_k = L_k orbo^T
    L = lib.einsum('pv,xov->xpo', orbv, xs)            # [nset,nao,nocc]
    vk = numpy.zeros((nset, nao, nao))
    for eri1 in dfobj.loop(blksize=240):
        nP = eri1.shape[0]
        B = lib.unpack_tril(eri1).reshape(nP, nao, nao)    # once per block for all dms
        BR = lib.dot(B.reshape(-1, nao), orbo).reshape(nP, nao, nocc)  # shared
        BRt = BR.transpose(1, 0, 2).reshape(nao, nP*nocc)
        for k in range(nset):
            BL = lib.dot(B.reshape(-1, nao), L[k]).reshape(nP, nao, nocc)
            lib.dot(BL.transpose(1, 0, 2).reshape(nao, nP*nocc), BRt.T, 1, vk[k], 1)
    return vk
t_kf, vkf = best(k_factored)
print(f'factored K, {nset} dms: {t_kf:.3f} s ({t_kf/nset*1e3:.0f} ms/dm); '
      f'max|dK| {abs(vkf-vk).max():.1e} (max|K| {abs(vk).max():.1e})')

# MO-basis K for TDA: Kov_k = Cocc^T K_k Cvir = sum_P (Cocc^T B_P Cocc) X_k ... not in AO basis;
# report the flop ratios instead
print('flops per dm: current 4*naux*nao^3 = %.2e; factored 4*naux*nao^2*nocc = %.2e (ratio %.2f)'
      % (4*naux*nao**3, 4*naux*nao**2*nocc, nocc/nao))
