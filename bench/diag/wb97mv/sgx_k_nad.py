'''One SGX-K build of NAD+ (wB97M-V/def2-TZVPD, 1916 AOs) from a saved density on SGX grid levels 0, 1 and 2, with
the debug timers of sgx_jk (integrals against the rest). Usage: sgx_k_nad.py DM.npy [LEVEL ...]'''
import sys, time, numpy
from pyscf import gto, dft, df, lib
mol = gto.M(atom='nad+_mmff.xyz', basis='def2-tzvpd', charge=-1, verbose=0)
dm = numpy.load(sys.argv[1])
levels = [int(x) for x in sys.argv[2:]] or [0, 1, 2]
mf = dft.RKS(mol, xc='wb97m-v').RIJCOSX(auxbasis=df.aug_etb(mol))
print(f'{lib.num_threads()} threads, sgx_tol_pair {mf.with_df.sgx_tol_pair}, optk {mf.with_df.optk}', flush=True)
for level in levels:
    mf.with_df.build(level=level)
    mf.with_df.verbose = mol.verbose = 6; mf._nsteps_direct = 0
    mf.with_df.stdout = mol.stdout = sys.stdout
    for rep in range(2):
        t = time.perf_counter(); vk = mf._sgx_exchange(dm); dt = time.perf_counter() - t
        print(f'>>> level {level}, {mf.with_df.grids.weights.size} points, build {rep + 1}: {dt:.1f} s, tr(K D) {numpy.einsum("ij,ji", vk, dm):.8f}', flush=True)
    mf.with_df.verbose = mol.verbose = 0
