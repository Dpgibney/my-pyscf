"""CPU PySCF against GPU4PySCF, analytic Hessians of closed-shell water with PBE/def2-SVP, RKS and UKS (node1): both
SCFs at conv_tol 1e-12, both CPHFs at 1e-10 and on the SCF's grid (GPU4PySCF's default cphf_grids is a coarser
(50, 194) SG-1 grid). Part of the broken-symmetry check (NOTES.md): the two codes' Hessians differ at the 2e-5 level
for any state. Run it with PYTHONPATH set to a GPU4PySCF tree.
Usage: hess_cpu_gpu.py"""
import numpy
import cupy
from pyscf import gto, dft
import pyscf.hessian  # noqa: F401  (gives PySCF's SCF objects their .Hessian)
from gpu4pyscf.dft import rks as grks, uks as guks
mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="def2-svp", verbose=0)
for name, cpu_cls, gpu_cls in (("RKS", dft.RKS, grks.RKS), ("UKS", dft.UKS, guks.UKS)):
    h = []
    for cls, gpu in ((cpu_cls, False), (gpu_cls, True)):
        mf = cls(mol, xc="pbe")
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        if gpu:
            mf.cphf_grids = mf.grids
        mf.kernel()
        x = mf.Hessian().kernel()
        h.append(cupy.asnumpy(x) if isinstance(x, cupy.ndarray) else numpy.asarray(x))
    print(f"{name}: max |H_cpu - H_gpu| {abs(h[0] - h[1]).max():.2e} (max |H| {abs(h[0]).max():.2f})", flush=True)
