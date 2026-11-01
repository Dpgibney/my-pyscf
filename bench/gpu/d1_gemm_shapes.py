"""cuBLAS FP64 throughput of the per-atom GEMMs in the Hessian's _get_vxc_deriv1 (node1): per grid block and atom,
v = aow (3n x g) mo0^T (g x nocc) + ao0 (n x g) mow^T (g x 3 nocc), as in hessian.rks._d1_dot_pair, with n screened
AOs, g grid points and nocc occupied orbitals. Times the atoms one at a time (as the library runs them) and batched:
one GEMM of (natm 3n x g)(g x nocc) and one of (n x g)(g x natm 3 nocc). Random inputs (the cost does not depend on
them). Defaults: caffeine def2-SVP (n 168, nocc 51, g 4096, 24 atoms). Source scripts/node1-gpu-env.sh perf first.
Usage: d1_gemm_shapes.py [--nao 168] [--nocc 51] [--ngrids 4096] [--natm 24] [--repeat 10]"""
import argparse
import time
import numpy
import cupy

ap = argparse.ArgumentParser()
ap.add_argument('--nao', type=int, default=168)
ap.add_argument('--nocc', type=int, default=51)
ap.add_argument('--ngrids', type=int, default=4096)
ap.add_argument('--natm', type=int, default=24)
ap.add_argument('--repeat', type=int, default=10)
args = ap.parse_args()
n, nocc, g, natm = args.nao, args.nocc, args.ngrids, args.natm

rng = cupy.random.default_rng(5)
aow = rng.standard_normal((natm, 3, n, g))
mow = rng.standard_normal((natm, 3, nocc, g))
ao0 = rng.standard_normal((n, g))
mo0 = rng.standard_normal((nocc, g))
flop = natm * 2 * (2 * 3 * n * nocc * g)


def per_atom():
    out = []
    for ia in range(natm):
        v = aow[ia].reshape(3*n, g).dot(mo0.T).reshape(3, n, nocc)
        v += ao0.dot(mow[ia].reshape(3*nocc, g).T).reshape(n, 3, nocc).transpose(1, 0, 2)
        out.append(v)
    return cupy.stack(out)


def batched():
    v = aow.reshape(natm*3*n, g).dot(mo0.T).reshape(natm, 3, n, nocc)
    v += ao0.dot(mow.reshape(natm*3*nocc, g).T).reshape(n, natm, 3, nocc).transpose(1, 2, 0, 3)
    return v


def gemm1_alone():
    return aow[0].reshape(3*n, g).dot(mo0.T)


def gemm2_alone():
    return ao0.dot(mow[0].reshape(3*nocc, g).T)


print(f'n {n}, nocc {nocc}, g {g}, natm {natm}; {flop/1e9:.2f} GFLOP per block')
ref = per_atom()
res = batched()
print(f'  max |batched - per atom| = {abs(res - ref).max():.2e} (max {abs(ref).max():.2e})')
for name, f, fl in (('per atom (library)', per_atom, flop), ('batched over atoms', batched, flop),
                    ('one GEMM1 (3n x g)(g x nocc)', gemm1_alone, 2*3*n*nocc*g),
                    ('one GEMM2 (n x g)(g x 3nocc)', gemm2_alone, 2*3*n*nocc*g)):
    f()
    ts = []
    for _ in range(args.repeat):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        f()
        cupy.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    t = numpy.median(ts)
    print(f'  {name:30s} {1e3*t:8.2f} ms  {fl/t/1e9:6.1f} GFLOP/s')
