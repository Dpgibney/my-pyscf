"""contract('yjg,xjg->xyj') of the RKS/UKS Hessian XC terms (hessian.rks._get_vxc_deriv2), for (3, nao_sub, blk)
inputs on node1: cuTENSOR (gpu4pyscf's contract), batched cupy.matmul, nine products with sums, and a fused
RawKernel (one block per AO), with their differences. Source scripts/node1-gpu-env.sh perf first.
Usage: xyj_bench.py"""
import time, cupy, numpy
from gpu4pyscf.lib.cupy_helper import contract
nsub, blk = 246, 4096
rng = numpy.random.default_rng(1)
A = cupy.asarray(rng.standard_normal((3, nsub, blk)))   # ao_mask[1:4]
B = cupy.asarray(rng.standard_normal((3, nsub, blk)))   # aow
C0 = cupy.asarray(rng.standard_normal((3, 3, nsub)))
def bench(name, f, n=20):
    f(); cupy.cuda.Device().synchronize(); t0 = time.perf_counter()
    for _ in range(n): r = f()
    cupy.cuda.Device().synchronize(); dt = (time.perf_counter() - t0) / n
    print(f'{name:40s} {dt*1e3:8.3f} ms'); return r
ref = bench('contract yjg,xjg->xyj (cuTENSOR)', lambda: contract('yjg,xjg->xyj', A, B))
r1 = bench('cupy.matmul batched over j', lambda: cupy.matmul(B.transpose(1, 0, 2), A.transpose(1, 2, 0)).transpose(1, 2, 0))
r2 = bench('9 x (product).sum(axis=-1)', lambda: cupy.stack([cupy.stack([(B[x] * A[y]).sum(axis=-1) for y in range(3)]) for x in range(3)]))
kern = cupy.RawKernel(r'''
extern "C" __global__ void xyj(const double *A, const double *B, double *C, int nsub, int blk)
{
    // one block per AO j: the 9 sums over g of B[x,j,g] * A[y,j,g]
    int j = blockIdx.x;
    double s[9] = {0};
    for (int g = threadIdx.x; g < blk; g += blockDim.x) {
        double a0 = A[(0*nsub+j)*blk+g], a1 = A[(1*nsub+j)*blk+g], a2 = A[(2*nsub+j)*blk+g];
        double b0 = B[(0*nsub+j)*blk+g], b1 = B[(1*nsub+j)*blk+g], b2 = B[(2*nsub+j)*blk+g];
        s[0] += b0*a0; s[1] += b0*a1; s[2] += b0*a2;
        s[3] += b1*a0; s[4] += b1*a1; s[5] += b1*a2;
        s[6] += b2*a0; s[7] += b2*a1; s[8] += b2*a2;
    }
    __shared__ double red[9][256];
    for (int k = 0; k < 9; ++k) red[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) for (int k = 0; k < 9; ++k) red[k][threadIdx.x] += red[k][threadIdx.x+w];
        __syncthreads();
    }
    if (threadIdx.x < 9) C[threadIdx.x*nsub+j] += red[threadIdx.x][0];
}''', 'xyj')
def raw():
    C = cupy.zeros((3, 3, nsub))
    kern((nsub,), (256,), (A, B, C, cupy.int32(nsub), cupy.int32(blk)))
    return C
r3 = bench('RawKernel, one block per AO', raw)
for name, r in (('matmul', r1), ('sum', r2), ('raw', r3)):
    print(f'  {name}: max |r - cuTENSOR| = {abs(r - ref).max():.2e} (max {abs(ref).max():.2e})')
bytes_ = 2 * 3 * nsub * blk * 8
print(f'  memory traffic per call {bytes_/1e6:.0f} MB: {bytes_/192e9*1e3:.3f} ms at 192 GB/s')
