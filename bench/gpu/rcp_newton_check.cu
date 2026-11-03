// The reciprocal of vv10_fock2_check.cu's lean VV10 kernel (rcp.approx.ftz.f64 and two Newton steps, part 33) against
// the IEEE 1/x (node1): over 2^28 values log-uniform in [LO, HI], the count of results that differ from the correctly
// rounded reciprocal and the largest difference in ulps.
// Build: nvcc -O3 -arch=sm_75 -o rcp_newton_check rcp_newton_check.cu
// Usage: rcp_newton_check [LO 1e-300] [HI 1e300]
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
__device__ __forceinline__ static double rcp_newton(double x)
{
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
    double e = fma(-x, r, 1.);
    r = fma(r, e, r);
    e = fma(-x, r, 1.);
    return fma(r, e, r);
}
__global__ void k(unsigned long long *nbad, unsigned long long *maxulp, long n, double lo, double hi)
{
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    // a log-uniform x from a hash of i
    unsigned long long h = (unsigned long long)i * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    double u = (h >> 11) * (1.0 / 9007199254740992.0);
    double x = exp(log(lo) + u * (log(hi) - log(lo)));
    double a = 1.0 / x, b = rcp_newton(x);
    long long d = llabs(__double_as_longlong(a) - __double_as_longlong(b));
    if (d) atomicAdd(nbad, 1ull);
    atomicMax(maxulp, (unsigned long long)d);
}
#include <stdlib.h>
int main(int argc, char **argv)
{
    long n = 1l << 28;
    double lo = argc > 1 ? atof(argv[1]) : 1e-300, hi = argc > 2 ? atof(argv[2]) : 1e300;
    unsigned long long *d, h[2];
    cudaMalloc(&d, 16); cudaMemset(d, 0, 16);
    k<<<(n + 255) / 256, 256>>>(d, d + 1, n, lo, hi);
    cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
    printf("%ld values in [%g, %g]: %llu differ from the correctly rounded 1/x, by at most %llu ulp\n", n, lo, hi, h[0],
           h[1]);
    return 0;
}
