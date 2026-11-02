// The Boys function of GPU4PySCF's MD J engine (gvhf-rys/gamma_inc.cu, eval_gamma_inc_fn) against part 25's version
// (node1): the series loop and the downward recursion multiply by tabulated 1/(n + 1/2) instead of dividing, and the
// erf branch skips erf(sqrt(t)) where it is 1 in double precision. Checks
//   1. that CUDA's erf(x) is exactly 1 for x >= sqrt(ERF_ONE_T), on a dense grid up to 40;
//   2. the two versions' F_n(t), n <= m, over m = 0..18 and t on a dense grid of [0, 80), in ulps per branch;
//   3. the time per evaluation of each version (one thread per t, all lanes of a warp in the same branch or mixed).
// Build: nvcc -O3 -arch=sm_75 -o boys_check boys_check.cu (the table is generated below, as gamma_inc.cu's patch has it)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define EPS_FLOAT64     DBL_EPSILON
#define SQRTPIE4        .886226925452758013
#define MD_BOYS_NINV    96
#define ERF_ONE_T       49.
#define MMAX            18

__constant__ double c_inv_half[MD_BOYS_NINV];

__device__ static void boys_old(double *f, double t, int m)
{
    if (t < EPS_FLOAT64) {
        f[0] = 1.;
        for (int i = 1; i <= m; i++) f[i] = 1./(2*i+1);
    } else if (m > 0 && t < m*.5+.5) {
        double bi = m + .5;
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        while (x > tol) {
            bi += 1.;
            x *= t / bi;
            s += x;
        }
        double b = m + 0.5;
        double fval = s / b;
        f[m] = fval;
        for (int i = m-1; i >= 0; i--) {
            b -= 1.;
            fval = (e + t * fval) / b;
            f[i] = fval;
        }
    } else {
        double tt = sqrt(t);
        double fval = SQRTPIE4 / tt * erf(tt);
        f[0] = fval;
        if (m > 0) {
            double e = .5 * exp(-t);
            double b = 1. / t;
            double b1 = .5;
            for (int i = 1; i <= m; i++) {
                fval = b * (b1 * fval - e);
                f[i] = fval;
                b1 += 1.;
            }
        }
    }
}

__device__ static void boys_new(double *f, double t, int m)
{
    if (t < EPS_FLOAT64) {
        f[0] = 1.;
        for (int i = 1; i <= m; i++) f[i] = 1./(2*i+1);
    } else if (m > 0 && t < m*.5+.5) {
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        for (int k = m+1; x > tol && k < MD_BOYS_NINV; k++) {
            x *= t * c_inv_half[k];
            s += x;
        }
        double fval = s * c_inv_half[m];
        f[m] = fval;
        for (int i = m-1; i >= 0; i--) {
            fval = (e + t * fval) * c_inv_half[i];
            f[i] = fval;
        }
    } else {
        double tt = sqrt(t);
        double fval = SQRTPIE4 / tt;
        if (t < ERF_ONE_T) {
            fval *= erf(tt);
        }
        f[0] = fval;
        if (m > 0) {
            double e = .5 * exp(-t);
            double b = 1. / t;
            double b1 = .5;
            for (int i = 1; i <= m; i++) {
                fval = b * (b1 * fval - e);
                f[i] = fval;
                b1 += 1.;
            }
        }
    }
}

__global__ void erf_one(double x0, double dx, long n, unsigned long long *bad, double *first_bad)
{
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    for (; i < n; i += (long)gridDim.x * blockDim.x) {
        double x = x0 + i * dx;
        if (erf(x) != 1.) {
            atomicAdd(bad, 1ull);
            *first_bad = x;
        }
    }
}

// ulp distance of two positive doubles
__device__ static long long ulps(double a, double b)
{
    long long ia = __double_as_longlong(a), ib = __double_as_longlong(b);
    return ia > ib ? ia - ib : ib - ia;
}

__global__ void compare(int m, double t0, double dt, int n, unsigned long long *max_ulp, unsigned long long *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double t = t0 + i * dt;
    double fo[MMAX+1], fn[MMAX+1];
    boys_old(fo, t, m);
    boys_new(fn, t, m);
    // branch: 0 tiny t, 1 series, 2 erf below ERF_ONE_T, 3 erf skipped
    int br = t < EPS_FLOAT64 ? 0 : (m > 0 && t < m*.5+.5) ? 1 : (t < ERF_ONE_T ? 2 : 3);
    for (int k = 0; k <= m; k++) {
        long long u = ulps(fo[k], fn[k]);
        atomicMax(max_ulp + br, (unsigned long long)u);
        if (u) atomicAdd(count + br, 1ull);
    }
}

template <int NEW>
__global__ void timing(int m, double t0, double dt, int n, int reps, double *sink)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double acc = 0.;
    double f[MMAX+1];
    for (int r = 0; r < reps; r++) {
        double t = t0 + ((i * 7919 + r * 104729) % n) * dt;   // mixed t across the warp, as in the J kernels
        if (NEW) boys_new(f, t, m); else boys_old(f, t, m);
        acc += f[m];
    }
    sink[i] = acc;
}

int main()
{
    double h_inv[MD_BOYS_NINV];
    for (int k = 0; k < MD_BOYS_NINV; k++) h_inv[k] = 1. / (k + .5);
    cudaMemcpyToSymbol(c_inv_half, h_inv, sizeof(h_inv));
    unsigned long long *d_bad, h_bad;
    double *d_first, h_first = 0;
    cudaMalloc(&d_bad, 8); cudaMalloc(&d_first, 8);
    cudaMemset(d_bad, 0, 8);
    long n = 1L << 30;
    double x0 = sqrt(ERF_ONE_T), x1 = 40.;
    erf_one<<<4096, 256>>>(x0, (x1 - x0) / n, n, d_bad, d_first);
    cudaMemcpy(&h_bad, d_bad, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_first, d_first, 8, cudaMemcpyDeviceToHost);
    printf("erf(x) != 1 for %llu of %ld x in [%.6f, %.1f]\n", h_bad, n, x0, x1);
    if (h_bad) printf("  e.g. x = %.17g\n", h_first);

    unsigned long long *d_ulp, *d_cnt, h_ulp[4], h_cnt[4];
    cudaMalloc(&d_ulp, 32); cudaMalloc(&d_cnt, 32);
    int nt = 1 << 22;
    double tmax = 80.;
    printf(" m   max ulps (tiny t, series, erf, erf skipped)   values differing (same order)\n");
    for (int m = 0; m <= MMAX; m++) {
        cudaMemset(d_ulp, 0, 32); cudaMemset(d_cnt, 0, 32);
        compare<<<(nt + 255) / 256, 256>>>(m, 0., tmax / nt, nt, d_ulp, d_cnt);
        cudaMemcpy(h_ulp, d_ulp, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cnt, d_cnt, 32, cudaMemcpyDeviceToHost);
        printf("%2d   %4llu %4llu %4llu %4llu        %llu %llu %llu %llu\n", m, h_ulp[0], h_ulp[1], h_ulp[2], h_ulp[3],
               h_cnt[0], h_cnt[1], h_cnt[2], h_cnt[3]);
    }

    double *d_sink;
    int ntime = 1 << 20;
    cudaMalloc(&d_sink, ntime * sizeof(double));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf(" m   t range     old ns/eval   new ns/eval   (one thread per t, t mixed within warps)\n");
    int ms[] = {1, 3, 6, 9};
    double ranges[][2] = {{0., 4.}, {0., 40.}, {0., 200.}};
    for (int a = 0; a < 4; a++) for (int b = 0; b < 3; b++) {
        int m = ms[a];
        float to, tn;
        int reps = 32;
        double dt = (ranges[b][1] - ranges[b][0]) / ntime;
        timing<0><<<ntime / 256, 256>>>(m, ranges[b][0], dt, ntime, reps, d_sink);
        cudaEventRecord(e0);
        timing<0><<<ntime / 256, 256>>>(m, ranges[b][0], dt, ntime, reps, d_sink);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&to, e0, e1);
        cudaEventRecord(e0);
        timing<1><<<ntime / 256, 256>>>(m, ranges[b][0], dt, ntime, reps, d_sink);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&tn, e0, e1);
        printf("%2d   %5.0f-%-5.0f %10.3f    %10.3f\n", m, ranges[b][0], ranges[b][1],
               to * 1e6 / ((double)ntime * reps), tn * 1e6 / ((double)ntime * reps));
    }
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
