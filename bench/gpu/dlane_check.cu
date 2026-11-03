// The density phase of a multi-density J+K build, as GPU4PySCF's Rys kernels do it against "density lanes" (node1;
// part 34, part 17's option 2). Only the memory pattern is modelled: the integrals of a quartet come from a cheap hash
// (in the real kernels they sit in registers after the Rys evaluation), so the time is that of loading the densities,
// the contractions and the atomic adds.
//   current: a thread per quartet (ij|kl); for each density d it loads the six density blocks D_lk, D_ji, D_jk, D_jl,
//            D_ik, D_il from D[d][nao][nao], contracts, and adds to J_ij, J_kl, K_il, K_ik, K_jl, K_jk with one FP64
//            atomic per element (6 nf^2 per quartet and density).
//   lanes:   a block takes one bra pair ij and its ket pairs kl sorted by k; a batch of quartets' integrals is staged in
//            shared memory, then each lane takes one density (D[nao][nao][n_dm], densities fastest: loads and atomics
//            are coalesced across lanes) and walks the batch: J_ij stays in registers for the whole block, K_ik and K_jk
//            while k repeats, J_kl, K_il, K_jl are added per quartet.
// Shells: nsh shells of one angular momentum (nf functions each) for all four indices; every bra pair i >= j takes
// every ket pair k >= l with (kl) <= (ij), as an s8 loop does. Checks lanes against current and times both.
// Build: nvcc -O3 -arch=sm_75 -o dlane_check dlane_check.cu
// Usage: dlane_check [nf 3] [nsh 60] [n_dm 64]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAXNF 6

// a quartet's integral (a b|c e): a cheap, deterministic stand-in
__device__ __forceinline__ double gval(int task, int a, int b, int c, int e)
{
    unsigned h = (unsigned)task * 2654435761u ^ (unsigned)(((a * 7 + b) * 7 + c) * 7 + e) * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return (double)(h & 0xffff) * (1.0 / 65536.0) - 0.5;
}

// tasks: for each bra pair ij (i >= j) every ket pair kl (k >= l, kl <= ij); pair index p = i*(i+1)/2 + j
struct Tasks { int npairs; int *pair_i, *pair_j; };

template <int NF>
__global__ void current_kernel(double *vj, double *vk, const double *dm, int nsh, int n_dm, int npairs,
                               const int *pair_i, const int *pair_j)
{
    // one thread per quartet (pij, pkl), pkl <= pij; quartets enumerated over a 2D grid: blockIdx.y = pij
    const int pij = blockIdx.y;
    const int pkl = blockIdx.x * blockDim.x + threadIdx.x;
    if (pkl > pij) return;
    const int nao = nsh * NF;
    const size_t nao2 = (size_t)nao * nao;
    const int i0 = pair_i[pij] * NF, j0 = pair_j[pij] * NF, k0 = pair_i[pkl] * NF, l0 = pair_j[pkl] * NF;
    const int task = pij * npairs + pkl;
    double g[NF][NF][NF][NF];
#pragma unroll
    for (int a = 0; a < NF; a++)
#pragma unroll
    for (int b = 0; b < NF; b++)
#pragma unroll
    for (int c = 0; c < NF; c++)
#pragma unroll
    for (int e = 0; e < NF; e++) g[a][b][c][e] = gval(task, a, b, c, e);
    for (int d = 0; d < n_dm; d++) {
        const double *D = dm + d * nao2;
        double *J = vj + d * nao2, *K = vk + d * nao2;
        // J_ij += (ij|kl) D_lk; J_kl += (ij|kl) D_ji
#pragma unroll
        for (int a = 0; a < NF; a++)
#pragma unroll
        for (int b = 0; b < NF; b++) {
            double s = 0;
#pragma unroll
            for (int c = 0; c < NF; c++)
#pragma unroll
            for (int e = 0; e < NF; e++) s += g[a][b][c][e] * D[(l0 + e) * nao + k0 + c];
            atomicAdd(J + (i0 + a) * nao + j0 + b, s);
        }
#pragma unroll
        for (int c = 0; c < NF; c++)
#pragma unroll
        for (int e = 0; e < NF; e++) {
            double s = 0;
#pragma unroll
            for (int a = 0; a < NF; a++)
#pragma unroll
            for (int b = 0; b < NF; b++) s += g[a][b][c][e] * D[(j0 + b) * nao + i0 + a];
            atomicAdd(J + (k0 + c) * nao + l0 + e, s);
        }
        // K_il += (ij|kl) D_jk, K_ik += D_jl, K_jl += D_ik, K_jk += D_il
#pragma unroll
        for (int a = 0; a < NF; a++)
#pragma unroll
        for (int e = 0; e < NF; e++) {
            double s = 0;
#pragma unroll
            for (int b = 0; b < NF; b++)
#pragma unroll
            for (int c = 0; c < NF; c++) s += g[a][b][c][e] * D[(j0 + b) * nao + k0 + c];
            atomicAdd(K + (i0 + a) * nao + l0 + e, s);
        }
#pragma unroll
        for (int a = 0; a < NF; a++)
#pragma unroll
        for (int c = 0; c < NF; c++) {
            double s = 0;
#pragma unroll
            for (int b = 0; b < NF; b++)
#pragma unroll
            for (int e = 0; e < NF; e++) s += g[a][b][c][e] * D[(j0 + b) * nao + l0 + e];
            atomicAdd(K + (i0 + a) * nao + k0 + c, s);
        }
#pragma unroll
        for (int b = 0; b < NF; b++)
#pragma unroll
        for (int e = 0; e < NF; e++) {
            double s = 0;
#pragma unroll
            for (int a = 0; a < NF; a++)
#pragma unroll
            for (int c = 0; c < NF; c++) s += g[a][b][c][e] * D[(i0 + a) * nao + k0 + c];
            atomicAdd(K + (j0 + b) * nao + l0 + e, s);
        }
#pragma unroll
        for (int b = 0; b < NF; b++)
#pragma unroll
        for (int c = 0; c < NF; c++) {
            double s = 0;
#pragma unroll
            for (int a = 0; a < NF; a++)
#pragma unroll
            for (int e = 0; e < NF; e++) s += g[a][b][c][e] * D[(i0 + a) * nao + l0 + e];
            atomicAdd(K + (j0 + b) * nao + k0 + c, s);
        }
    }
}

// density lanes: block = (32 lanes) x (n_dm / 32 warps); each block takes one bra pair pij and walks its ket pairs
// pkl = 0..pij (k-major: pairs of one k are consecutive) in batches of BATCH quartets staged in shared memory
#define BATCH 16
template <int NF>
__global__ void lanes_kernel(double *vj, double *vk, const double *dmT, int nsh, int n_dm, int npairs,
                             const int *pair_i, const int *pair_j)
{
    const int pij = blockIdx.x;
    const int d = threadIdx.y * 32 + threadIdx.x;   // this lane's density
    const int tid = d;
    const int nthreads = blockDim.x * blockDim.y;
    const int nao = nsh * NF;
    const int i0 = pair_i[pij] * NF, j0 = pair_j[pij] * NF;
    __shared__ double gs[BATCH][NF * NF * NF * NF];
    double Jij[NF][NF], Kik[NF][NF], Kjk[NF][NF];
#pragma unroll
    for (int a = 0; a < NF; a++)
#pragma unroll
    for (int b = 0; b < NF; b++) { Jij[a][b] = 0; Kik[a][b] = 0; Kjk[a][b] = 0; }
    int kcur = -1;
#define DT(row, col) dmT[((size_t)(row) * nao + (col)) * n_dm + d]
#define VJ(row, col) (vj + ((size_t)(row) * nao + (col)) * n_dm + d)
#define VK(row, col) (vk + ((size_t)(row) * nao + (col)) * n_dm + d)
    auto flush_k = [&](int k0) {
#pragma unroll
        for (int a = 0; a < NF; a++)
#pragma unroll
        for (int c = 0; c < NF; c++) {
            atomicAdd(VK(i0 + a, k0 + c), Kik[a][c]);
            atomicAdd(VK(j0 + a, k0 + c), Kjk[a][c]);
            Kik[a][c] = 0; Kjk[a][c] = 0;
        }
    };
    for (int q0 = 0; q0 <= pij; q0 += BATCH) {
        const int nq = min(BATCH, pij + 1 - q0);
        __syncthreads();
        for (int x = tid; x < nq * NF * NF * NF * NF; x += nthreads) {
            const int q = x / (NF * NF * NF * NF), r = x % (NF * NF * NF * NF);
            const int a = r / (NF * NF * NF), b = r / (NF * NF) % NF, c = r / NF % NF, e = r % NF;
            gs[q][r] = gval(pij * npairs + q0 + q, a, b, c, e);
        }
        __syncthreads();
        if (d < n_dm) {
            for (int q = 0; q < nq; q++) {
                const int pkl = q0 + q;
                const int k0 = pair_i[pkl] * NF, l0 = pair_j[pkl] * NF;
                if (k0 != kcur) {
                    if (kcur >= 0) flush_k(kcur);
                    kcur = k0;
                }
                const double *g = gs[q];
#define G(a, b, c, e) g[(((a) * NF + (b)) * NF + (c)) * NF + (e)]
                // J_ij in registers
#pragma unroll
                for (int a = 0; a < NF; a++)
#pragma unroll
                for (int b = 0; b < NF; b++) {
                    double s = 0;
#pragma unroll
                    for (int c = 0; c < NF; c++)
#pragma unroll
                    for (int e = 0; e < NF; e++) s += G(a, b, c, e) * DT(l0 + e, k0 + c);
                    Jij[a][b] += s;
                }
#pragma unroll
                for (int c = 0; c < NF; c++)
#pragma unroll
                for (int e = 0; e < NF; e++) {
                    double s = 0;
#pragma unroll
                    for (int a = 0; a < NF; a++)
#pragma unroll
                    for (int b = 0; b < NF; b++) s += G(a, b, c, e) * DT(j0 + b, i0 + a);
                    atomicAdd(VJ(k0 + c, l0 + e), s);
                }
#pragma unroll
                for (int a = 0; a < NF; a++)
#pragma unroll
                for (int e = 0; e < NF; e++) {
                    double s = 0;
#pragma unroll
                    for (int b = 0; b < NF; b++)
#pragma unroll
                    for (int c = 0; c < NF; c++) s += G(a, b, c, e) * DT(j0 + b, k0 + c);
                    atomicAdd(VK(i0 + a, l0 + e), s);
                }
#pragma unroll
                for (int a = 0; a < NF; a++)
#pragma unroll
                for (int c = 0; c < NF; c++) {
                    double s = 0;
#pragma unroll
                    for (int b = 0; b < NF; b++)
#pragma unroll
                    for (int e = 0; e < NF; e++) s += G(a, b, c, e) * DT(j0 + b, l0 + e);
                    Kik[a][c] += s;
                }
#pragma unroll
                for (int b = 0; b < NF; b++)
#pragma unroll
                for (int e = 0; e < NF; e++) {
                    double s = 0;
#pragma unroll
                    for (int a = 0; a < NF; a++)
#pragma unroll
                    for (int c = 0; c < NF; c++) s += G(a, b, c, e) * DT(i0 + a, k0 + c);
                    atomicAdd(VK(j0 + b, l0 + e), s);
                }
#pragma unroll
                for (int b = 0; b < NF; b++)
#pragma unroll
                for (int c = 0; c < NF; c++) {
                    double s = 0;
#pragma unroll
                    for (int a = 0; a < NF; a++)
#pragma unroll
                    for (int e = 0; e < NF; e++) s += G(a, b, c, e) * DT(i0 + a, l0 + e);
                    Kjk[b][c] += s;
                }
#undef G
            }
        }
    }
    if (d < n_dm) {
        if (kcur >= 0) flush_k(kcur);
#pragma unroll
        for (int a = 0; a < NF; a++)
#pragma unroll
        for (int b = 0; b < NF; b++) atomicAdd(VJ(i0 + a, j0 + b), Jij[a][b]);
    }
#undef DT
#undef VJ
#undef VK
}

template <int NF>
static void run(int nsh, int n_dm)
{
    const int nao = nsh * NF;
    const int npairs = nsh * (nsh + 1) / 2;
    int *hp = (int *)malloc(2 * npairs * sizeof(int));
    for (int i = 0, p = 0; i < nsh; i++)
        for (int j = 0; j <= i; j++, p++) { hp[p] = i; hp[npairs + p] = j; }   // pairs k-major: (i, j<=i)
    int *dp;
    cudaMalloc(&dp, 2 * npairs * sizeof(int));
    cudaMemcpy(dp, hp, 2 * npairs * sizeof(int), cudaMemcpyHostToDevice);
    const size_t n = (size_t)n_dm * nao * nao;
    double *hdm = (double *)malloc(n * sizeof(double)), *hdmT = (double *)malloc(n * sizeof(double));
    srand(7);
    for (int d = 0; d < n_dm; d++)
        for (int r = 0; r < nao; r++)
            for (int c = 0; c <= r; c++) {
                const double v = 2.0 * rand() / RAND_MAX - 1;
                hdm[((size_t)d * nao + r) * nao + c] = hdm[((size_t)d * nao + c) * nao + r] = v;
            }
    for (int d = 0; d < n_dm; d++)
        for (size_t rc = 0; rc < (size_t)nao * nao; rc++) hdmT[rc * n_dm + d] = hdm[(size_t)d * nao * nao + rc];
    double *ddm, *ddmT, *vj0, *vk0, *vj1, *vk1;
    cudaMalloc(&ddm, n * 8); cudaMalloc(&ddmT, n * 8);
    cudaMalloc(&vj0, n * 8); cudaMalloc(&vk0, n * 8); cudaMalloc(&vj1, n * 8); cudaMalloc(&vk1, n * 8);
    cudaMemcpy(ddm, hdm, n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(ddmT, hdmT, n * 8, cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    dim3 g0((npairs + 127) / 128, npairs);
    auto cur = [&]() {
        cudaMemset(vj0, 0, n * 8); cudaMemset(vk0, 0, n * 8);
        current_kernel<NF><<<g0, 128>>>(vj0, vk0, ddm, nsh, n_dm, npairs, dp, dp + npairs);
    };
    dim3 t1d(32, (n_dm + 31) / 32);
    auto lan = [&]() {
        cudaMemset(vj1, 0, n * 8); cudaMemset(vk1, 0, n * 8);
        lanes_kernel<NF><<<npairs, t1d>>>(vj1, vk1, ddmT, nsh, n_dm, npairs, dp, dp + npairs);
    };
    cur(); cudaEventRecord(a); cur(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    lan(); cudaEventRecord(a); lan(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
    // compare: vj1/vk1 are [nao][nao][n_dm]
    double *h0 = (double *)malloc(2 * n * 8), *h1 = (double *)malloc(2 * n * 8);
    cudaMemcpy(h0, vj0, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(h0 + n, vk0, n * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1, vj1, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(h1 + n, vk1, n * 8, cudaMemcpyDeviceToHost);
    double diff = 0, mx = 0;
    for (int m = 0; m < 2; m++)
        for (int dd = 0; dd < n_dm; dd++)
            for (size_t rc = 0; rc < (size_t)nao * nao; rc++) {
                const double x0 = h0[m * n + (size_t)dd * nao * nao + rc], x1 = h1[m * n + rc * n_dm + dd];
                diff = fmax(diff, fabs(x1 - x0));
                mx = fmax(mx, fabs(x0));
            }
    const double quartets = (double)npairs * (npairs + 1) / 2;
    printf("nf %d, %d shells (nao %d), %d densities, %.3g quartets: current %8.1f ms, lanes %8.1f ms (%.2fx); "
           "max |difference| %.1e (max %.1e)\n", NF, nsh, nao, n_dm, quartets, t0, t1, t0 / t1, diff, mx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaFree(ddm); cudaFree(ddmT); cudaFree(vj0); cudaFree(vk0); cudaFree(vj1); cudaFree(vk1); cudaFree(dp);
    free(hp); free(hdm); free(hdmT); free(h0); free(h1);
}

int main(int argc, char **argv)
{
    const int nf = argc > 1 ? atoi(argv[1]) : 3;
    const int nsh = argc > 2 ? atoi(argv[2]) : 60;
    const int n_dm = argc > 3 ? atoi(argv[3]) : 64;
    if (n_dm % 32) { printf("n_dm must be a multiple of 32\n"); return 1; }
    if (nf == 1) run<1>(nsh, n_dm);
    else if (nf == 3) run<3>(nsh, n_dm);
    else { printf("nf 1 or 3\n"); return 1; }
    return 0;
}
