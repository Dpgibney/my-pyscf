// The Hessian's VV10 sums U, W, A, B, C, E (gdft/vv10.cu, vv10_hess_eval_UWABCE_kernel) against part 29's version over
// each pair of grid points once (node1). With inv = 1/(g_ij g_ji (g_ij + g_ji)) symmetric, point i gets from j
//   E_ij = rho_w_j Phi_ij,  U_ij = E_ij (g_ij + g_sum) g_ji inv,  W_ij = U_ij r_ij^2,
//   A_ij = E_ij (g_ij^2 + g_ij g_sum + g_sum^2) (g_ji inv)^2,  B_ij = A_ij r_ij^2,  C_ij = B_ij r_ij^2,
// and j gets the same with i <-> j, from the same r_ij^2, g's and reciprocal. The new kernel is part 26's rotation scheme
// (vv10_check.cu) with six sums: tiles of 128 points, a block takes one I tile against VV10_JTILES tiles J >= I, each
// lane keeps its i and meets the 32 j's of a chunk in turn, the j side's sums travelling with their j through the warp;
// atomics into the outputs, then the factors. Checks against the original and a long double reference, and times both.
// Build: nvcc -O3 -arch=sm_75 -o vv10_uwabce_check vv10_uwabce_check.cu
// Usage: vv10_uwabce_check [ngrids 60000]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NG_PER_BLOCK      128

// the original, verbatim
__global__
static void vv10_hess_eval_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                         const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                         const double* __restrict__ omega, const double* __restrict__ kappa,
                                         const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double U_i = 0;
    double W_i = 0;
    double A_i = 0;
    double B_i = 0;
    double C_i = 0;
    double E_i = 0;

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_sum_1 = g_ji * g_ij_ji_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
        const double U_ij = E_ij * (g_ij + g_sum) * g_ij_sum_1;
        const double W_ij = U_ij * r_ij2;
        const double A_ij = E_ij * (g_ij * g_ij + g_ij * g_sum + g_sum * g_sum) * g_ij_sum_1 * g_ij_sum_1;
        const double B_ij = A_ij * r_ij2;
        const double C_ij = B_ij * r_ij2;

        U_i += U_ij;
        W_i += W_ij;
        A_i += A_ij;
        B_i += B_ij;
        C_i += C_ij;
        E_i += E_ij;
    }

    U[i] = -1.5 * U_i;
    W[i] = -1.5 * W_i;
    A[i] = (1.5 * 2) * A_i;
    B[i] = (1.5 * 2) * B_i;
    C[i] = (1.5 * 2) * C_i;
    E[i] = 1.5 * E_i;
}

#define VV10_TILE       128
#define VV10_WARPS      (VV10_TILE / 32)

// part 29: U, W, A, B, C, E over each pair of points once (part 26's rotation; the outputs zeroed before, scaled after)
__global__
static void vv10_hess_eval_UWABCE_rot_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A,
                                             double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                             const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                             const double* __restrict__ omega, const double* __restrict__ kappa,
                                             const int ngrids, const int jtiles)
{
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const int I = blockIdx.x;
    const int J0 = I + blockIdx.y * jtiles;
    if (J0 >= ntiles) {
        return;
    }
    const int J1 = min(J0 + jtiles, ntiles);
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int i = I * VV10_TILE + t;
    double omega_i = 1., kappa_i = 1., rho_weight_i = 0.;
    double3 r_i = {0., 0., 0.};
    if (i < ngrids) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        rho_weight_i = rho_weight[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }
    double U_i = 0, W_i = 0, A_i = 0, B_i = 0, C_i = 0, E_i = 0;

    __shared__ double s_x[VV10_TILE], s_y[VV10_TILE], s_z[VV10_TILE];
    __shared__ double s_o[VV10_TILE], s_k[VV10_TILE], s_w[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][6][VV10_TILE];
    const int next = (lane + 1) % 32;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_x[t] = grid_coord[jt * 3 + 0];
            s_y[t] = grid_coord[jt * 3 + 1];
            s_z[t] = grid_coord[jt * 3 + 2];
            s_o[t] = omega[jt];
            s_k[t] = kappa[jt];
            s_w[t] = rho_weight[jt];
        } else {
            s_x[t] = 0.; s_y[t] = 0.; s_z[t] = 0.;
            s_o[t] = 1.; s_k[t] = 1.; s_w[t] = 0.;
        }
        __syncthreads();
        const bool diag = J == I;
        for (int c = 0; c < VV10_WARPS; c++) {
            double U_j = 0, W_j = 0, A_j = 0, B_j = 0, C_j = 0, E_j = 0;
            // in a diagonal tile the chunks below this warp's are the other warps' (j > i there)
            if (!diag || c >= warp) {
                for (int s = 0; s < 32; s++) {
                    const int jj = c * 32 + (lane + s) % 32;
                    const double dx = r_i.x - s_x[jj];
                    const double dy = r_i.y - s_y[jj];
                    const double dz = r_i.z - s_z[jj];
                    const double omega_j = s_o[jj];
                    const double kappa_j = s_k[jj];
                    const double rho_weight_j = s_w[jj];
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i * r_ij2 + kappa_i;
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_sum = g_ij + g_ji;
                    const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
                    const double Phi_ij = -g_ij_ji_sum_1;
                    // the i side: j >= i in a diagonal tile (j == i once); the j side: j > i there
                    if (!diag || jj >= t) {
                        const double g_ij_sum_1 = g_ji * g_ij_ji_sum_1;
                        const double E_ij = rho_weight_j * Phi_ij;
                        const double U_ij = E_ij * (g_ij + g_sum) * g_ij_sum_1;
                        const double A_ij = E_ij * (g_ij * g_ij + g_ij * g_sum + g_sum * g_sum) * g_ij_sum_1 * g_ij_sum_1;
                        const double B_ij = A_ij * r_ij2;
                        U_i += U_ij;
                        W_i += U_ij * r_ij2;
                        A_i += A_ij;
                        B_i += B_ij;
                        C_i += B_ij * r_ij2;
                        E_i += E_ij;
                    }
                    if (!diag || jj > t) {
                        const double g_ji_sum_1 = g_ij * g_ij_ji_sum_1;
                        const double E_ji = rho_weight_i * Phi_ij;
                        const double U_ji = E_ji * (g_ji + g_sum) * g_ji_sum_1;
                        const double A_ji = E_ji * (g_ji * g_ji + g_ji * g_sum + g_sum * g_sum) * g_ji_sum_1 * g_ji_sum_1;
                        const double B_ji = A_ji * r_ij2;
                        U_j += U_ji;
                        W_j += U_ji * r_ij2;
                        A_j += A_ji;
                        B_j += B_ji;
                        C_j += B_ji * r_ij2;
                        E_j += E_ji;
                    }
                    // the sums of the j this lane meets next, from the lane that has it now
                    U_j = __shfl_sync(0xffffffff, U_j, next);
                    W_j = __shfl_sync(0xffffffff, W_j, next);
                    A_j = __shfl_sync(0xffffffff, A_j, next);
                    B_j = __shfl_sync(0xffffffff, B_j, next);
                    C_j = __shfl_sync(0xffffffff, C_j, next);
                    E_j = __shfl_sync(0xffffffff, E_j, next);
                }
            }
            // after 32 steps the sums of j = c*32 + lane are back on this lane
            s_acc[warp][0][c * 32 + lane] = U_j;
            s_acc[warp][1][c * 32 + lane] = W_j;
            s_acc[warp][2][c * 32 + lane] = A_j;
            s_acc[warp][3][c * 32 + lane] = B_j;
            s_acc[warp][4][c * 32 + lane] = C_j;
            s_acc[warp][5][c * 32 + lane] = E_j;
        }
        __syncthreads();
        if (jt < ngrids) {
            double sum[6] = {0, 0, 0, 0, 0, 0};
            for (int k = 0; k < VV10_WARPS; k++) {
                for (int q = 0; q < 6; q++) {
                    sum[q] += s_acc[k][q][t];
                }
            }
            atomicAdd(U + jt, sum[0]);
            atomicAdd(W + jt, sum[1]);
            atomicAdd(A + jt, sum[2]);
            atomicAdd(B + jt, sum[3]);
            atomicAdd(C + jt, sum[4]);
            atomicAdd(E + jt, sum[5]);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(U + i, U_i);
        atomicAdd(W + i, W_i);
        atomicAdd(A + i, A_i);
        atomicAdd(B + i, B_i);
        atomicAdd(C + i, C_i);
        atomicAdd(E + i, E_i);
    }
}

__global__
static void vv10_scale_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A,
                                     double* __restrict__ B, double* __restrict__ C, double* __restrict__ E, const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngrids) {
        U[i] *= -1.5;
        W[i] *= -1.5;
        A[i] *= 1.5 * 2;
        B[i] *= 1.5 * 2;
        C[i] *= 1.5 * 2;
        E[i] *= 1.5;
    }
}

static void run_rot(double *out, const double *coord, const double *rw, const double *om, const double *ka, int n,
                    int jtiles)
{
    cudaMemset(out, 0, 6 * n * sizeof(double));
    int ntiles = (n + VV10_TILE - 1) / VV10_TILE;
    dim3 blocks(ntiles, (ntiles + jtiles - 1) / jtiles);
    vv10_hess_eval_UWABCE_rot_kernel<<<blocks, VV10_TILE>>>(out, out + n, out + 2 * n, out + 3 * n, out + 4 * n,
                                                            out + 5 * n, coord, rw, om, ka, n, jtiles);
    vv10_scale_UWABCE_kernel<<<(n + 255) / 256, 256>>>(out, out + n, out + 2 * n, out + 3 * n, out + 4 * n, out + 5 * n, n);
}

// U, W, A, B, C, E of point i on the CPU in long double with compensated sums
static void reference(const double *h, int n, int i, long double out[6])
{
    const double *x = h, *rw = h + 3 * n, *om = h + 4 * n, *ka = h + 5 * n;
    long double sum[6] = {0, 0, 0, 0, 0, 0}, comp[6] = {0, 0, 0, 0, 0, 0};
    for (int j = 0; j < n; j++) {
        long double dx = (long double)x[i*3] - x[j*3], dy = (long double)x[i*3+1] - x[j*3+1], dz = (long double)x[i*3+2] - x[j*3+2];
        long double r2 = dx*dx + dy*dy + dz*dz;
        long double gij = om[i] * r2 + ka[i], gji = om[j] * r2 + ka[j], gs = gij + gji;
        long double inv = 1.0L / (gij * gji * gs), g1 = gji * inv;
        long double Eij = rw[j] * -inv;
        long double Uij = Eij * (gij + gs) * g1;
        long double Aij = Eij * (gij * gij + gij * gs + gs * gs) * g1 * g1;
        long double v[6] = {Uij, Uij * r2, Aij, Aij * r2, Aij * r2 * r2, Eij};
        for (int c = 0; c < 6; c++) {
            long double y = v[c] - comp[c], u = sum[c] + y;
            comp[c] = (u - sum[c]) - y;
            sum[c] = u;
        }
    }
    const long double f[6] = {-1.5L, -1.5L, 3.L, 3.L, 3.L, 1.5L};
    for (int c = 0; c < 6; c++) out[c] = f[c] * sum[c];
}

int main(int argc, char **argv)
{
    const int n = argc > 1 ? atoi(argv[1]) : 60000;
    // synthetic grid data in the ranges of caffeine's: points within 8 bohr, omega 0.05-5, kappa 0.1-2, rho*w 1e-6-1e-2
    double *h = (double *)malloc(n * 6 * sizeof(double));
    srand(12345);
    for (int k = 0; k < n; k++) {
        for (int x = 0; x < 3; x++) h[k * 3 + x] = 16. * rand() / RAND_MAX - 8.;
        h[3 * n + k] = 1e-6 * pow(1e4, (double)rand() / RAND_MAX);         // rho * weight
        h[4 * n + k] = 0.05 * pow(100., (double)rand() / RAND_MAX);        // omega
        h[5 * n + k] = 0.1 * pow(20., (double)rand() / RAND_MAX);          // kappa
    }
    double *d, *o0, *o1;
    cudaMalloc(&d, n * 6 * sizeof(double));
    cudaMemcpy(d, h, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    double *coord = d, *rw = d + 3 * n, *om = d + 4 * n, *ka = d + 5 * n;
    cudaMalloc(&o0, 6 * n * sizeof(double));
    cudaMalloc(&o1, 6 * n * sizeof(double));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    auto run_orig = [&]() {
        vv10_hess_eval_UWABCE_kernel<<<(n + 127) / 128, 128>>>(o0, o0 + n, o0 + 2 * n, o0 + 3 * n, o0 + 4 * n, o0 + 5 * n,
                                                               coord, rw, om, ka, n);
    };
    run_orig();
    cudaEventRecord(a);
    run_orig();
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    printf("n = %d: original %.1f ms\n", n, t0);
    double *h0 = (double *)malloc(6 * n * sizeof(double)), *h1 = (double *)malloc(6 * n * sizeof(double));
    cudaMemcpy(h0, o0, 6 * n * sizeof(double), cudaMemcpyDeviceToHost);
    // the reference at 256 points
    const int nref = 256;
    long double (*ref)[6] = (long double (*)[6])malloc(nref * sizeof(*ref));
    int *ri = (int *)malloc(nref * sizeof(int));
    for (int q = 0; q < nref; q++) {
        ri[q] = (int)((long)q * 7919 % n);
        reference(h, n, ri[q], ref[q]);
    }
    const char *names = "UWABCE";
    auto ref_err = [&](const double *hv, double e[6]) {
        for (int c = 0; c < 6; c++) e[c] = 0;
        for (int q = 0; q < nref; q++)
            for (int c = 0; c < 6; c++) {
                const double r = (double)fabsl((hv[c * n + ri[q]] - ref[q][c]) / ref[q][c]);
                if (r > e[c]) e[c] = r;
            }
    };
    double e0[6];
    ref_err(h0, e0);
    printf("  original against long double at %d points:", nref);
    for (int c = 0; c < 6; c++) printf(" %c %.1e", names[c], e0[c]);
    printf("\n");
    const int jt_list[] = {1, 4, 16};
    for (int jt : jt_list) {
        run_rot(o1, coord, rw, om, ka, n, jt);
        cudaEventRecord(a);
        run_rot(o1, coord, rw, om, ka, n, jt);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        cudaMemcpy(h1, o1, 6 * n * sizeof(double), cudaMemcpyDeviceToHost);
        double rel[6] = {0, 0, 0, 0, 0, 0}, e1[6];
        for (int k = 0; k < n; k++)
            for (int c = 0; c < 6; c++) {
                const double x0 = h0[c * n + k], x1 = h1[c * n + k];
                rel[c] = fmax(rel[c], fabs(x1 - x0) / fmax(fabs(x0), 1e-300));
            }
        ref_err(h1, e1);
        printf("  each pair once, %2d J tiles per block: %.1f ms (%.2fx); max relative difference", jt, t1, t0 / t1);
        for (int c = 0; c < 6; c++) printf(" %c %.1e", names[c], rel[c]);
        printf("\n    against long double:");
        for (int c = 0; c < 6; c++) printf(" %c %.1e", names[c], e1[c]);
        printf("\n");
    }
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
