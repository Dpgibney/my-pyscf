// VV10's Fock kernel (gdft/vv10.cu, vv10_fock_eval_UWE_kernel) against part 26's symmetric version (node1).
// Phi_ij = -1/(g_ij g_ji (g_ij + g_ji)) with g_ij = omega_i r_ij^2 + kappa_i is symmetric, and so are r_ij^2, the sum and
// the reciprocal; the symmetric kernel evaluates each unordered pair once and adds to both points:
//   E_i += rho_w_j Phi,  U_i += rho_w_j Phi (g_sum + g_ij) g_ji / (g_ij g_ji g_sum),  W_i += U_ij r_ij^2,  and i <-> j.
// Tiles of 128 points; a block takes one I tile against a run of J >= I tiles; threads own the i's and walk the J tile's
// points together; the j side is summed over each warp with shuffles, per warp in shared memory, and added to global
// memory once per tile; the i side once per block. Diagonal tiles take j > i, and the i == j term on the i side only.
// Checks U, W, E against the original on synthetic grid data and times both.
// Build: nvcc -O3 -arch=sm_75 -o vv10_check vv10_check.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NG_PER_BLOCK      128

// the original, verbatim
__global__
static void vv10_fock_eval_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                      const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                      const double* __restrict__ omega, const double* __restrict__ kappa,
                                      const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ngrids;

    double omega_i = NAN;
    double kappa_i = NAN;
    double3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

#define VV10_TILE       128
#define VV10_WARPS      (VV10_TILE / 32)

// The symmetric version: block (I, g) takes the J tiles I + g*jtiles .. I + (g+1)*jtiles - 1 (< ntiles). U, W, E are
// accumulated raw (zeroed before, scaled after by vv10_scale_UWE_kernel)
__global__
static void vv10_fock_eval_UWE_sym_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
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
    // a point past the end takes part with no weight, so that it adds nothing to the j side
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
    double U_i = 0, W_i = 0, E_i = 0;

    __shared__ double3 s_okr[VV10_TILE];     // omega, kappa, rho_weight of the J tile
    __shared__ double3 s_r[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];

    for (int J = J0; J < J1; J++) {
        const int j0 = J * VV10_TILE;
        const int jt = j0 + t;
        if (jt < ngrids) {
            s_okr[t].x = omega[jt];
            s_okr[t].y = kappa[jt];
            s_okr[t].z = rho_weight[jt];
            s_r[t].x = grid_coord[jt * 3 + 0];
            s_r[t].y = grid_coord[jt * 3 + 1];
            s_r[t].z = grid_coord[jt * 3 + 2];
        } else {
            s_okr[t] = make_double3(1., 1., 0.);
            s_r[t] = make_double3(0., 0., 0.);
        }
        __syncthreads();
        const bool diag = J == I;
        const int nj = min(VV10_TILE, ngrids - j0);
        for (int jj = 0; jj < nj; jj++) {
            const double omega_j = s_okr[jj].x;
            const double kappa_j = s_okr[jj].y;
            const double rho_weight_j = s_okr[jj].z;
            const double3 r_j = s_r[jj];
            const double dx = r_i.x - r_j.x;
            const double dy = r_i.y - r_j.y;
            const double dz = r_i.z - r_j.z;
            const double r_ij2 = dx * dx + dy * dy + dz * dz;
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1;
            // the i side: every j of an off-diagonal tile; j >= i in a diagonal one (j == i once)
            const bool take_i = !diag || jj >= t;
            // the j side: j > i only in a diagonal tile
            const bool take_j = !diag || jj > t;
            double E_ij = rho_weight_j * Phi_ij;
            double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            double E_ji = rho_weight_i * Phi_ij;
            double U_ji = E_ji * (g_sum + g_ji) * g_ij * g_ij_ji_sum_1;
            if (take_i) {
                U_i += U_ij;
                W_i += U_ij * r_ij2;
                E_i += E_ij;
            }
            double u = take_j ? U_ji : 0.;
            double w = take_j ? U_ji * r_ij2 : 0.;
            double e = take_j ? E_ji : 0.;
            for (int off = 16; off > 0; off /= 2) {
                u += __shfl_down_sync(0xffffffff, u, off);
                w += __shfl_down_sync(0xffffffff, w, off);
                e += __shfl_down_sync(0xffffffff, e, off);
            }
            if (lane == 0) {
                s_acc[warp][0][jj] = u;
                s_acc[warp][1][jj] = w;
                s_acc[warp][2][jj] = e;
            }
        }
        __syncthreads();
        if (t < nj) {
            double u = 0, w = 0, e = 0;
            for (int k = 0; k < VV10_WARPS; k++) {
                u += s_acc[k][0][t];
                w += s_acc[k][1][t];
                e += s_acc[k][2][t];
            }
            atomicAdd(U + jt, u);
            atomicAdd(W + jt, w);
            atomicAdd(E + jt, e);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(U + i, U_i);
        atomicAdd(W + i, W_i);
        atomicAdd(E + i, E_i);
    }
}

// The symmetric version without per-j reductions: in a warp each lane keeps its i and meets the 32 j's of a chunk in
// turn, j = (lane + s) % 32 at step s, reading their data from shared memory; the j side's three sums travel with their
// j from lane to lane (a shuffle per step), so after 32 steps each is back on its lane, complete for the warp's i's.
__global__
static void vv10_fock_eval_UWE_rot_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
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
    double U_i = 0, W_i = 0, E_i = 0;

    __shared__ double s_x[VV10_TILE], s_y[VV10_TILE], s_z[VV10_TILE];
    __shared__ double s_o[VV10_TILE], s_k[VV10_TILE], s_w[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
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
            double U_j = 0, W_j = 0, E_j = 0;
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
                        const double E_ij = rho_weight_j * Phi_ij;
                        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
                        U_i += U_ij;
                        W_i += U_ij * r_ij2;
                        E_i += E_ij;
                    }
                    if (!diag || jj > t) {
                        const double E_ji = rho_weight_i * Phi_ij;
                        const double U_ji = E_ji * (g_sum + g_ji) * g_ij * g_ij_ji_sum_1;
                        U_j += U_ji;
                        W_j += U_ji * r_ij2;
                        E_j += E_ji;
                    }
                    // the sums of the j this lane meets next, from the lane that has it now
                    U_j = __shfl_sync(0xffffffff, U_j, next);
                    W_j = __shfl_sync(0xffffffff, W_j, next);
                    E_j = __shfl_sync(0xffffffff, E_j, next);
                }
            }
            // after 32 steps the sums of j = c*32 + lane are back on this lane
            s_acc[warp][0][c * 32 + lane] = U_j;
            s_acc[warp][1][c * 32 + lane] = W_j;
            s_acc[warp][2][c * 32 + lane] = E_j;
        }
        __syncthreads();
        if (jt < ngrids) {
            double u = 0, w = 0, e = 0;
            for (int k = 0; k < VV10_WARPS; k++) {
                u += s_acc[k][0][t];
                w += s_acc[k][1][t];
                e += s_acc[k][2][t];
            }
            atomicAdd(U + jt, u);
            atomicAdd(W + jt, w);
            atomicAdd(E + jt, e);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(U + i, U_i);
        atomicAdd(W + i, W_i);
        atomicAdd(E + i, E_i);
    }
}

__global__
static void vv10_scale_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E, const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngrids) {
        U[i] *= -1.5;
        W[i] *= -1.5;
        E[i] *= 1.5;
    }
}

static void run_sym(double *U, double *W, double *E, const double *coord, const double *rw, const double *om,
                    const double *ka, int n, int jtiles, int rot)
{
    cudaMemset(U, 0, n * sizeof(double));
    cudaMemset(W, 0, n * sizeof(double));
    cudaMemset(E, 0, n * sizeof(double));
    int ntiles = (n + VV10_TILE - 1) / VV10_TILE;
    dim3 blocks(ntiles, (ntiles + jtiles - 1) / jtiles);
    if (rot)
        vv10_fock_eval_UWE_rot_kernel<<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
    else
        vv10_fock_eval_UWE_sym_kernel<<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
    vv10_scale_UWE_kernel<<<(n + 255) / 256, 256>>>(U, W, E, n);
}

// U, W, E of point i on the CPU in long double with compensated sums: the reference for both GPU versions
static void reference(const double *h, int n, int i, long double out[3])
{
    const double *x = h, *rw = h + 3 * n, *om = h + 4 * n, *ka = h + 5 * n;
    long double sum[3] = {0, 0, 0}, comp[3] = {0, 0, 0};
    for (int j = 0; j < n; j++) {
        long double dx = (long double)x[i*3] - x[j*3], dy = (long double)x[i*3+1] - x[j*3+1], dz = (long double)x[i*3+2] - x[j*3+2];
        long double r2 = dx*dx + dy*dy + dz*dz;
        long double gij = om[i] * r2 + ka[i], gji = om[j] * r2 + ka[j], gs = gij + gji;
        long double inv = 1.0L / (gij * gji * gs);
        long double Eij = rw[j] * -inv;
        long double Uij = Eij * (gs + gij) * gji * inv;
        long double v[3] = {Uij, Uij * r2, Eij};
        for (int c = 0; c < 3; c++) {
            long double y = v[c] - comp[c], t = sum[c] + y;
            comp[c] = (t - sum[c]) - y;
            sum[c] = t;
        }
    }
    out[0] = -sum[0] * 1.5L; out[1] = -sum[1] * 1.5L; out[2] = sum[2] * 1.5L;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 60000;
    // synthetic grid data in the ranges of caffeine's: points within 8 bohr, omega 0.05-5, kappa 0.1-2, rho*w 1e-6-1e-2
    double *h = (double *)malloc(n * 6 * sizeof(double));
    srand(12345);
    for (int k = 0; k < n; k++) {
        for (int x = 0; x < 3; x++) h[k * 3 + x] = 16. * rand() / RAND_MAX - 8.;
        h[3 * n + k] = 1e-6 * pow(1e4, (double)rand() / RAND_MAX);         // rho * weight
        h[4 * n + k] = 0.05 * pow(100., (double)rand() / RAND_MAX);        // omega
        h[5 * n + k] = 0.1 * pow(20., (double)rand() / RAND_MAX);          // kappa
    }
    double *d, *U0, *W0, *E0, *U1, *W1, *E1;
    cudaMalloc(&d, n * 6 * sizeof(double));
    cudaMemcpy(d, h, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    double *coord = d, *rw = d + 3 * n, *om = d + 4 * n, *ka = d + 5 * n;
    cudaMalloc(&U0, n * 8); cudaMalloc(&W0, n * 8); cudaMalloc(&E0, n * 8);
    cudaMalloc(&U1, n * 8); cudaMalloc(&W1, n * 8); cudaMalloc(&E1, n * 8);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    vv10_fock_eval_UWE_kernel<<<(n + 127) / 128, 128>>>(U0, W0, E0, coord, rw, om, ka, n);
    cudaEventRecord(a);
    vv10_fock_eval_UWE_kernel<<<(n + 127) / 128, 128>>>(U0, W0, E0, coord, rw, om, ka, n);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    printf("n = %d: original %.1f ms\n", n, t0);
    int jt_list[] = {1, 4, 16, 64};
    for (int q = 0; q < 8; q++) {
        int jt = jt_list[q % 4], rot = q >= 4;
        run_sym(U1, W1, E1, coord, rw, om, ka, n, jt, rot);
        cudaEventRecord(a);
        run_sym(U1, W1, E1, coord, rw, om, ka, n, jt, rot);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        double *hu = (double *)malloc(n * 6 * 8);
        cudaMemcpy(hu, U0, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(hu + n, W0, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hu + 2 * n, E0, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(hu + 3 * n, U1, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hu + 4 * n, W1, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(hu + 5 * n, E1, n * 8, cudaMemcpyDeviceToHost);
        double rel[3] = {0, 0, 0};
        for (int k = 0; k < n; k++)
            for (int c = 0; c < 3; c++) {
                double x0 = hu[c * n + k], x1 = hu[(c + 3) * n + k];
                double r = fabs(x1 - x0) / fmax(fabs(x0), 1e-300);
                if (r > rel[c]) rel[c] = r;
            }
        printf("  symmetric (%s), %2d J tiles per block: %.1f ms (%.2fx); max relative difference U %.1e W %.1e E %.1e\n",
               rot ? "rotation" : "reduction", jt, t1, t0 / t1, rel[0], rel[1], rel[2]);
        if (jt == 4) {   // against the long double reference at 256 points
            double e0[3] = {0, 0, 0}, e1[3] = {0, 0, 0};
            for (int q2 = 0; q2 < 256; q2++) {
                int k = (int)((long)q2 * 7919 % n);
                long double r[3];
                reference(h, n, k, r);
                for (int c = 0; c < 3; c++) {
                    double d0 = (double)fabsl((hu[c * n + k] - r[c]) / r[c]);
                    double d1 = (double)fabsl((hu[(c + 3) * n + k] - r[c]) / r[c]);
                    if (d0 > e0[c]) e0[c] = d0;
                    if (d1 > e1[c]) e1[c] = d1;
                }
            }
            printf("    against long double at 256 points: original U %.1e W %.1e E %.1e; this one U %.1e W %.1e E %.1e\n",
                   e0[0], e0[1], e0[2], e1[0], e1[1], e1[2]);
        }
        free(hu);
    }
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
