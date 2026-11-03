// VV10's Fock kernel: part 26's rotation kernel (gdft/vv10.cu, vv10_fock_eval_UWE_rot_kernel) against a leaner
// version (node1), with v1.9.0's original as the reference. The rotation kernel's inner loop is 78 instructions per pair,
// 34 of them FP64 (sm_75 SASS): the diagonal tile's tests (j >= i, j > i) branch on every pair, the rotation index takes
// a signed % 32, and g_ij is recomputed in each branch. The lean version:
//   - off-diagonal tiles (all but one in ntiles) run a loop with no tests; the diagonal tile keeps them;
//   - the rotation index is (lane + s) & 31;
//   - with inv = 1/(g_ij g_ji g_sum) and p = g_ij g_ji, U_ij = -rho_w_j inv^2 (g_ji g_sum + p): the sums kept are
//     S_E = sum rho_w_j inv, S_U = sum rho_w_j inv^2 (g_ji g_sum + p), S_W = sum ... r_ij^2, and
//     U = 1.5 S_U, W = 1.5 S_W, E = -1.5 S_E (29 FP64 instructions per pair).
// Two more variants: the tile's values packed as double2 (3 LDS.128 per pair instead of 6 LDS.64), and with that a
// reciprocal from rcp.approx.ftz.f64 and two Newton steps instead of the IEEE division (no slow-path test; bitwise the
// same 1/x, rcp_newton_check.cu): 46.5 instructions per pair, 28 FP64. The library version is the last, between the
// ===== markers (exp_prim_screen.py --vv10-fock lean). Checks all against the original and a long double reference,
// and times them.
// Build: nvcc -O3 -arch=sm_75 -o vv10_fock2_check vv10_fock2_check.cu
// Usage: vv10_fock2_check [ngrids 60000]
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


// part 33: the lean rotation kernel; S_U, S_W, S_E zeroed before, vv10_scale_UWE2_kernel applies the factors
template <bool diag>
__device__ __forceinline__
static void vv10_rot2_chunk(double &U_i, double &W_i, double &E_i, double &U_j, double &W_j, double &E_j,
                            const double3 r_i, const double omega_i, const double kappa_i, const double rho_weight_i,
                            const double *s_x, const double *s_y, const double *s_z, const double *s_o,
                            const double *s_k, const double *s_w, const int c, const int lane, const int t,
                            const int next)
{
    #pragma unroll 2
    for (int s = 0; s < 32; s++) {
        const int jj = c * 32 + ((lane + s) & 31);
        const double dx = r_i.x - s_x[jj];
        const double dy = r_i.y - s_y[jj];
        const double dz = r_i.z - s_z[jj];
        const double r_ij2 = dx * dx + dy * dy + dz * dz;
        const double g_ij = fma(omega_i, r_ij2, kappa_i);
        const double g_ji = fma(s_o[jj], r_ij2, s_k[jj]);
        const double g_sum = g_ij + g_ji;
        const double p = g_ij * g_ji;
        const double inv = 1 / (p * g_sum);
        const double inv2 = inv * inv;
        // the i side: j >= i in a diagonal tile (j == i once); the j side: j > i there
        if (!diag || jj >= t) {
            const double rho_weight_j = s_w[jj];
            const double u = rho_weight_j * inv2 * fma(g_ji, g_sum, p);
            E_i = fma(rho_weight_j, inv, E_i);
            U_i += u;
            W_i = fma(u, r_ij2, W_i);
        }
        if (!diag || jj > t) {
            const double u = rho_weight_i * inv2 * fma(g_ij, g_sum, p);
            E_j = fma(rho_weight_i, inv, E_j);
            U_j += u;
            W_j = fma(u, r_ij2, W_j);
        }
        U_j = __shfl_sync(0xffffffff, U_j, next);
        W_j = __shfl_sync(0xffffffff, W_j, next);
        E_j = __shfl_sync(0xffffffff, E_j, next);
    }
}

__global__
static void vv10_fock_eval_UWE_rot2_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
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
    const int next = (lane + 1) & 31;

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
        for (int c = 0; c < VV10_WARPS; c++) {
            double U_j = 0, W_j = 0, E_j = 0;
            if (J != I) {
                vv10_rot2_chunk<false>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                       s_x, s_y, s_z, s_o, s_k, s_w, c, lane, t, next);
            } else if (c >= warp) {
                // in a diagonal tile the chunks below this warp's are the other warps' (j > i there)
                vv10_rot2_chunk<true>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                      s_x, s_y, s_z, s_o, s_k, s_w, c, lane, t, next);
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
static void vv10_scale_UWE2_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E, const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngrids) {
        U[i] *= 1.5;
        W[i] *= 1.5;
        E[i] *= -1.5;
    }
}

// 1/x from the hardware's approximation and two Newton steps (4 FMAs), without the IEEE division's slow-path test;
// within about 1 ulp. x here is a product of three g's in [0.1, 1e5]: never 0, denormal or infinite.
__device__ __forceinline__ static double rcp_newton_v(double x)
{
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
    double e = fma(-x, r, 1.);
    r = fma(r, e, r);
    e = fma(-x, r, 1.);
    return fma(r, e, r);
}

// the lean chunk with the tile's six values packed as three double2 (x,y | z,omega | kappa,rho_w): 3 LDS.128 per pair
template <bool diag, bool fast_rcp>
__device__ __forceinline__
static void vv10_rot3_chunk(double &U_i, double &W_i, double &E_i, double &U_j, double &W_j, double &E_j,
                            const double3 r_i, const double omega_i, const double kappa_i, const double rho_weight_i,
                            const double2 *s_xy, const double2 *s_zo, const double2 *s_kw, const int c,
                            const int lane, const int t, const int next)
{
    #pragma unroll 2
    for (int s = 0; s < 32; s++) {
        const int jj = c * 32 + ((lane + s) & 31);
        const double2 xy = s_xy[jj], zo = s_zo[jj], kw = s_kw[jj];
        const double dx = r_i.x - xy.x;
        const double dy = r_i.y - xy.y;
        const double dz = r_i.z - zo.x;
        const double r_ij2 = dx * dx + dy * dy + dz * dz;
        const double g_ij = fma(omega_i, r_ij2, kappa_i);
        const double g_ji = fma(zo.y, r_ij2, kw.x);
        const double g_sum = g_ij + g_ji;
        const double p = g_ij * g_ji;
        const double inv = fast_rcp ? rcp_newton_v(p * g_sum) : 1 / (p * g_sum);
        const double inv2 = inv * inv;
        if (!diag || jj >= t) {
            const double u = kw.y * inv2 * fma(g_ji, g_sum, p);
            E_i = fma(kw.y, inv, E_i);
            U_i += u;
            W_i = fma(u, r_ij2, W_i);
        }
        if (!diag || jj > t) {
            const double u = rho_weight_i * inv2 * fma(g_ij, g_sum, p);
            E_j = fma(rho_weight_i, inv, E_j);
            U_j += u;
            W_j = fma(u, r_ij2, W_j);
        }
        U_j = __shfl_sync(0xffffffff, U_j, next);
        W_j = __shfl_sync(0xffffffff, W_j, next);
        E_j = __shfl_sync(0xffffffff, E_j, next);
    }
}

template <bool fast_rcp>
__global__
static void vv10_fock_eval_UWE_rot3_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
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

    __shared__ double2 s_xy[VV10_TILE], s_zo[VV10_TILE], s_kw[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
    const int next = (lane + 1) & 31;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_xy[t] = make_double2(grid_coord[jt * 3 + 0], grid_coord[jt * 3 + 1]);
            s_zo[t] = make_double2(grid_coord[jt * 3 + 2], omega[jt]);
            s_kw[t] = make_double2(kappa[jt], rho_weight[jt]);
        } else {
            s_xy[t] = make_double2(0., 0.);
            s_zo[t] = make_double2(0., 1.);
            s_kw[t] = make_double2(1., 0.);
        }
        __syncthreads();
        for (int c = 0; c < VV10_WARPS; c++) {
            double U_j = 0, W_j = 0, E_j = 0;
            if (J != I) {
                vv10_rot3_chunk<false, fast_rcp>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                                 s_xy, s_zo, s_kw, c, lane, t, next);
            } else if (c >= warp) {
                vv10_rot3_chunk<true, fast_rcp>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                                s_xy, s_zo, s_kw, c, lane, t, next);
            }
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

// ===== the library version (exp_prim_screen.py --vv10-fock lean extracts from here to the end marker) =====
// 1/x from the hardware's approximation and two Newton steps (4 FMAs), without the IEEE division's slow-path test
// (for 0, denormals and infinities, which a product of three g's >= kappa > 0 never is). Bitwise equal to the correctly
// rounded 1/x on 2^28 values each of [1e-300, 1e300] and [1e300, 1e307] (rcp_newton_check.cu).
__device__ __forceinline__ static double rcp_newton(double x)
{
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
    double e = fma(-x, r, 1.);
    r = fma(r, e, r);
    e = fma(-x, r, 1.);
    return fma(r, e, r);
}

// 32 steps of a warp's lanes against one chunk of 32 j's (part 33): the tile's six values packed as three double2
// (x,y | z,omega | kappa,rho_w), 3 LDS.128 per pair; off-diagonal tiles (diag false) run with no tests
template <bool diag>
__device__ __forceinline__
static void vv10_lean_chunk(double &U_i, double &W_i, double &E_i, double &U_j, double &W_j, double &E_j,
                            const double3 r_i, const double omega_i, const double kappa_i, const double rho_weight_i,
                            const double2 *s_xy, const double2 *s_zo, const double2 *s_kw, const int c,
                            const int lane, const int t, const int next)
{
    #pragma unroll 2
    for (int s = 0; s < 32; s++) {
        const int jj = c * 32 + ((lane + s) & 31);
        const double2 xy = s_xy[jj], zo = s_zo[jj], kw = s_kw[jj];
        const double dx = r_i.x - xy.x;
        const double dy = r_i.y - xy.y;
        const double dz = r_i.z - zo.x;
        const double r_ij2 = dx * dx + dy * dy + dz * dz;
        const double g_ij = fma(omega_i, r_ij2, kappa_i);
        const double g_ji = fma(zo.y, r_ij2, kw.x);
        const double g_sum = g_ij + g_ji;
        const double p = g_ij * g_ji;
        const double inv = rcp_newton(p * g_sum);
        const double inv2 = inv * inv;
        if (!diag || jj >= t) {
            const double u = kw.y * inv2 * fma(g_ji, g_sum, p);
            E_i = fma(kw.y, inv, E_i);
            U_i += u;
            W_i = fma(u, r_ij2, W_i);
        }
        if (!diag || jj > t) {
            const double u = rho_weight_i * inv2 * fma(g_ij, g_sum, p);
            E_j = fma(rho_weight_i, inv, E_j);
            U_j += u;
            W_j = fma(u, r_ij2, W_j);
        }
        U_j = __shfl_sync(0xffffffff, U_j, next);
        W_j = __shfl_sync(0xffffffff, W_j, next);
        E_j = __shfl_sync(0xffffffff, E_j, next);
    }
}

// part 33: VV10's Fock kernel over each pair once (part 26's rotation) with a lean inner loop: no per-pair tests in the
// off-diagonal tiles, the rotation index by & 31, packed loads, the reciprocal without the IEEE slow path, and the sums
// S_E = sum rho_w_j inv, S_U = sum rho_w_j inv^2 (g_ji g_sum + g_ij g_ji), S_W = sum ... r_ij^2 (zeroed before;
// vv10_scale_UWE_lean_kernel makes U = 1.5 S_U, W = 1.5 S_W, E = -1.5 S_E)
__global__
static void vv10_fock_eval_UWE_lean_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
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

    __shared__ double2 s_xy[VV10_TILE], s_zo[VV10_TILE], s_kw[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
    const int next = (lane + 1) & 31;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_xy[t] = make_double2(grid_coord[jt * 3 + 0], grid_coord[jt * 3 + 1]);
            s_zo[t] = make_double2(grid_coord[jt * 3 + 2], omega[jt]);
            s_kw[t] = make_double2(kappa[jt], rho_weight[jt]);
        } else {
            s_xy[t] = make_double2(0., 0.);
            s_zo[t] = make_double2(0., 1.);
            s_kw[t] = make_double2(1., 0.);
        }
        __syncthreads();
        for (int c = 0; c < VV10_WARPS; c++) {
            double U_j = 0, W_j = 0, E_j = 0;
            if (J != I) {
                vv10_lean_chunk<false>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                       s_xy, s_zo, s_kw, c, lane, t, next);
            } else if (c >= warp) {
                // in a diagonal tile the chunks below this warp's are the other warps' (j > i there)
                vv10_lean_chunk<true>(U_i, W_i, E_i, U_j, W_j, E_j, r_i, omega_i, kappa_i, rho_weight_i,
                                      s_xy, s_zo, s_kw, c, lane, t, next);
            }
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
static void vv10_scale_UWE_lean_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                       const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngrids) {
        U[i] *= 1.5;
        W[i] *= 1.5;
        E[i] *= -1.5;
    }
}
// ===== end of the library version =====

static void run_rot(double *U, double *W, double *E, const double *coord, const double *rw, const double *om,
                    const double *ka, int n, int jtiles, int version)
{
    cudaMemset(U, 0, n * sizeof(double));
    cudaMemset(W, 0, n * sizeof(double));
    cudaMemset(E, 0, n * sizeof(double));
    int ntiles = (n + VV10_TILE - 1) / VV10_TILE;
    dim3 blocks(ntiles, (ntiles + jtiles - 1) / jtiles);
    if (version == 1) {
        vv10_fock_eval_UWE_rot_kernel<<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
        vv10_scale_UWE_kernel<<<(n + 255) / 256, 256>>>(U, W, E, n);
    } else {
        if (version == 2)
            vv10_fock_eval_UWE_rot2_kernel<<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
        else if (version == 3)
            vv10_fock_eval_UWE_rot3_kernel<false><<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
        else if (version == 4)
            vv10_fock_eval_UWE_rot3_kernel<true><<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
        else
            vv10_fock_eval_UWE_lean_kernel<<<blocks, VV10_TILE>>>(U, W, E, coord, rw, om, ka, n, jtiles);
        if (version < 5)
            vv10_scale_UWE2_kernel<<<(n + 255) / 256, 256>>>(U, W, E, n);
        else
            vv10_scale_UWE_lean_kernel<<<(n + 255) / 256, 256>>>(U, W, E, n);
    }
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
    double *d, *o;
    cudaMalloc(&d, n * 6 * sizeof(double));
    cudaMemcpy(d, h, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    double *coord = d, *rw = d + 3 * n, *om = d + 4 * n, *ka = d + 5 * n;
    cudaMalloc(&o, 18 * n * sizeof(double));
    double *U0 = o, *W0 = o + n, *E0 = o + 2 * n;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    vv10_fock_eval_UWE_kernel<<<(n + 127) / 128, 128>>>(U0, W0, E0, coord, rw, om, ka, n);
    cudaEventRecord(a);
    vv10_fock_eval_UWE_kernel<<<(n + 127) / 128, 128>>>(U0, W0, E0, coord, rw, om, ka, n);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    printf("n = %d: original (v1.9.0) %.1f ms\n", n, t0);
    double *hu = (double *)malloc(18 * n * sizeof(double));
    const int nref = 256;
    long double (*ref)[3] = (long double (*)[3])malloc(nref * sizeof(*ref));
    for (int q = 0; q < nref; q++)
        reference(h, n, (int)((long)q * 7919 % n), ref[q]);
    float tv[6] = {0, 0, 0, 0, 0, 0};
    for (int version = 1; version <= 5; version++) {
        double *Uv = o + 3 * version * n, *Wv = Uv + n, *Ev = Uv + 2 * n;
        run_rot(Uv, Wv, Ev, coord, rw, om, ka, n, 4, version);
        cudaEventRecord(a);
        run_rot(Uv, Wv, Ev, coord, rw, om, ka, n, 4, version);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        tv[version] = t1;
    }
    cudaMemcpy(hu, o, 18 * n * sizeof(double), cudaMemcpyDeviceToHost);
    const char *names[6] = {"original", "part 26 (rotation)", "lean", "lean, packed loads", "lean, packed, fast rcp",
                            "library version"};
    for (int version = 0; version < 6; version++) {
        double rel[3] = {0, 0, 0}, err[3] = {0, 0, 0};
        for (int k = 0; k < n; k++)
            for (int c = 0; c < 3; c++) {
                double x0 = hu[c * n + k], x1 = hu[(3 * version + c) * n + k];
                rel[c] = fmax(rel[c], fabs(x1 - x0) / fmax(fabs(x0), 1e-300));
            }
        for (int q = 0; q < nref; q++) {
            int k = (int)((long)q * 7919 % n);
            for (int c = 0; c < 3; c++)
                err[c] = fmax(err[c], (double)fabsl((hu[(3 * version + c) * n + k] - ref[q][c]) / ref[q][c]));
        }
        if (version == 0)
            printf("  %-24s against long double at %d points: U %.1e W %.1e E %.1e\n", names[0], nref, err[0], err[1], err[2]);
        else
            printf("  %-24s %8.1f ms (%.2fx the original, %.2fx part 26); against the original U %.1e W %.1e E %.1e; "
                   "against long double U %.1e W %.1e E %.1e\n", names[version], tv[version], t0 / tv[version],
                   tv[1] / tv[version], rel[0], rel[1], rel[2], err[0], err[1], err[2]);
    }
    cudaError_t errc = cudaGetLastError();
    printf("%s\n", errc == cudaSuccess ? "ok" : cudaGetErrorString(errc));
    return 0;
}
