// The Hessian's VV10 sums with the grid weights' derivatives (gdft/vv10.cu, vv10_hess_eval_EUW_with_weight1_kernel)
// against part 29's wide version (node1). For each point i and derivative d (3 natm of them):
//   Ew[d][i] = 1.5 sum_j w1[d][j] E_ij,  Uw[d][i] = -1.5 sum_j w1[d][j] U_ij,  Ww[d][i] = -1.5 sum_j w1[d][j] W_ij
// with E_ij = rho_j Phi_ij, U_ij, W_ij as in the Fock kernel: about 25 operations per pair including a division, then
// 3 FMAs per derivative. The original gives a thread 6 derivatives, so it evaluates the pair once per 6 derivatives;
// the wide version takes up to 24 (with 3 sums each, 36 would not fit in the registers). Same arithmetic per output, so
// the results are bitwise equal. Checks and times both.
// Build: nvcc -O3 -arch=sm_75 -o vv10_euw_w1_check vv10_euw_w1_check.cu
// Usage: vv10_euw_w1_check [ngrids 20000] [nderivative 36]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NG_PER_BLOCK      128

// the original, verbatim
template <int n_derivative_per_thread>
__global__
static void vv10_hess_eval_EUW_with_weight1_kernel(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                   const int ngrids, const int nderivative)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_derivative_per_thread;
    if (i >= ngrids || i_derivative_start >= nderivative)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double Ew_i[n_derivative_per_thread] {0};
    double Uw_i[n_derivative_per_thread] {0};
    double Ww_i[n_derivative_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
        const double W_ij = U_ij * r_ij2;

        #pragma unroll
        for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
            if (i_derivative + i_derivative_start >= nderivative) continue;
            const double weight_j = grid_weight1[(i_derivative + i_derivative_start) * ngrids + j];
            Ew_i[i_derivative] += weight_j * E_ij;
            Uw_i[i_derivative] += weight_j * U_ij;
            Ww_i[i_derivative] += weight_j * W_ij;
        }
    }

    #pragma unroll
    for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
        if (i_derivative + i_derivative_start >= nderivative) continue;
        Ew[(i_derivative + i_derivative_start) * ngrids + i] =  1.5 * Ew_i[i_derivative];
        Uw[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Uw_i[i_derivative];
        Ww[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Ww_i[i_derivative];
    }
}

// part 29: up to 24 derivatives per thread
template <int n_derivative_per_thread>
__global__
static void vv10_hess_eval_EUW_with_weight1_wide_kernel(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                   const int ngrids, const int nderivative)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_start = blockIdx.y * n_derivative_per_thread;
    if (i >= ngrids)
        return;
    const int n_derivative_here = min(n_derivative_per_thread, nderivative - i_derivative_start);

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double Ew_i[n_derivative_per_thread] {0};
    double Uw_i[n_derivative_per_thread] {0};
    double Ww_i[n_derivative_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
        const double W_ij = U_ij * r_ij2;

        #pragma unroll
        for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
            if (i_derivative >= n_derivative_here) continue;
            const double weight_j = grid_weight1[(i_derivative + i_derivative_start) * ngrids + j];
            Ew_i[i_derivative] += weight_j * E_ij;
            Uw_i[i_derivative] += weight_j * U_ij;
            Ww_i[i_derivative] += weight_j * W_ij;
        }
    }

    #pragma unroll
    for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
        if (i_derivative >= n_derivative_here) continue;
        Ew[(i_derivative + i_derivative_start) * ngrids + i] =  1.5 * Ew_i[i_derivative];
        Uw[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Uw_i[i_derivative];
        Ww[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Ww_i[i_derivative];
    }
}


struct Data {
    const double *coord, *w1, *rho, *om, *ka;
};

template <int T>
static void run_orig(double *out, const Data &d, int n, int nd)
{
    dim3 blocks((n + NG_PER_BLOCK - 1) / NG_PER_BLOCK, (nd + T - 1) / T);
    const size_t no = (size_t)nd * n;
    vv10_hess_eval_EUW_with_weight1_kernel<T><<<blocks, NG_PER_BLOCK>>>(out, out + no, out + 2 * no, d.coord, d.w1,
                                                                        d.rho, d.om, d.ka, n, nd);
}

template <int T>
static void run_wide(double *out, const Data &d, int n, int nd)
{
    dim3 blocks((n + NG_PER_BLOCK - 1) / NG_PER_BLOCK, (nd + T - 1) / T);
    const size_t no = (size_t)nd * n;
    vv10_hess_eval_EUW_with_weight1_wide_kernel<T><<<blocks, NG_PER_BLOCK>>>(out, out + no, out + 2 * no, d.coord,
                                                                             d.w1, d.rho, d.om, d.ka, n, nd);
}

int main(int argc, char **argv)
{
    const int n = argc > 1 ? atoi(argv[1]) : 20000;
    const int nd = argc > 2 ? atoi(argv[2]) : 36;
    // synthetic grid data: points within 8 bohr, rho 1e-5-1, omega 0.05-5, kappa 0.1-2, weight derivatives of either sign
    const size_t nh = (size_t)n * (6 + nd);
    double *h = (double *)malloc(nh * sizeof(double));
    srand(12345);
    auto u = []() { return (double)rand() / RAND_MAX; };
    for (int k = 0; k < n; k++) {
        for (int x = 0; x < 3; x++) h[k * 3 + x] = 16. * u() - 8.;
        h[3 * n + k] = 1e-5 * pow(1e5, u());                 // rho
        h[4 * n + k] = 0.05 * pow(100., u());                // omega
        h[5 * n + k] = 0.1 * pow(20., u());                  // kappa
    }
    for (size_t k = (size_t)6 * n; k < nh; k++) h[k] = 2 * u() - 1;
    double *dd;
    cudaMalloc(&dd, nh * sizeof(double));
    cudaMemcpy(dd, h, nh * sizeof(double), cudaMemcpyHostToDevice);
    Data d = {dd, dd + 6 * n, dd + 3 * n, dd + 4 * n, dd + 5 * n};
    const size_t no = (size_t)nd * n;
    double *o0, *o1;
    cudaMalloc(&o0, 3 * no * sizeof(double));
    cudaMalloc(&o1, 3 * no * sizeof(double));
    double *h0 = (double *)malloc(3 * no * sizeof(double)), *h1 = (double *)malloc(3 * no * sizeof(double));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    run_orig<6>(o0, d, n, nd);
    cudaEventRecord(a);
    run_orig<6>(o0, d, n, nd);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    cudaMemcpy(h0, o0, 3 * no * sizeof(double), cudaMemcpyDeviceToHost);
    printf("n = %d, nderivative = %d: original (6 derivatives per thread) %.1f ms\n", n, nd, t0);
    struct Variant { const char *name; void (*run)(double *, const Data &, int, int); };
    Variant variants[] = {
        {"original, 12 derivatives per thread", run_orig<12>},
        {"original, 24 derivatives per thread", run_orig<24>},
        {"wide,  6 derivatives per thread", run_wide<6>},
        {"wide, 12 derivatives per thread", run_wide<12>},
        {"wide, 18 derivatives per thread", run_wide<18>},
        {"wide, 24 derivatives per thread", run_wide<24>},
    };
    for (auto &v : variants) {
        cudaMemset(o1, 0, 3 * no * sizeof(double));
        v.run(o1, d, n, nd);
        cudaEventRecord(a);
        v.run(o1, d, n, nd);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        cudaMemcpy(h1, o1, 3 * no * sizeof(double), cudaMemcpyDeviceToHost);
        bool same = true;
        for (size_t k = 0; k < 3 * no; k++) same = same && h1[k] == h0[k];
        printf("  %-34s %8.1f ms (%.2fx); %s\n", v.name, t1, t0 / t1, same ? "bitwise equal" : "DIFFERS");
    }
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
