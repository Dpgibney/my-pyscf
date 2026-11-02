// The Hessian's VV10 response kernel (gdft/vv10.cu, vv10_hess_eval_f_t_offdiagonal_kernel) against part 29's version
// (node1). For each point i and trial t:
//   f_rho_t[t][i]   = 1.5 sum_j w_j (f_rr_ij rho_t[t][j] + f_rg_ij gamma_t[t][j])
//   f_gamma_t[t][i] = 1.5 sum_j w_j (f_gr_ij rho_t[t][j] + f_gg_ij gamma_t[t][j])
// The 2x2 pair matrix takes one division and about 50 other operations; each trial then takes 6 (3 per output).
// The original gives a thread one i and 6 trials, so it evaluates the pair matrix once per 6 trials. Part 29's version
// gives a thread up to 36 trials and folds w_j into the pair matrix: 4 operations per trial, two FMAs per output.
// Checks both against a long double reference at a few points, the new against the original everywhere, and times them.
// Build: nvcc -O3 -arch=sm_75 -o vv10_ft_check vv10_ft_check.cu
// Usage: vv10_ft_check [ngrids 20000] [ntrial 36]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NG_PER_BLOCK      128

// the original, verbatim
template <int n_trial_per_thread>
__global__
static void vv10_hess_eval_f_t_offdiagonal_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                                  const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                  const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                  const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                                  const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                                  const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_trial_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_trial_per_thread;
    if (i >= ngrids || i_trial_start >= ntrial)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    double f_rho_t_i[n_trial_per_thread] {0};
    double f_gamma_t_i[n_trial_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double domega_drho_j = domega_drho[j];
        const double domega_dgamma_j = domega_dgamma[j];
        const double dkappa_drho_j = dkappa_drho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
        const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
        const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_ji_1);

        const double f_rho_rho_ij = Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                              - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                              - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
        const double rho_domegadgamma_i_rij2 = rho_i * domega_dgamma_i * r_ij2;
        const double f_gamma_rho_ij = rho_domegadgamma_i_rij2 * Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
        const double rho_domegadgamma_j_rij2 = rho_j * domega_dgamma_j * r_ij2;
        const double f_rho_gamma_ij = rho_domegadgamma_j_rij2 * Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
        const double f_gamma_gamma_ij = rho_domegadgamma_i_rij2 * rho_domegadgamma_j_rij2 * Phi_ij * d2Phi_dgij_dgji_over_Phi;

        const double weight_j = grid_weight[j];

        #pragma unroll
        for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
            if (i_trial + i_trial_start >= ntrial) continue;
            const double   rho_t_j =   rho_t[(i_trial + i_trial_start) * ngrids + j];
            const double gamma_t_j = gamma_t[(i_trial + i_trial_start) * ngrids + j];
            f_rho_t_i  [i_trial] += weight_j * (  f_rho_rho_ij * rho_t_j +   f_rho_gamma_ij * gamma_t_j);
            f_gamma_t_i[i_trial] += weight_j * (f_gamma_rho_ij * rho_t_j + f_gamma_gamma_ij * gamma_t_j);
        }
    }

    #pragma unroll
    for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
        if (i_trial + i_trial_start >= ntrial) continue;
        f_rho_t  [(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_rho_t_i  [i_trial];
        f_gamma_t[(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_gamma_t_i[i_trial];
    }
}

// part 29: the same pair matrix, with w_j folded in, for up to 36 trials per thread
template <int n_trial_per_thread>
__global__
static void vv10_hess_eval_f_t_offdiagonal_wide_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                                       const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                       const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                       const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                                       const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                                       const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_trial_start = blockIdx.y * n_trial_per_thread;
    if (i >= ngrids)
        return;
    const int n_trial_here = min(n_trial_per_thread, ntrial - i_trial_start);
    rho_t += i_trial_start * ngrids;
    gamma_t += i_trial_start * ngrids;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    double f_rho_t_i[n_trial_per_thread] {0};
    double f_gamma_t_i[n_trial_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double domega_drho_j = domega_drho[j];
        const double domega_dgamma_j = domega_dgamma[j];
        const double dkappa_drho_j = dkappa_drho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
        const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
        const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_ji_1);

        // the pair matrix times w_j
        const double weight_j = grid_weight[j];
        const double w_Phi_ij = weight_j * Phi_ij;
        const double f_rho_rho_ij = w_Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                                - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                                - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
        const double rho_domegadgamma_i_rij2 = rho_i * domega_dgamma_i * r_ij2;
        const double f_gamma_rho_ij = rho_domegadgamma_i_rij2 * w_Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
        const double rho_domegadgamma_j_rij2 = rho_j * domega_dgamma_j * r_ij2;
        const double f_rho_gamma_ij = rho_domegadgamma_j_rij2 * w_Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
        const double f_gamma_gamma_ij = rho_domegadgamma_i_rij2 * rho_domegadgamma_j_rij2 * w_Phi_ij * d2Phi_dgij_dgji_over_Phi;

        #pragma unroll
        for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
            if (i_trial < n_trial_here) {
                const double   rho_t_j =   rho_t[i_trial * ngrids + j];
                const double gamma_t_j = gamma_t[i_trial * ngrids + j];
                f_rho_t_i  [i_trial] = fma(  f_rho_rho_ij, rho_t_j, fma(  f_rho_gamma_ij, gamma_t_j,   f_rho_t_i[i_trial]));
                f_gamma_t_i[i_trial] = fma(f_gamma_rho_ij, rho_t_j, fma(f_gamma_gamma_ij, gamma_t_j, f_gamma_t_i[i_trial]));
            }
        }
    }

    f_rho_t += i_trial_start * ngrids;
    f_gamma_t += i_trial_start * ngrids;
    #pragma unroll
    for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
        if (i_trial < n_trial_here) {
            f_rho_t  [i_trial * ngrids + i] = 1.5 * f_rho_t_i  [i_trial];
            f_gamma_t[i_trial * ngrids + i] = 1.5 * f_gamma_t_i[i_trial];
        }
    }
}

struct Data {
    const double *coord, *w, *rho, *om, *ka, *dodr, *dodg, *dkdr, *rt, *gt;
};

template <int T>
static void run_orig(double *frt, double *fgt, const Data &d, int n, int nt)
{
    dim3 blocks((n + NG_PER_BLOCK - 1) / NG_PER_BLOCK, (nt + T - 1) / T);
    vv10_hess_eval_f_t_offdiagonal_kernel<T><<<blocks, NG_PER_BLOCK>>>(frt, fgt, d.coord, d.w, d.rho, d.om, d.ka,
                                                                       d.dodr, d.dodg, d.dkdr, d.rt, d.gt, n, nt);
}

template <int T>
static void run_wide(double *frt, double *fgt, const Data &d, int n, int nt)
{
    dim3 blocks((n + NG_PER_BLOCK - 1) / NG_PER_BLOCK, (nt + T - 1) / T);
    vv10_hess_eval_f_t_offdiagonal_wide_kernel<T><<<blocks, NG_PER_BLOCK>>>(frt, fgt, d.coord, d.w, d.rho, d.om, d.ka,
                                                                            d.dodr, d.dodg, d.dkdr, d.rt, d.gt, n, nt);
}

// long double reference for point i, trial t: {f_rho_t, f_gamma_t}
static void reference(const double *h, int n, int nt, int i, int t, long double out[2])
{
    const double *x = h, *w = h + 3 * n, *rho = h + 4 * n, *om = h + 5 * n, *ka = h + 6 * n;
    const double *dodr = h + 7 * n, *dodg = h + 8 * n, *dkdr = h + 9 * n, *rt = h + 10 * n, *gt = rt + nt * n;
    long double s0 = 0, s1 = 0, c0 = 0, c1 = 0;
    for (int j = 0; j < n; j++) {
        long double dx = (long double)x[i*3] - x[j*3], dy = (long double)x[i*3+1] - x[j*3+1], dz = (long double)x[i*3+2] - x[j*3+2];
        long double r2 = dx*dx + dy*dy + dz*dz;
        long double gij = om[i] * r2 + ka[i], gji = om[j] * r2 + ka[j], gs = gij + gji;
        long double inv = 1.0L / (gij * gji * gs), Phi = -inv;
        long double gijji_1 = gs * inv, gij_1 = 1.0L / gij, gji_1 = 1.0L / gji, gs_1 = 1.0L / gs;
        long double ai = rho[i] * (r2 * dodr[i] + dkdr[i]), aj = rho[j] * (r2 * dodr[j] + dkdr[j]);
        long double d2 = 2 * (gs_1 * gs_1 + gijji_1);
        long double frr = Phi * (ai * aj * d2 - ai * (gs_1 + gij_1) - aj * (gs_1 + gji_1) + 1);
        long double bi = (long double)rho[i] * dodg[i] * r2, bj = (long double)rho[j] * dodg[j] * r2;
        long double fgr = bi * Phi * (aj * d2 - (gs_1 + gij_1));
        long double frg = bj * Phi * (ai * d2 - (gs_1 + gji_1));
        long double fgg = bi * bj * Phi * d2;
        long double v0 = w[j] * (frr * rt[t * n + j] + frg * gt[t * n + j]);
        long double v1 = w[j] * (fgr * rt[t * n + j] + fgg * gt[t * n + j]);
        long double y0 = v0 - c0, u0 = s0 + y0; c0 = (u0 - s0) - y0; s0 = u0;
        long double y1 = v1 - c1, u1 = s1 + y1; c1 = (u1 - s1) - y1; s1 = u1;
    }
    out[0] = 1.5L * s0; out[1] = 1.5L * s1;
}

int main(int argc, char **argv)
{
    const int n = argc > 1 ? atoi(argv[1]) : 20000;
    const int nt = argc > 2 ? atoi(argv[2]) : 36;
    // synthetic grid data: points within 8 bohr, weights 1e-4-1, rho 1e-5-1, omega 0.05-5, kappa 0.1-2, the
    // derivatives of omega and kappa of either sign, and trial densities of either sign
    const size_t nh = (size_t)n * (10 + 2 * nt);
    double *h = (double *)malloc(nh * sizeof(double));
    srand(12345);
    auto u = []() { return (double)rand() / RAND_MAX; };
    for (int k = 0; k < n; k++) {
        for (int x = 0; x < 3; x++) h[k * 3 + x] = 16. * u() - 8.;
        h[3 * n + k] = 1e-4 * pow(1e4, u());                 // w
        h[4 * n + k] = 1e-5 * pow(1e5, u());                 // rho
        h[5 * n + k] = 0.05 * pow(100., u());                // omega
        h[6 * n + k] = 0.1 * pow(20., u());                  // kappa
        h[7 * n + k] = (2 * u() - 1) * 10.;                  // domega/drho
        h[8 * n + k] = (2 * u() - 1) * 10.;                  // domega/dgamma
        h[9 * n + k] = (2 * u() - 1) * 10.;                  // dkappa/drho
    }
    for (size_t k = (size_t)10 * n; k < nh; k++) h[k] = 2 * u() - 1;
    double *dd;
    cudaMalloc(&dd, nh * sizeof(double));
    cudaMemcpy(dd, h, nh * sizeof(double), cudaMemcpyHostToDevice);
    Data d = {dd, dd + 3 * n, dd + 4 * n, dd + 5 * n, dd + 6 * n, dd + 7 * n, dd + 8 * n, dd + 9 * n, dd + 10 * n,
              dd + (size_t)(10 + nt) * n};
    const size_t no = (size_t)nt * n;
    double *f0, *f1;
    cudaMalloc(&f0, 2 * no * sizeof(double));
    cudaMalloc(&f1, 2 * no * sizeof(double));
    double *h0 = (double *)malloc(2 * no * sizeof(double)), *h1 = (double *)malloc(2 * no * sizeof(double));

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t0, t1;
    run_orig<6>(f0, f0 + no, d, n, nt);
    cudaEventRecord(a);
    run_orig<6>(f0, f0 + no, d, n, nt);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
    cudaMemcpy(h0, f0, 2 * no * sizeof(double), cudaMemcpyDeviceToHost);
    printf("n = %d, ntrial = %d: original (6 trials per thread) %.1f ms\n", n, nt, t0);

    // the reference at 64 (point, trial) pairs
    const int nref = 64;
    long double ref[nref][2];
    int ri[nref], rt[nref];
    for (int q = 0; q < nref; q++) {
        ri[q] = (int)((long)q * 7919 % n);
        rt[q] = q % nt;
        reference(h, n, nt, ri[q], rt[q], ref[q]);
    }
    auto ref_err = [&](const double *hf) {
        double e = 0;
        for (int q = 0; q < nref; q++)
            for (int c = 0; c < 2; c++) {
                const double v = hf[(size_t)c * no + (size_t)rt[q] * n + ri[q]];
                const double r = (double)fabsl((v - ref[q][c]) / ref[q][c]);
                if (r > e) e = r;
            }
        return e;
    };
    printf("  original against long double at %d (point, trial) pairs: max relative error %.1e\n", nref, ref_err(h0));

    struct Variant { const char *name; void (*run)(double *, double *, const Data &, int, int); };
    Variant variants[] = {
        {"original, 12 trials per thread", run_orig<12>},
        {"original, 18 trials per thread", run_orig<18>},
        {"original, 36 trials per thread", run_orig<36>},
        {"w folded in,  6 trials per thread", run_wide<6>},
        {"w folded in, 12 trials per thread", run_wide<12>},
        {"w folded in, 18 trials per thread", run_wide<18>},
        {"w folded in, 24 trials per thread", run_wide<24>},
        {"w folded in, 30 trials per thread", run_wide<30>},
        {"w folded in, 36 trials per thread", run_wide<36>},
    };
    for (auto &v : variants) {
        cudaMemset(f1, 0, 2 * no * sizeof(double));
        v.run(f1, f1 + no, d, n, nt);
        cudaEventRecord(a);
        v.run(f1, f1 + no, d, n, nt);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        cudaMemcpy(h1, f1, 2 * no * sizeof(double), cudaMemcpyDeviceToHost);
        double rel = 0, scale = 0;
        bool same = true;
        for (size_t k = 0; k < 2 * no; k++) scale = fmax(scale, fabs(h0[k]));
        for (size_t k = 0; k < 2 * no; k++) {
            rel = fmax(rel, fabs(h1[k] - h0[k]) / scale);
            same = same && h1[k] == h0[k];
        }
        printf("  %-36s %8.1f ms (%.2fx); %s, max |difference| / max |f| %.1e; against long double %.1e\n", v.name, t1,
               t0 / t1, same ? "bitwise equal" : "differs", rel, ref_err(h1));
    }
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
