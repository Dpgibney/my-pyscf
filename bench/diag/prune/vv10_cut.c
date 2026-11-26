/* The two pair kernels of the VV10 Hessian with a cutoff between blocks of grid points: an experiment for
 * bench/diag/prune/prune_check.py, not part of PySCF. VV10CUT_eval_f_t and VV10CUT_eval_EUW_with_weight1 are
 * pyscf/lib/dft/nr_numint.c's VXC_vv10nlc_hessian_eval_f_t and VXC_vv10nlc_hessian_eval_EUW_with_weight1 (perf
 * 8b430d696), except that the pairs of two blocks of VV10_BI points are left out when the bounding spheres of the
 * blocks are more than rcut apart (VV10CUT_set_rcut). The decision is the same for (i, j) and (j, i). With rcut <= 0
 * every pair is kept and the kernels do the operations of the originals in their order. The caller orders the points
 * so that the blocks are compact in space.
 *
 * gcc -Wall -msse3 -O2 -g -DNDEBUG -std=gnu99 -fPIC -fopenmp -O3 -march=x86-64-v3 -ffp-contract=off -shared \
 *     -o libvv10cut.so vv10_cut.c -L$LIB -lopenblas    (the flags of nr_numint.c in the tree's build) */
#include <stdlib.h>
#include <math.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define VV10_BI 128
#define VV10_BJ 256

void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);

static volatile int _done = 0;
static volatile int _total = 0;
static double _rcut = 0;
static long _kept = 0;
static long _all = 0;

void VV10CUT_set_rcut(double rcut) { _rcut = rcut; }
void VV10CUT_progress(int *done_total) { done_total[0] = _done; done_total[1] = _total; }
/* the block pairs (VV10_BI x VV10_BI) of the last call: kept, all */
void VV10CUT_stats(long *kept_all) { kept_all[0] = _kept; kept_all[1] = _all; }

/* centre and radius of each block of VV10_BI points */
static double *block_spheres(const double *coord, int n)
{
        int nb = (n + VV10_BI - 1) / VV10_BI;
        double *sph = malloc(sizeof(double) * 4 * (nb + 1));
        for (int b = 0; b < nb; b++) {
                int p0 = b * VV10_BI;
                int p1 = MIN(n, p0 + VV10_BI);
                double cx = 0, cy = 0, cz = 0, r2 = 0;
                for (int p = p0; p < p1; p++) {
                        cx += coord[p*3+0];
                        cy += coord[p*3+1];
                        cz += coord[p*3+2];
                }
                cx /= (p1 - p0);
                cy /= (p1 - p0);
                cz /= (p1 - p0);
                for (int p = p0; p < p1; p++) {
                        double dx = coord[p*3+0] - cx;
                        double dy = coord[p*3+1] - cy;
                        double dz = coord[p*3+2] - cz;
                        double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 > r2) r2 = d2;
                }
                sph[b*4+0] = cx;
                sph[b*4+1] = cy;
                sph[b*4+2] = cz;
                sph[b*4+3] = sqrt(r2);
        }
        return sph;
}

static int pair_kept(const double *sph, int a, int b)
{
        if (_rcut <= 0) {
                return 1;
        }
        double dx = sph[a*4+0] - sph[b*4+0];
        double dy = sph[a*4+1] - sph[b*4+1];
        double dz = sph[a*4+2] - sph[b*4+2];
        return sqrt(dx * dx + dy * dy + dz * dz) - sph[a*4+3] - sph[b*4+3] <= _rcut;
}

/* the contiguous range [*js, *js + *bj) of the points j0:j0+bj0 (bj0 <= VV10_BJ = 2 VV10_BI) whose blocks pair
 * with the block ib; 0 if none. With both halves kept the whole range, as the original kernels. */
static int kept_range(const double *sph, int ib, int j0, int bj0, int *js, int *bj, long *nkept, long *nall)
{
        int jb = j0 / VV10_BI;
        int k0 = pair_kept(sph, ib, jb);
        int two = bj0 > VV10_BI;
        int k1 = two ? pair_kept(sph, ib, jb + 1) : 0;
        *nall += 1 + two;
        *nkept += k0 + k1;
        if (k0 && (k1 || !two)) {
                *js = j0;
                *bj = bj0;
                return 1;
        } else if (k0) {
                *js = j0;
                *bj = VV10_BI;
                return 1;
        } else if (k1) {
                *js = j0 + VV10_BI;
                *bj = bj0 - VV10_BI;
                return 1;
        }
        return 0;
}

void VV10CUT_eval_f_t(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                      const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                      const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                      const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                      const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                      const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                      const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                      const int ngrids, const int ntrial)
{
    if (ngrids <= 0 || ntrial <= 0) {
        return;
    }
    const char TRANS_N = 'N';
    const double D1 = 1;
    const int ld = ngrids;
    const int nt = ntrial;
    const size_t nblk = VV10_BI * VV10_BJ;
    double *sph = block_spheres(grid_coord, ngrids);
    _done = 0;
    _total = ngrids;
    _kept = 0;
    _all = 0;

#pragma omp parallel
{
    double *K = malloc(sizeof(double) * (4 * nblk + 9 * VV10_BI));
    double *Krr = K;
    double *Krg = Krr + nblk;
    double *Kgr = Krg + nblk;
    double *Kgg = Kgr + nblk;
    double *xi = Kgg + nblk;
    double *yi = xi + VV10_BI;
    double *zi = yi + VV10_BI;
    double *omega_i = zi + VV10_BI;
    double *kappa_i = omega_i + VV10_BI;
    double *rho_i = kappa_i + VV10_BI;
    double *domega_drho_i = rho_i + VV10_BI;
    double *dkappa_drho_i = domega_drho_i + VV10_BI;
    double *rho_domega_dgamma_i = dkappa_drho_i + VV10_BI;

#pragma omp for schedule(dynamic, 1)
    for (int i0 = 0; i0 < ngrids; i0 += VV10_BI) {
        const int bi = MIN(VV10_BI, ngrids - i0);
        long nkept = 0, nall = 0;
        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            xi[ii] = grid_coord[i * 3 + 0];
            yi[ii] = grid_coord[i * 3 + 1];
            zi[ii] = grid_coord[i * 3 + 2];
            omega_i[ii] = omega[i];
            kappa_i[ii] = kappa[i];
            rho_i[ii] = rho[i];
            domega_drho_i[ii] = domega_drho[i];
            dkappa_drho_i[ii] = dkappa_drho[i];
            rho_domega_dgamma_i[ii] = rho[i] * domega_dgamma[i];
        }
        for (int t = 0; t < nt; t++) {
            for (int ii = 0; ii < bi; ii++) {
                f_rho_t  [(size_t)t * ld + i0 + ii] = 0;
                f_gamma_t[(size_t)t * ld + i0 + ii] = 0;
            }
        }

        for (int j00 = 0; j00 < ngrids; j00 += VV10_BJ) {
            int j0, bj;
            if (!kept_range(sph, i0 / VV10_BI, j00, MIN(VV10_BJ, ngrids - j00), &j0, &bj, &nkept, &nall)) {
                continue;
            }
            for (int jj = 0; jj < bj; jj++) {
                const int j = j0 + jj;
                const double xj = grid_coord[j * 3 + 0];
                const double yj = grid_coord[j * 3 + 1];
                const double zj = grid_coord[j * 3 + 2];
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double rho_j = rho[j];
                const double domega_drho_j = domega_drho[j];
                const double dkappa_drho_j = dkappa_drho[j];
                const double rho_domega_dgamma_j = rho_j * domega_dgamma[j];
                const double weight_j = grid_weight[j];
                double *krr = Krr + jj * bi;
                double *krg = Krg + jj * bi;
                double *kgr = Kgr + jj * bi;
                double *kgg = Kgg + jj * bi;
#pragma omp simd
                for (int ii = 0; ii < bi; ii++) {
                    const double dx = xi[ii] - xj;
                    const double dy = yi[ii] - yj;
                    const double dz = zi[ii] - zj;
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i[ii] * r_ij2 + kappa_i[ii];
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_ij_1 = 1 / g_ij;
                    const double g_ji_1 = 1 / g_ji;
                    const double g_sum_1 = 1 / (g_ij + g_ji);
                    const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

                    const double rho_dgdrho_i = rho_i[ii] * (r_ij2 * domega_drho_i[ii] + dkappa_drho_i[ii]);
                    const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
                    const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_1 * g_ji_1);
                    const double wPhi = weight_j * Phi_ij;

                    krr[ii] = wPhi * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                      - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                      - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
                    kgr[ii] = wPhi * rho_domega_dgamma_i[ii] * r_ij2 * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
                    krg[ii] = wPhi * rho_domega_dgamma_j * r_ij2 * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
                    kgg[ii] = wPhi * rho_domega_dgamma_i[ii] * rho_domega_dgamma_j * r_ij2 * r_ij2 * d2Phi_dgij_dgji_over_Phi;
                }
            }
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Krr, &bi, rho_t   + j0, &ld, &D1, f_rho_t   + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Krg, &bi, gamma_t + j0, &ld, &D1, f_rho_t   + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Kgr, &bi, rho_t   + j0, &ld, &D1, f_gamma_t + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Kgg, &bi, gamma_t + j0, &ld, &D1, f_gamma_t + i0, &ld);
        }

        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            const double rho_i_ = rho[i];
            const double U_i = U[i];
            const double W_i = W[i];
            const double A_i = A[i];
            const double B_i = B[i];
            const double C_i = C[i];
            const double domega_drho_i_ = domega_drho[i];
            const double domega_dgamma_i = domega_dgamma[i];
            const double dkappa_drho_i_ = dkappa_drho[i];
            const double d2omega_drho2_i = d2omega_drho2[i];
            const double d2omega_dgamma2_i = d2omega_dgamma2[i];
            const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
            const double d2kappa_drho2_i = d2kappa_drho2[i];

            const double f_rho_rho_ii = 2 * domega_drho_i_ * W_i + 2 * dkappa_drho_i_ * U_i
                                        + rho_i_ * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i_ * dkappa_drho_i_ * A_i
                                                + domega_drho_i_ * domega_drho_i_ * C_i + 2 * domega_drho_i_ * dkappa_drho_i_ * B_i);
            const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i_ * (d2omega_drho_dgamma_i * W_i
                                                                        + domega_dgamma_i * (dkappa_drho_i_ * B_i + domega_drho_i_ * C_i));
            const double f_rho_gamma_ii = f_gamma_rho_ii;
            const double f_gamma_gamma_ii = rho_i_ * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

            for (int t = 0; t < nt; t++) {
                const double rho_t_i   =   rho_t[(size_t)t * ld + i];
                const double gamma_t_i = gamma_t[(size_t)t * ld + i];
                f_rho_t  [(size_t)t * ld + i] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
                f_gamma_t[(size_t)t * ld + i] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);
            }
        }
#pragma omp atomic
        _done += bi;
#pragma omp atomic
        _kept += nkept;
#pragma omp atomic
        _all += nall;
    }
    free(K);
}
    free(sph);
}

void VV10CUT_eval_EUW_with_weight1(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                   const int ngrids, const int nderivative)
{
    if (ngrids <= 0 || nderivative <= 0) {
        return;
    }
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double DM1 = -1;
    const int ld = ngrids;
    const int nd = nderivative;
    const size_t nblk = VV10_BI * VV10_BJ;
    double *sph = block_spheres(grid_coord, ngrids);
    _done = 0;
    _total = ngrids;
    _kept = 0;
    _all = 0;

#pragma omp parallel
{
    double *K = malloc(sizeof(double) * (3 * nblk + 5 * VV10_BI));
    double *KE = K;
    double *KU = KE + nblk;
    double *KW = KU + nblk;
    double *xi = KW + nblk;
    double *yi = xi + VV10_BI;
    double *zi = yi + VV10_BI;
    double *omega_i = zi + VV10_BI;
    double *kappa_i = omega_i + VV10_BI;

#pragma omp for schedule(dynamic, 1)
    for (int i0 = 0; i0 < ngrids; i0 += VV10_BI) {
        const int bi = MIN(VV10_BI, ngrids - i0);
        long nkept = 0, nall = 0;
        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            xi[ii] = grid_coord[i * 3 + 0];
            yi[ii] = grid_coord[i * 3 + 1];
            zi[ii] = grid_coord[i * 3 + 2];
            omega_i[ii] = omega[i];
            kappa_i[ii] = kappa[i];
        }
        for (int d = 0; d < nd; d++) {
            for (int ii = 0; ii < bi; ii++) {
                Ew[(size_t)d * ld + i0 + ii] = 0;
                Uw[(size_t)d * ld + i0 + ii] = 0;
                Ww[(size_t)d * ld + i0 + ii] = 0;
            }
        }

        for (int j00 = 0; j00 < ngrids; j00 += VV10_BJ) {
            int j0, bj;
            if (!kept_range(sph, i0 / VV10_BI, j00, MIN(VV10_BJ, ngrids - j00), &j0, &bj, &nkept, &nall)) {
                continue;
            }
            for (int jj = 0; jj < bj; jj++) {
                const int j = j0 + jj;
                const double xj = grid_coord[j * 3 + 0];
                const double yj = grid_coord[j * 3 + 1];
                const double zj = grid_coord[j * 3 + 2];
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double rho_j = rho[j];
                double *kE = KE + jj * bi;
                double *kU = KU + jj * bi;
                double *kW = KW + jj * bi;
#pragma omp simd
                for (int ii = 0; ii < bi; ii++) {
                    const double dx = xi[ii] - xj;
                    const double dy = yi[ii] - yj;
                    const double dz = zi[ii] - zj;
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i[ii] * r_ij2 + kappa_i[ii];
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_ij_1 = 1 / g_ij;
                    const double g_sum_1 = 1 / (g_ij + g_ji);
                    const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

                    const double E_ij = rho_j * Phi_ij;
                    const double U_ij = E_ij * (g_sum_1 + g_ij_1);
                    kE[ii] = E_ij;
                    kU[ii] = U_ij;
                    kW[ii] = U_ij * r_ij2;
                }
            }
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &D1 , KE, &bi, grid_weight1 + j0, &ld, &D1, Ew + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &DM1, KU, &bi, grid_weight1 + j0, &ld, &D1, Uw + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &DM1, KW, &bi, grid_weight1 + j0, &ld, &D1, Ww + i0, &ld);
        }
#pragma omp atomic
        _done += bi;
#pragma omp atomic
        _kept += nkept;
#pragma omp atomic
        _all += nall;
    }
    free(K);
}
    free(sph);
}
