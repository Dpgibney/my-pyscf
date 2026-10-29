/* Shell-quartet statistics for a semi-direct ERI cache (semidirect_stats.py).
 * For the s8 quartets of CVHFnr_direct_jk_batch, counts the quartets and the
 * integrals (doubles) that pass CVHFnrs8_prescreen (mode 0) or
 * CVHFnrs8_vj_prescreen (mode 1) for each recorded dm_cond, and how many of
 * them a cache holding the quartets with q_ij*q_kl >= tau would cover.
 * Build: gcc -O2 -fopenmp -shared -fPIC -o semidirect_count.so semidirect_count.c
 */
#include <stdlib.h>
#include <string.h>

static int pass(const double *q, const double *dmc, size_t n, int mode, double cutoff,
                int i, int j, int k, int l)
{
        double qijkl = q[i*n+j] * q[k*n+l];
        if (!(qijkl > cutoff)) return 0;
        if (4*dmc[j*n+i]*qijkl > cutoff || 4*dmc[l*n+k]*qijkl > cutoff) return 1;
        if (mode == 1) return 0;
        return dmc[j*n+k]*qijkl > cutoff || dmc[j*n+l]*qijkl > cutoff ||
               dmc[i*n+k]*qijkl > cutoff || dmc[i*n+l]*qijkl > cutoff;
}

/* out layout (doubles):
 *   pass_n[ncall], pass_sz[ncall],
 *   cov_n[ncall*ntau], cov_sz[ncall*ntau],
 *   stored_n[ntau], stored_sz[ntau],
 *   union_n, union_sz, ucov_n[ntau], ucov_sz[ntau],
 *   schwarz_n, schwarz_sz
 */
void count_quartets(const double *q, const double *dmcs, int ncall, int nbas,
                    const int *nf, double cutoff, int mode,
                    const double *taus, int ntau, double *out)
{
        size_t n = nbas;
        size_t nout = 2*ncall + 2*ncall*ntau + 2*ntau + 2 + 2*ntau + 2;
        memset(out, 0, sizeof(double) * nout);
#pragma omp parallel
{
        double *acc = calloc(nout, sizeof(double));
        double *pass_n = acc;
        double *pass_sz = pass_n + ncall;
        double *cov_n = pass_sz + ncall;
        double *cov_sz = cov_n + ncall*ntau;
        double *stored_n = cov_sz + ncall*ntau;
        double *stored_sz = stored_n + ntau;
        double *union_n = stored_sz + ntau;
        double *union_sz = union_n + 1;
        double *ucov_n = union_sz + 1;
        double *ucov_sz = ucov_n + ntau;
        double *schwarz_n = ucov_sz + ntau;
        double *schwarz_sz = schwarz_n + 1;
        size_t npair = n * (n + 1) / 2;
        size_t ij;
#pragma omp for schedule(dynamic, 16)
        for (ij = 0; ij < npair; ij++) {
                int i = 0;
                while ((size_t)(i+1)*(i+2)/2 <= ij) i++;
                int j = ij - (size_t)i*(i+1)/2;
                int k, l, c, t, any;
                for (k = 0; k <= i; k++) {
                int l1 = k < i ? k + 1 : j + 1;
                for (l = 0; l < l1; l++) {
                        double qijkl = q[i*n+j] * q[k*n+l];
                        if (!(qijkl > cutoff)) continue;
                        double sz = (double)nf[i] * nf[j] * nf[k] * nf[l];
                        schwarz_n[0] += 1;
                        schwarz_sz[0] += sz;
                        for (t = 0; t < ntau; t++) {
                                if (qijkl >= taus[t]) {
                                        stored_n[t] += 1;
                                        stored_sz[t] += sz;
                                }
                        }
                        any = 0;
                        for (c = 0; c < ncall; c++) {
                                if (!pass(q, dmcs + c*n*n, n, mode, cutoff, i, j, k, l)) continue;
                                any = 1;
                                pass_n[c] += 1;
                                pass_sz[c] += sz;
                                for (t = 0; t < ntau; t++) {
                                        if (qijkl >= taus[t]) {
                                                cov_n[c*ntau+t] += 1;
                                                cov_sz[c*ntau+t] += sz;
                                        }
                                }
                        }
                        if (any) {
                                union_n[0] += 1;
                                union_sz[0] += sz;
                                for (t = 0; t < ntau; t++) {
                                        if (qijkl >= taus[t]) {
                                                ucov_n[t] += 1;
                                                ucov_sz[t] += sz;
                                        }
                                }
                        }
                } }
        }
#pragma omp critical
        {
                size_t m;
                for (m = 0; m < nout; m++) out[m] += acc[m];
        }
        free(acc);
}
}
