/*
 * qcint's CINTrys_roots_chunk (the roots of a SIMD chunk of points at once, sgx-exp branch) against CINTrys_roots
 * point by point, for one and two roots: the largest relative differences of the roots u and the weights w and
 * the fraction that is bitwise equal, by interval of x. Chunks mix intervals as grid batches do (mostly one).
 *     rchunk_check LIBCINT.so LANES        (LANES = SIMDD of the build: 4 for AVX2, 8 for AVX-512)
 *     gcc -O2 rchunk_check.c -o rchunk_check -ldl -lm
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
        if (argc < 3) { fprintf(stderr, "usage: rchunk_check LIBCINT.so LANES\n"); return 1; }
        void *lib = dlopen(argv[1], RTLD_NOW);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        int (*roots)(int, double, double *, double *) = dlsym(lib, "CINTrys_roots");
        int (*chunk)(int, double *, double *, double *) = dlsym(lib, "CINTrys_roots_chunk");
        if (!roots || !chunk) { fprintf(stderr, "no CINTrys_roots or CINTrys_roots_chunk\n"); return 1; }
        int L = atoi(argv[2]);
        const double edges[] = {0, 1e-6, 1, 3, 5, 10, 15, 33, 40, 45, 60};
        const int nb = 10;
        double *x = aligned_alloc(64, 64), *u = aligned_alloc(64, 128), *w = aligned_alloc(64, 128);
        for (int nroots = 1; nroots <= 2; nroots++) {
                double mu[16] = {0}, mw[16] = {0};
                long nbit[16] = {0}, ncnt[16] = {0}, nskip = 0;
                srand(5);
                for (int it = 0; it < 400000; it++) {
                        int b = it % nb;
                        for (int l = 0; l < L; l++) {
                                double r = (rand() + .5) / ((double)RAND_MAX + 1);
                                int bb = (l == 0 || rand() % 4) ? b : rand() % nb;
                                x[l] = edges[bb] + r * (edges[bb+1] - edges[bb]);
                                if (bb == 0) x[l] = 3.1e-7 + r * (1e-6 - 3.1e-7);
                        }
                        if (!chunk(nroots, x, u, w)) { nskip++; continue; }
                        for (int l = 0; l < L; l++) {
                                double ur[2], wr[2];
                                roots(nroots, x[l], ur, wr);
                                int bb = 0;
                                while (bb < nb - 1 && x[l] >= edges[bb+1]) bb++;
                                for (int i = 0; i < nroots; i++) {
                                        double du = fabs(u[i*L+l] - ur[i]) / fabs(ur[i]);
                                        double dw = fabs(w[i*L+l] - wr[i]) / fabs(wr[i]);
                                        if (du > mu[bb]) mu[bb] = du;
                                        if (dw > mw[bb]) mw[bb] = dw;
                                        nbit[bb] += (u[i*L+l] == ur[i]) + (w[i*L+l] == wr[i]);
                                        ncnt[bb] += 2;
                                }
                        }
                }
                printf("%d root(s), %d lanes (%ld chunks left to CINTrys_roots)\n", nroots, L, nskip);
                printf("  x range         max rel diff u   max rel diff w   bitwise equal\n");
                for (int b = 0; b < nb; b++) {
                        printf("  [%5g, %5g)   %12.2e     %12.2e     %6.2f%%\n", edges[b], edges[b+1], mu[b], mw[b],
                               ncnt[b] ? 100. * nbit[b] / ncnt[b] : 0);
                }
        }
        return 0;
}
