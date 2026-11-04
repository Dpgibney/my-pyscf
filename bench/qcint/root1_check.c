/*
 * A vectorizable one-root Rys quadrature for int1e_grids (It25 prototype): t^2 = F1(x)/F0(x) and the weight F0(x),
 * against qcint's CINTrys_roots(1, x) (whose u gives t^2 = u/(u+1)) over x in [0, 80]: the largest relative
 * differences, by range of x, and the time per point of both. F0 and F1:
 *   x < XSERIES: F_m(x) = exp(-x) sum_k (2x)^k / ((2m+1)(2m+3)...(2m+2k+1)), positive terms (no cancellation);
 *   otherwise:   F0 = sqrt(pi/4x) erf(sqrt(x)), F1 = (F0 - exp(-x)) / 2x, which loses at most a few ulps there.
 * The loop evaluates both forms for every point and selects, so that it vectorizes (exp and erf from libmvec).
 *     root1_check LIBCINT.so
 * Build with qcint's flags and -fopenmp-simd: gcc -O2 -ffast-math -ftree-vectorize -fopenmp-simd -march=x86-64-v3 \
 *     root1_check.c -ldl -lm
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define XSERIES 1.
#define NSERIES 26
#define SQRTPIE4 .8862269254527580136

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

__attribute__((noinline))
void root1(double *restrict t2, double *restrict w, const double *restrict x, int n)
{
#pragma omp simd
        for (int i = 0; i < n; i++) {
                double xx = x[i];
                int small = xx < XSERIES;
                /* the series, on x clamped to its range */
                double xs = small ? xx : XSERIES;
                double s0 = 1, s1 = 1, term0 = 1, term1 = 1;
#pragma GCC unroll 32
                for (int k = 1; k < NSERIES; k++) {
                        term0 *= 2 * xs / (2 * k + 1);
                        term1 *= 2 * xs / (2 * k + 3);
                        s0 += term0;
                        s1 += term1;
                }
                double es = exp(-xs);
                double f0s = es * s0;
                double f1s = es * s1 / 3;
                /* the erf form, on x clamped to its range */
                double xl = small ? XSERIES : xx;
                double tt = sqrt(xl);
                double f0l = SQRTPIE4 / tt * erf(tt);
                double f1l = .5 / xl * (f0l - exp(-xl));
                double f0 = small ? f0s : f0l;
                double f1 = small ? f1s : f1l;
                t2[i] = f1 / f0;
                w[i] = f0;
        }
}

int main(int argc, char **argv)
{
        if (argc < 2) { fprintf(stderr, "usage: root1_check LIBCINT.so\n"); return 1; }
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        int (*roots)(int, double, double *, double *) = dlsym(lib, "CINTrys_roots");
        if (!roots) { fprintf(stderr, "no CINTrys_roots\n"); return 1; }
        const int n = 1 << 20;
        double *x = malloc(sizeof(double) * n), *t2 = malloc(sizeof(double) * n), *w = malloc(sizeof(double) * n);
        double *t2r = malloc(sizeof(double) * n), *wr = malloc(sizeof(double) * n);
        /* x spread over [0, 80]: half uniform, half log-uniform in [1e-12, 80] */
        srand(7);
        for (int i = 0; i < n; i++) {
                double r = (rand() + .5) / ((double)RAND_MAX + 1);
                x[i] = i % 2 ? 80 * r : exp(log(1e-12) + r * (log(80.) - log(1e-12)));
        }
        x[0] = 0;
        double t0 = now();
        for (int i = 0; i < n; i++) {
                double u, ww;
                roots(1, x[i], &u, &ww);
                t2r[i] = u / (u + 1);
                wr[i] = ww;
        }
        double tref = now() - t0;
        root1(t2, w, x, n);  /* warm-up */
        t0 = now();
        root1(t2, w, x, n);
        double tnew = now() - t0;
        const double edges[] = {0, 1e-6, 1e-3, .5, 1, 3, 10, 33, 40, 80.0001};
        const int nb = sizeof(edges) / sizeof(edges[0]) - 1;
        double dt[16] = {0}, dw[16] = {0};
        for (int i = 0; i < n; i++) {
                int b = 0;
                while (b < nb - 1 && x[i] >= edges[b + 1]) b++;
                double et = fabs(t2[i] - t2r[i]) / fabs(t2r[i]);
                double ew = fabs(w[i] - wr[i]) / fabs(wr[i]);
                if (et > dt[b]) dt[b] = et;
                if (ew > dw[b]) dw[b] = ew;
        }
        printf("%s: one root, %d points; CINTrys_roots %.2f ns per point, candidate %.2f ns (%.1fx)\n",
               argv[1], n, 1e9 * tref / n, 1e9 * tnew / n, tref / tnew);
        printf("   x range           max rel. diff t^2   max rel. diff weight\n");
        for (int b = 0; b < nb; b++) {
                printf("   [%-6g, %-6g)   %12.2e        %12.2e\n", edges[b], edges[b + 1], dt[b], dw[b]);
        }
        return 0;
}
