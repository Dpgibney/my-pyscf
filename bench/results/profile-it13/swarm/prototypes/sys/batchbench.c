// nset separate dgemm(T,N,ns,ns,224) sharing B (as VXCfused_dot_wv per set) vs one dgemm with M = nset*ns;
// and nset separate dgemm(N,T,224,ns,ns) sharing A (VXCfused_eval_rho per dm) vs one with N = nset*ns.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void (*dgemm_t)(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
int main(int argc, char **argv) {
        dgemm_t dgemm = (dgemm_t)dlsym(dlopen(argv[1], RTLD_NOW), "dgemm_");
        const double one = 1, zero = 0;
        int ng = 224;
        for (int ns = 100; ns <= 300; ns += 60) for (int nset = 4; nset <= 16; nset *= 2) {
                double *A = malloc(sizeof(double) * ng * ns * nset), *B = malloc(sizeof(double) * ng * ns), *C = malloc(sizeof(double) * ns * ns * nset);
                double *Dm = malloc(sizeof(double) * ns * ns * nset), *R = malloc(sizeof(double) * ng * ns * nset);
                for (long i = 0; i < (long)ng * ns * nset; i++) A[i] = 1e-2 * (i % 97) - .3;
                for (long i = 0; i < (long)ng * ns; i++) B[i] = 1e-2 * (i % 89) - .4;
                for (long i = 0; i < (long)ns * ns * nset; i++) Dm[i] = 1e-2 * (i % 83) - .4;
                double tsep = 1e9, tbat = 1e9, tsep2 = 1e9, tbat2 = 1e9;
                int M = ns * nset, N = ns * nset;
                for (int rep = 0; rep < 7; rep++) {
                        double t0 = now();
                        for (int r = 0; r < 20; r++) for (int s = 0; s < nset; s++)   // TN: A_s [ng x ns] (lda ng), B shared [ng x ns]
                                dgemm("T", "N", &ns, &ns, &ng, &one, A + (long)s * ng * ns, &ng, B, &ng, &zero, C + (long)s * ns * ns, &ns);
                        double t1 = now();
                        for (int r = 0; r < 20; r++)                                   // one call, M = nset*ns: A_all [ng x nset*ns]
                                dgemm("T", "N", &M, &ns, &ng, &one, A, &ng, B, &ng, &zero, C, &M);
                        double t2 = now();
                        for (int r = 0; r < 20; r++) for (int s = 0; s < nset; s++)   // NT: ao [ng x ns] shared, dm_s [ns x ns]
                                dgemm("N", "T", &ng, &ns, &ns, &one, B, &ng, Dm + (long)s * ns * ns, &ns, &zero, R + (long)s * ng * ns, &ng);
                        double t3 = now();
                        for (int r = 0; r < 20; r++)                                   // one call, N = nset*ns: dm_all [nset*ns x ns]
                                dgemm("N", "T", &ng, &N, &ns, &one, B, &ng, Dm, &N, &zero, R, &ng);
                        double t4 = now();
                        if (t1 - t0 < tsep) tsep = t1 - t0;
                        if (t2 - t1 < tbat) tbat = t2 - t1;
                        if (t3 - t2 < tsep2) tsep2 = t3 - t2;
                        if (t4 - t3 < tbat2) tbat2 = t4 - t3;
                }
                printf("ns %3d nset %2d: dot_wv TN separate %8.1f us batched %8.1f us (%.3fx) | rho NT separate %8.1f us batched %8.1f us (%.3fx)\n",
                       ns, nset, tsep / 20 * 1e6, tbat / 20 * 1e6, tsep / tbat, tsep2 / 20 * 1e6, tbat2 / 20 * 1e6, tsep2 / tbat2);
                free(A); free(B); free(C); free(Dm); free(R);
        }
        return 0;
}
