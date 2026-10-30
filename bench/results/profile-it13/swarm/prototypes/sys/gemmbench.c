// Single-thread timing of the BLAS calls of the fused XC kernels (nr_numint_fused.c) through
// the Fortran interface PySCF uses. Usage: gemmbench LIBBLAS [seconds_per_shape]
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

typedef void (*dgemm_t)(const char*, const char*, const int*, const int*, const int*,
                        const double*, const double*, const int*, const double*, const int*,
                        const double*, double*, const int*);
typedef void (*dsyrk_t)(const char*, const char*, const int*, const int*, const double*,
                        const double*, const int*, const double*, double*, const int*);

static double now(void) {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// FMA peak: 12 independent 256-bit FMA chains held in registers
__attribute__((target("avx2,fma"), noinline))
static double fma_peak(long iters) {
        __m256d b = _mm256_set1_pd(0.999999), c = _mm256_set1_pd(1e-7);
        __m256d a0 = _mm256_set1_pd(0), a1 = _mm256_set1_pd(1), a2 = _mm256_set1_pd(2), a3 = _mm256_set1_pd(3);
        __m256d a4 = _mm256_set1_pd(4), a5 = _mm256_set1_pd(5), a6 = _mm256_set1_pd(6), a7 = _mm256_set1_pd(7);
        __m256d a8 = _mm256_set1_pd(8), a9 = _mm256_set1_pd(9), a10 = _mm256_set1_pd(10), a11 = _mm256_set1_pd(11);
        double t0 = now();
        for (long it = 0; it < iters; it++) {
                a0 = _mm256_fmadd_pd(a0, b, c); a1 = _mm256_fmadd_pd(a1, b, c); a2 = _mm256_fmadd_pd(a2, b, c);
                a3 = _mm256_fmadd_pd(a3, b, c); a4 = _mm256_fmadd_pd(a4, b, c); a5 = _mm256_fmadd_pd(a5, b, c);
                a6 = _mm256_fmadd_pd(a6, b, c); a7 = _mm256_fmadd_pd(a7, b, c); a8 = _mm256_fmadd_pd(a8, b, c);
                a9 = _mm256_fmadd_pd(a9, b, c); a10 = _mm256_fmadd_pd(a10, b, c); a11 = _mm256_fmadd_pd(a11, b, c);
                __asm__ volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5));
                __asm__ volatile("" : "+x"(a6), "+x"(a7), "+x"(a8), "+x"(a9), "+x"(a10), "+x"(a11));
        }
        double t = now() - t0;
        __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)),
                    _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7)),
                                  _mm256_add_pd(_mm256_add_pd(a8, a9), _mm256_add_pd(a10, a11))));
        double out[4]; _mm256_storeu_pd(out, s);
        if (out[0] == 42.) printf("x");
        return iters * 12.0 * 8 / t / 1e9;  // GFLOP/s
}

static dgemm_t dgemm;
static dsyrk_t dsyrk;

static double run_gemm(char ta, char tb, int m, int n, int k, double secs) {
        int lda = (ta == 'N') ? m : k;
        int ldb = (tb == 'N') ? k : n;
        int ldc = m;
        size_t sa = (size_t)lda * ((ta == 'N') ? k : m);
        size_t sb = (size_t)ldb * ((tb == 'N') ? n : k);
        double *A = aligned_alloc(64, (sa + 8) * 8), *B = aligned_alloc(64, (sb + 8) * 8);
        double *C = aligned_alloc(64, ((size_t)ldc * n + 8) * 8);
        for (size_t i = 0; i < sa; i++) A[i] = (double)(i % 97) * 1e-2 - .3;
        for (size_t i = 0; i < sb; i++) B[i] = (double)(i % 89) * 1e-2 - .4;
        const double one = 1, zero = 0;
        // calibrate reps
        int reps = 1;
        double t;
        for (;;) {
                double t0 = now();
                for (int r = 0; r < reps; r++) dgemm(&ta, &tb, &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
                t = now() - t0;
                if (t > secs / 8) break;
                reps *= 2;
        }
        double best = 1e30;
        for (int b = 0; b < 7; b++) {
                double t0 = now();
                for (int r = 0; r < reps; r++) dgemm(&ta, &tb, &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
                t = (now() - t0) / reps;
                if (t < best) best = t;
        }
        free(A); free(B); free(C);
        return best;
}

static double run_syrk(int n, int k, double secs) {
        // dsyrk_('U','T', n, k, A[k x n] (lda=k)) as VXCfused_dot_wv
        int lda = k, ldc = n;
        double *A = aligned_alloc(64, ((size_t)k * n + 8) * 8), *C = aligned_alloc(64, ((size_t)n * n + 8) * 8);
        for (size_t i = 0; i < (size_t)k * n; i++) A[i] = (double)(i % 97) * 1e-2 - .3;
        const double one = 1, zero = 0;
        char u = 'U', tt = 'T';
        int reps = 1; double t;
        for (;;) {
                double t0 = now();
                for (int r = 0; r < reps; r++) dsyrk(&u, &tt, &n, &k, &one, A, &lda, &zero, C, &ldc);
                t = now() - t0;
                if (t > secs / 8) break;
                reps *= 2;
        }
        double best = 1e30;
        for (int b = 0; b < 7; b++) {
                double t0 = now();
                for (int r = 0; r < reps; r++) dsyrk(&u, &tt, &n, &k, &one, A, &lda, &zero, C, &ldc);
                t = (now() - t0) / reps;
                if (t < best) best = t;
        }
        free(A); free(C);
        return best;
}

int main(int argc, char **argv) {
        void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        dgemm = (dgemm_t)dlsym(h, "dgemm_");
        dsyrk = (dsyrk_t)dlsym(h, "dsyrk_");
        double secs = argc > 2 ? atof(argv[2]) : 0.3;
        const char *(*corename)(void) = (const char *(*)(void))dlsym(h, "openblas_get_corename");
        fma_peak(100000000); double peak0 = fma_peak(50000000);
        printf("core=%s fma_peak=%.1f GFLOP/s\n", corename ? corename() : "?", peak0);
        struct { char ta, tb; int m, n, k; const char *what; } S[] = {
                {'N','T', 224, 100, 100, "rho dm   NT 224 x ns x ns, ns=100 (benzene)"},
                {'N','T', 224, 160, 160, "rho dm   NT 224 x ns x ns, ns=160 (caffeine/alkane SVP)"},
                {'N','T', 224, 300, 300, "rho dm   NT 224 x ns x ns, ns=300 (caffeine TZVP)"},
                {'N','T', 224,  51, 160, "rho mo   NT 224 x nocc x ns, 51 x 160"},
                {'N','T', 224,  81, 160, "rho mo   NT 224 x nocc x ns, 81 x 160 (alkane)"},
                {'N','T', 224,  51, 300, "rho mo   NT 224 x nocc x ns, 51 x 300"},
                {'N','T', 224,  21, 100, "lowrank  NT 224 x nocc x ns, 21 x 100 (benzene)"},
                {'T','N', 100, 100, 224, "dot_wv   TN ns x ns x 224, ns=100"},
                {'T','N', 160, 160, 224, "dot_wv   TN ns x ns x 224, ns=160"},
                {'T','N', 300, 300, 224, "dot_wv   TN ns x ns x 224, ns=300"},
                {'T','N', 160, 160, 448, "dot_terms TN ns x ns x 448, ns=160"},
                {'T','N',  51, 160, 448, "dot_ao_mo TN nocc x ns x 448, 51 x 160"},
                {'T','N',  21, 100, 448, "dot_ao_mo TN nocc x ns x 448, 21 x 100"},
                {'N','N', 224, 160,  51, "zbuf     NN 224 x ns x nocc, 160 x 51"},
                {'N','N', 246, 4915, 246, "NPdgemm chunk NN 246 x 4915 x 246 (grid eval_rho, 1/10 of 50k)"},
        };
        int ns = sizeof(S) / sizeof(S[0]);
        for (int i = 0; i < ns; i++) {
                double t = run_gemm(S[i].ta, S[i].tb, S[i].m, S[i].n, S[i].k, secs);
                double gf = 2.0 * S[i].m * S[i].n * S[i].k / t / 1e9;
                printf("%-62s %9.2f us %7.1f GFLOP/s %5.1f%% of fma peak\n", S[i].what, t * 1e6, gf, 100 * gf / peak0);
        }
        int sk[][2] = {{100, 224}, {160, 224}, {300, 224}, {160, 672}};
        for (int i = 0; i < 4; i++) {
                double t = run_syrk(sk[i][0], sk[i][1], secs);
                double gf = 1.0 * sk[i][0] * (sk[i][0] + 1) * sk[i][1] / t / 1e9;
                printf("dsyrk U,T n=%d k=%-4d (useful flops n(n+1)k)%23s %9.2f us %7.1f GFLOP/s %5.1f%% of fma peak\n",
                       sk[i][0], sk[i][1], "", t * 1e6, gf, 100 * gf / peak0);
        }
        double peak1 = fma_peak(20000000);
        printf("fma_peak after=%.1f GFLOP/s\n", peak1);
        return 0;
}
