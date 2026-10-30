// dgemm throughput at the factored kernels' shapes: each OpenMP thread calls dgemm on its own buffers
// (BLAS single-threaded inside the parallel region). C[M x N] = A[M x K] B[K x N] col-major. N = 51 is one
// set's GEMM (caffeine/def2-SVP: nocc 51, about 150 AOs per group of 112 points); larger N batches sets.
// Build: gcc -O2 -fopenmp gemm_shapes.c -o gemm_shapes -L../../install/openblas/lib -lopenblas \
//            -Wl,-rpath,$PWD/../../install/openblas/lib
// Run:   OMP_NUM_THREADS=10 OMP_PROC_BIND=true OMP_PLACES=cores ./gemm_shapes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*,
            const int*, const double*, const int*, const double*, double*, const int*);
static double run(int M, int N, int K, const char *tb, long total_flops) {
        int nt = omp_get_max_threads();
        long ncall = total_flops / (2L * M * N * K) / nt + 1;
        double t0 = omp_get_wtime();
#pragma omp parallel
{
        double *a = malloc(sizeof(double) * M * K), *b = malloc(sizeof(double) * K * N), *c = malloc(sizeof(double) * M * N);
        for (long i = 0; i < (long)M * K; i++) a[i] = 1e-3 * (i % 97);
        for (long i = 0; i < (long)K * N; i++) b[i] = 1e-3 * (i % 89);
        const double one = 1, zero = 0;
        int ldb = tb[0] == 'T' ? N : K;
        for (long i = 0; i < ncall; i++)
                dgemm_("N", tb, &M, &N, &K, &one, a, &M, b, &ldb, &zero, c, &M);
        free(a); free(b); free(c);
}
        double t = omp_get_wtime() - t0;
        return 2.0 * M * N * K * ncall * nt / t / 1e9;
}
int main(int argc, char **argv) {
        int Ms[] = {112, 224};
        int Ns[] = {51, 102, 204, 408, 1020};
        for (int im = 0; im < 2; im++) {
                for (int in = 0; in < 5; in++) {
                        int M = Ms[im], N = Ns[in], K = 150;
                        run(M, N, K, "T", 2e10);
                        printf("M=%d N=%4d K=%d: %.0f GFLOP/s (B^T)  %.0f GFLOP/s (B)\n", M, N, K,
                               run(M, N, K, "T", 1e11), run(M, N, K, "N", 1e11));
                }
        }
        return 0;
}
