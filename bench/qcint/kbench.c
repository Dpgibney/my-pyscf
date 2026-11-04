/*
 * The SGX exchange contraction of PySCF (nrs2_ijg_gj_gi of vhf/nr_sgx_direct.c, the K of sgx_fit with pjs) on
 * caffeine-def2-SVP-like shell sizes, single thread: the current loops against the same updates tiled over the
 * grid points (KT points at a time, so that the rows of the shell pair stay in L1), which leaves the order of the
 * additions into every element unchanged. Checks that the two give the same bits and times them.
 *     gcc -O2 -msse3 kbench.c -o kbench                                  (PySCF's default flags)
 *     gcc -O3 -march=x86-64-v3 -ffp-contract=off kbench.c -o kbench_v3   (its NUMINT_X86_64_V3 kernel flags)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DK 224
#ifndef KT
#define KT 32
#endif

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* nrs2_ijg_gj_gi, ncomp = 1 */
__attribute__((noinline))
static void k_ref(double *eri, double *dm, double *data, int i0, int i1, int j0, int j1, int dk, size_t nk_global)
{
        int i, j, k;
        int ij = 0;
        for (j = j0; j < j1; j++) {
        for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[i*dk+k] += eri[ij*dk+k] * dm[j*nk_global+k];
                }
                for (k = 0; k < dk; k++) {
                        data[j*dk+k] += eri[ij*dk+k] * dm[i*nk_global+k];
                }
        } }
}

/* the same updates, KT grid points at a time */
__attribute__((noinline))
static void k_tiled(double *eri, double *dm, double *data, int i0, int i1, int j0, int j1, int dk, size_t nk_global)
{
        int i, j, k, k0, k1, ij;
        for (k0 = 0; k0 < dk; k0 += KT) {
                k1 = k0 + KT < dk ? k0 + KT : dk;
                ij = 0;
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        double *restrict di = data + i*dk;
                        double *restrict dj = data + j*dk;
                        const double *restrict e = eri + ij*dk;
                        const double *restrict mi = dm + i*nk_global;
                        const double *restrict mj = dm + j*nk_global;
                        for (k = k0; k < k1; k++) {
                                di[k] += e[k] * mj[k];
                                dj[k] += e[k] * mi[k];
                        }
                } }
        }
}

int main(int argc, char **argv)
{
        /* caffeine def2-SVP: per atom C/N/O 3s2p1d (14 AOs), H 2s1p (5 AOs); 14 heavy atoms, 10 H */
        int nsh = 0, dims[512], ao0[512], nao = 0;
        for (int a = 0; a < 24; a++) {
                int heavy = a < 14;
                int ls[6], nl = 0;
                if (heavy) { ls[nl++] = 0; ls[nl++] = 0; ls[nl++] = 0; ls[nl++] = 1; ls[nl++] = 1; ls[nl++] = 2; }
                else { ls[nl++] = 0; ls[nl++] = 0; ls[nl++] = 1; }
                for (int s = 0; s < nl; s++) { dims[nsh] = 2 * ls[s] + 1; ao0[nsh] = nao; nao += dims[nsh]; nsh++; }
        }
        ao0[nsh] = nao;
        size_t nk_global = 70928;
        int maxd = 5;
        double *dm = malloc(sizeof(double) * nao * nk_global);
        double *eri = malloc(sizeof(double) * maxd * maxd * DK);
        double *data_ref = calloc(nao * DK, sizeof(double)), *data_new = calloc(nao * DK, sizeof(double));
        srand(3);
        for (int n = 0; n < maxd * maxd * DK; n++) eri[n] = rand() / (double)RAND_MAX - .5;
        for (size_t n = 0; n < (size_t)nao * nk_global; n++) dm[n] = rand() / (double)RAND_MAX - .5;
        int nbatch = argc > 1 ? atoi(argv[1]) : 20;
        double tref = 0, tnew = 0, nint = 0;
        for (int b = 0; b < nbatch; b++) {
                size_t k0 = (size_t)b * DK * 3;
                for (int rep = 0; rep < 2; rep++) {
                        double *data = rep ? data_new : data_ref;
                        double t = now();
                        for (int ish = 0; ish < nsh; ish++) {
                        for (int jsh = 0; jsh < ish; jsh++) {
                                /* the integrals of this pair: any values, the same for both kernels */
                                int n = dims[ish] * dims[jsh] * DK;
                                for (int m = 0; m < n; m += 97) eri[m] = (ish * 31 + jsh * 7 + m) * 1e-3;
                                if (rep) {
                                        k_tiled(eri, dm + k0, data, ao0[ish], ao0[ish+1], ao0[jsh], ao0[jsh+1], DK, nk_global);
                                } else {
                                        k_ref(eri, dm + k0, data, ao0[ish], ao0[ish+1], ao0[jsh], ao0[jsh+1], DK, nk_global);
                                        nint += n;
                                }
                        } }
                        if (rep) tnew += now() - t; else tref += now() - t;
                }
        }
        long same = 0;
        for (int n = 0; n < nao * DK; n++) same += data_ref[n] == data_new[n];
        printf("nao %d, %d shells, %d batches: %.3e integrals; current %.3f s (%.2f ns/int), tiled KT=%d %.3f s "
               "(%.2f ns/int, %.2fx); %.4f%% of the results bitwise equal\n", nao, nsh, nbatch, nint, tref,
               1e9 * tref / nint, KT, tnew, 1e9 * tnew / nint, tref / tnew, 100. * same / (nao * DK));
        return 0;
}
