/*
 * int1e_grids_sph with the optimizer of int1e_grids_optimizer against the same library's per-call path (opt NULL),
 * on all shell pairs (i, j) of a molecule (both orders) and a few SGX batches of a grid: the fraction bitwise equal
 * and the largest difference. Then with env[PTR_RANGE_OMEGA] set to -0.3 and +0.3 after the optimizer was made for
 * omega = 0, where the optimizer must not be used (bitwise the per-call results).
 *     optcheck LIBCINT.so MOL.bin GRID.bin        (as gbench.c; gcc -O2 optcheck.c -o optcheck -ldl -lm)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SGX_BLKSIZE 224
#define PTR_RANGE_OMEGA 8
#define NGRIDS 11
#define PTR_GRIDS 12
#define ANG_OF 1
#define NCTR_OF 3
#define BAS_SLOTS 8
#define ATM_SLOTS 6

typedef int (*FINT)(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, void *opt, double *cache);
typedef void (*FOPT)(void **opt, int *atm, int natm, int *bas, int nbas, double *env);

int main(int argc, char **argv)
{
        if (argc < 4) { fprintf(stderr, "usage: optcheck LIB.so MOL.bin GRID.bin\n"); return 1; }
        FILE *f = fopen(argv[2], "rb");
        int hdr[3];
        if (!f || fread(hdr, sizeof(int), 3, f) != 3) return 1;
        int natm = hdr[0], nbas = hdr[1], nenv = hdr[2];
        int *atm = malloc(sizeof(int) * natm * ATM_SLOTS), *bas = malloc(sizeof(int) * nbas * BAS_SLOTS);
        if (fread(atm, sizeof(int), natm * ATM_SLOTS, f) != (size_t)natm * ATM_SLOTS ||
            fread(bas, sizeof(int), nbas * BAS_SLOTS, f) != (size_t)nbas * BAS_SLOTS) return 1;
        f = freopen(argv[3], "rb", f);
        int ngrids;
        if (!f || fread(&ngrids, sizeof(int), 1, f) != 1) return 1;
        double *env = malloc(sizeof(double) * (nenv + (size_t)ngrids * 3));
        FILE *fm = fopen(argv[2], "rb");
        fseek(fm, sizeof(int) * (3 + natm * ATM_SLOTS + nbas * BAS_SLOTS), SEEK_SET);
        if (fread(env, sizeof(double), nenv, fm) != (size_t)nenv) return 1;
        fclose(fm);
        if (fread(env + nenv, sizeof(double), (size_t)ngrids * 3, f) != (size_t)ngrids * 3) return 1;
        fclose(f);
        env[NGRIDS] = ngrids;
        env[PTR_GRIDS] = nenv;
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        FINT fint = (FINT)dlsym(lib, "int1e_grids_sph");
        FOPT fopt = (FOPT)dlsym(lib, "int1e_grids_optimizer");
        void (*fdel)(void **) = (void (*)(void **))dlsym(lib, "CINTdel_optimizer");
        void *opt = NULL;
        env[PTR_RANGE_OMEGA] = 0;
        fopt(&opt, atm, natm, bas, nbas, env);
        printf("%s: optimizer %s\n", argv[1], opt ? "made" : "NULL");
        double *cache = malloc(sizeof(double) << 22);
        double *b1 = malloc(sizeof(double) * 64 * 64 * SGX_BLKSIZE), *b2 = malloc(sizeof(double) * 64 * 64 * SGX_BLKSIZE);
        const double omegas[] = {0., -0.3, 0.3};
        int nbatch = (ngrids + SGX_BLKSIZE - 1) / SGX_BLKSIZE;
        for (int o = 0; o < 3; o++) {
                env[PTR_RANGE_OMEGA] = omegas[o];
                double nval = 0, maxd = 0, maxv = 0; long nbit = 0, nflag = 0;
                for (int b = 0; b < nbatch; b += nbatch / 6 + 1) {
                        int ig0 = b * SGX_BLKSIZE, ig1 = ig0 + SGX_BLKSIZE < ngrids ? ig0 + SGX_BLKSIZE : ngrids;
                        for (int i = 0; i < nbas; i++) {
                        for (int j = 0; j < nbas; j++) {
                                int di = (2 * bas[i*BAS_SLOTS+ANG_OF] + 1) * bas[i*BAS_SLOTS+NCTR_OF];
                                int dj = (2 * bas[j*BAS_SLOTS+ANG_OF] + 1) * bas[j*BAS_SLOTS+NCTR_OF];
                                int shls[4] = {i, j, ig0, ig1};
                                int dims[3] = {di, dj, ig1 - ig0};
                                int h1 = fint(b1, dims, shls, atm, natm, bas, nbas, env, opt, cache);
                                int h2 = fint(b2, dims, shls, atm, natm, bas, nbas, env, NULL, cache);
                                nflag += h1 != h2;
                                size_t n = (size_t)di * dj * (ig1 - ig0);
                                for (size_t k = 0; k < n; k++) {
                                        double d = fabs(b1[k] - b2[k]);
                                        nbit += b1[k] == b2[k];
                                        if (d > maxd) maxd = d;
                                        if (fabs(b2[k]) > maxv) maxv = fabs(b2[k]);
                                }
                                nval += n;
                        } }
                }
                printf("  omega %+.1f: %.0f integrals, %.4f%% bitwise equal, max |diff| %.2e (max |value| %.2e), "
                       "%ld calls with different nonzero flags\n", omegas[o], nval, 100. * nbit / nval, maxd, maxv, nflag);
        }
        if (opt && fdel) fdel(&opt);
        return 0;
}
